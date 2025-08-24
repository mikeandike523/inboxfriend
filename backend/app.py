from __future__ import annotations
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode

from flask import Flask, request, jsonify, redirect, make_response, Response
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
import os

import requests
from termcolor import colored

from config import Config
from models import Base, Token, Message, Classification

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request as GoogleRequest
from googleapiclient.discovery import build

from gmail_stream import GmailMessageStream, GmailPreviewMessageStream
import json
from pathlib import Path

# Load label metadata for SetFit model to map prediction indices to labels and provide categories
_model_dir = Path(__file__).parent / "setfit_email_category"
_label_metadata_path = _model_dir / "label_metadata.json"
try:
    with open(_label_metadata_path, 'r') as _f:
        _label_metadata = json.load(_f)
    _ID2LABEL = {int(k): v for k, v in _label_metadata.get("id2label", {}).items()}
    _MODEL_CATEGORIES = _label_metadata.get("categories", [])
except Exception:
    _ID2LABEL = {}
    _MODEL_CATEGORIES = []

def _map_pred(pred):
    """Map a raw prediction (int or digit string) to its label string via metadata."""
    if isinstance(pred, int):
        return _ID2LABEL.get(pred, str(pred))
    if isinstance(pred, str) and pred.isdigit():
        return _ID2LABEL.get(int(pred), pred)
    return pred

app = Flask(__name__)
app.config.from_object(Config)

# --- Database setup ---
print(app.config["DB_URL"])
engine = create_engine(app.config["DB_URL"], pool_pre_ping=True, future=True)
Base.metadata.create_all(engine)

# Build a client_config dict instead of client_secret.json
GOOGLE_CLIENT_CONFIG = {
    "web": {
        "client_id": app.config["GOOGLE_CLIENT_ID"],
        "client_secret": app.config["GOOGLE_CLIENT_SECRET"],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": [app.config["OAUTH_REDIRECT_URI"]],
    }
}

SCOPES = app.config["GOOGLE_SCOPES"]

# -----------------------------
# Helpers
# -----------------------------

def build_flow(state: str | None = None) -> Flow:
    return Flow.from_client_config(
        GOOGLE_CLIENT_CONFIG,
        scopes=SCOPES,
        redirect_uri=app.config["OAUTH_REDIRECT_URI"],
        state=state,
    )


def upsert_token(session: Session, user_id: str, creds: Credentials) -> None:
    # Persist tokens (store refresh if provided)
    expiry = creds.expiry if hasattr(creds, "expiry") else None
    token = session.get(Token, user_id)
    if token is None:
        token = Token(
            user_id=user_id,
            access_token=creds.token,
            refresh_token=getattr(creds, "refresh_token", None),
            token_type=getattr(creds, "token_type", None),
            scope=" ".join(SCOPES),
            expiry=expiry,
        )
        session.add(token)
    else:
        token.access_token = creds.token
        # Only update refresh_token if a new one is present
        token.refresh_token = getattr(creds, "refresh_token", token.refresh_token)
        token.token_type = getattr(creds, "token_type", token.token_type)
        token.scope = " ".join(SCOPES)
        token.expiry = expiry
    session.commit()


def load_creds(session: Session, user_id: str) -> Credentials | None:
    token = session.get(Token, user_id)
    if not token:
        return None
    creds = Credentials(
        token=token.access_token,
        refresh_token=token.refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=app.config["GOOGLE_CLIENT_ID"],
        client_secret=app.config["GOOGLE_CLIENT_SECRET"],
        scopes=SCOPES,
    )
    if token.expiry:
        # Google OAuth library expects timezone-naive datetime (assumes UTC)
        if token.expiry.tzinfo is not None:
            # If stored as timezone-aware, convert to naive UTC
            creds.expiry = token.expiry.astimezone(timezone.utc).replace(tzinfo=None)
        else:
            # If already naive, assume it's UTC and use as-is
            creds.expiry = token.expiry
    return creds


def ensure_fresh_creds(session: Session, user_id: str) -> Credentials:
    creds = load_creds(session, user_id)
    if creds is None:
        raise ValueError("No credentials for user. Please login.")
    if creds.expired and creds.refresh_token:
        creds.refresh(GoogleRequest())
        # Persist new access token (and possibly rotated refresh token)
        upsert_token(session, user_id, creds)
    return creds


def get_current_user_creds(session: Session) -> tuple[Credentials, str]:
    """Get credentials for the current user (assumes single-user app).
    Returns (credentials, user_email).
    """
    # Get all tokens from DB
    tokens = session.execute(select(Token)).scalars().all()
    if not tokens:
        raise ValueError("No authenticated user found. Please run 'login' first.")
    
    # For single-user app, try the first (and likely only) token
    # If multiple exist, we'll use the first valid one
    for token in tokens:
        try:
            creds = ensure_fresh_creds(session, token.user_id)
            return creds, token.user_id
        except Exception:
            continue
    
    raise ValueError("No valid credentials found. Please run 'login' first.")


# -----------------------------
# Routes
# -----------------------------

@app.get("/")
def root():
    return jsonify({"ok": True, "service": "inbox-backend"})


@app.get("/auth/login")
def auth_login():
    """Return an authorization URL. CLI prints it, user pastes in browser.
    Optional 'state' is passed through.
    """
    state = request.args.get("state")
    flow = build_flow(state=state)
    # Ensure we get a refresh token at least once
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes=True,
        prompt="consent",
    )
    auth_url = auth_url.replace("include_granted_scopes=True", "include_granted_scopes=true")
    return jsonify({"authorization_url": auth_url})


@app.get("/auth/callback")
def auth_callback():
    """Exchange the code for tokens, store them, then show a simple success page.
    """
    flow = build_flow()
    flow.fetch_token(authorization_response=request.url)

    creds = flow.credentials

    # Identify the user by Gmail profile (email)
    gmail = build("gmail", "v1", credentials=creds)
    prof = gmail.users().getProfile(userId="me").execute()
    user_email = prof.get("emailAddress")

    with Session(engine) as s:
        upsert_token(s, user_email, creds)

    html = f"""
    <html>
      <body>
        <h2>Login complete</h2>
        <p>Connected as: <strong>{user_email}</strong></p>
        <p>You can close this tab and return to your CLI.</p>
      </body>
    </html>
    """
    resp = make_response(html)
    return resp


@app.post("/auth/logout")
def auth_logout():
    with Session(engine) as s:
        try:
            creds, user_email = get_current_user_creds(s)
        except ValueError:
            return jsonify({"ok": True, "message": "No user to logout."})
        
        tok = s.get(Token, user_email)
        if not tok or not tok.refresh_token:
            # Nothing to revoke, but ensure DB is clean
            if tok:
                s.delete(tok)
                s.commit()
            return jsonify({"ok": True, "message": "No refresh token to revoke; credentials removed."})

        # Call Google's revocation endpoint
        revoke_resp = requests.post(
            "https://oauth2.googleapis.com/revoke",
            data={"token": tok.refresh_token},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )

        # Remove from DB regardless of revoke success to enforce local policy
        s.delete(tok)
        s.commit()

        return jsonify({
            "ok": True,
            "revocation_status": revoke_resp.status_code,
        })


@app.get("/me")
def me():
    with Session(engine) as s:
        creds, user_email = get_current_user_creds(s)
        # Use Gmail profile as a quick check
        gmail = build("gmail", "v1", credentials=creds)
        prof = gmail.users().getProfile(userId="me").execute()
        return jsonify({"email": prof.get("emailAddress"), "messagesTotal": prof.get("messagesTotal")})


@app.get("/emails/recent")
def emails_recent():
    n = int(request.args.get("n", 20))
    if n <= 0 or n > 100:
        return jsonify({"error": "n must be 1..100"}), 400

    with Session(engine) as s:
        creds, user_email = get_current_user_creds(s)
        gmail = build("gmail", "v1", credentials=creds)

        # List messages only from the inbox
        lst = (
            gmail.users()
            .messages()
            .list(userId="me", maxResults=n, labelIds=["INBOX"])
            .execute()
        )
        messages = lst.get("messages", [])

        out = []
        for m in messages:
            msg = gmail.users().messages().get(
                userId="me",
                id=m["id"],
                format="metadata",
                metadataHeaders=["From", "Subject"],
            ).execute()
            headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
            out.append({
                "id": msg.get("id"),
                "from": headers.get("from"),
                "subject": headers.get("subject"),
                "snippet": msg.get("snippet"),
                "internalDate": msg.get("internalDate"),
            })

        return jsonify({"count": len(out), "messages": out})


@app.get("/emails/stream")
def emails_stream():
    n = int(request.args.get("n", 25))
    page_token = request.args.get("page_token")
    skip_classified = request.args.get("skip_classified", "false").lower() == "true"
    use_before = request.args.get("use_before", "true").lower() == "true"
    if n <= 0 or n > 100:
        return jsonify({"error": "n must be 1..100"}), 400

    with Session(engine) as s:
        creds, user_email = get_current_user_creds(s)
        gmail = build("gmail", "v1", credentials=creds)

        before: str | None = None
        if skip_classified and use_before:
            last_id = s.execute(
                select(Message.gmail_id)
                .join(Classification)
                .order_by(Classification.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            if last_id:
                try:
                    msg = (
                        gmail.users()
                        .messages()
                        .get(userId="me", id=last_id, format="metadata")
                        .execute()
                    )
                    internal = msg.get("internalDate")
                    if internal:
                        dt = datetime.fromtimestamp(int(internal) / 1000, tz=timezone.utc)
                        before = (dt + timedelta(days=1)).strftime("%Y/%m/%d")
                except Exception:
                    before = None

        stream = GmailMessageStream(gmail, batch_size=n, before=before)
        stream._next_page_token = page_token  # seed token from client

        messages = []
        next_token = None
        while True:
            batch, next_token = stream.next_batch()
            if skip_classified and batch:
                ids = [m["id"] for m in batch]
                classified_ids = set(
                    s.execute(
                        select(Message.gmail_id)
                        .join(Classification)
                        .where(Message.gmail_id.in_(ids))
                    ).scalars()
                )
                batch = [m for m in batch if m["id"] not in classified_ids]
            if batch or not next_token:
                messages = batch
                break

        return jsonify({"messages": messages, "next_page_token": next_token})

@app.get("/emails/experiment-classify-marketing-newsletter-other")
def emails_experiment_classify_marketing_newsletter_other():
    """Dry-run classification of emails into MARKETING/NEWSLETTER/OTHER using SetFit model."""
    """Dry-run classification of emails into MARKETING/NEWSLETTER/OTHER via external model service."""
    model_server_url = app.config["MODEL_SERVER_URL"]
    # Prepare Gmail client stream
    with Session(engine) as s:
        creds, user_email = get_current_user_creds(s)
        gmail = build("gmail", "v1", credentials=creds)
    n = int(request.args.get("n", 25))
    stream = GmailPreviewMessageStream(gmail, batch_size=n)
    stream._next_page_token = request.args.get("page_token")

    def generate():
        while True:
            batch, next_token = stream.next_batch()
            if not batch:
                break
            # Prepare inputs for model server
            texts = [f"{m.get('subject') or ''} {m.get('snippet') or ''}".strip() for m in batch]
            resp = requests.post(
                f"{model_server_url}/predict", json={"texts": texts}, timeout=30
            )
            resp.raise_for_status()
            result = resp.json()
            preds = [_map_pred(p) for p in result.get("predictions", [])]
            probas = result.get("probabilities") or [None] * len(preds)
            for m, pred, proba in zip(batch, preds, probas):
                conf = float(max(proba)) if proba is not None else None
                label = colored(pred, "green") if pred != "OTHER" else colored(pred, "cyan")
                subject = m.get("subject") or ""
                snippet = m.get("snippet") or ""
                if conf is not None:
                    yield f"{label} ({conf:.2f}) | {subject}\n    {snippet}\n"
                else:
                    yield f"{label} | {subject}\n    {snippet}\n"

    return Response(generate(), mimetype="text/plain")


@app.get("/emails/declutter")
def emails_declutter():
    """Preview and delete clutter emails based on specified classes. Supports dry-run mode (no deletions)."""
    model_server_url = app.config["MODEL_SERVER_URL"]
    with Session(engine) as s:
        creds, user_email = get_current_user_creds(s)
        gmail = build("gmail", "v1", credentials=creds)
    n = int(request.args.get("n", 25))
    classes = [c.lower() for c in request.args.getlist("classes")] or ["marketing", "newsletter", "notification"]
    dry_run = request.args.get("dry_run", "false").lower() == "true"
    before_this_year = request.args.get("before_this_year", "false").lower() == "true"
    # Restrict to messages before the start of the current year if requested
    before = None
    if before_this_year:
        now = datetime.now(timezone.utc)
        start = datetime(now.year, 1, 1, tzinfo=timezone.utc)
        before = start.strftime("%Y/%m/%d")
    stream = GmailPreviewMessageStream(gmail, batch_size=n, before=before)
    stream._next_page_token = request.args.get("page_token")

    def generate():
        while True:
            batch, next_token = stream.next_batch()
            if not batch:
                break
            # Classify batch via external model service
            texts = [f"Subject: {m.get('subject') or ''}\nBody:\n{m.get('snippet') or ''}".strip() for m in batch]
            resp = requests.post(f"{model_server_url}/predict", json={"texts": texts}, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            preds = [_map_pred(p) for p in result.get("predictions", [])]
            probas = result.get("probabilities") or [None] * len(preds)
            classes_set = set(classes)
            for m, pred, proba in zip(batch, preds, probas):
                conf = float(max(proba)) if proba is not None else None
                subject = m.get("subject") or ""
                snippet = m.get("snippet") or ""
                # Delete only specified classes with sufficient confidence
                if conf is not None and conf >= 0.95 and pred in classes_set:
                    if dry_run:
                        yield f"Would delete ({pred}, {conf:.2f}) | {subject}\n    {snippet}\n"
                    else:
                        gmail.users().messages().delete(userId="me", id=m["id"]).execute()
                        yield f"Deleted ({pred}, {conf:.2f}) | {subject}\n    {snippet}\n"
                else:
                    label = colored(pred, "green") if pred != "OTHER" else colored(pred, "cyan")
                    if conf is not None:
                        yield f"{label} ({conf:.2f}) | {subject}\n    {snippet}\n"
                    else:
                        yield f"{label} | {subject}\n    {snippet}\n"

    return Response(generate(), mimetype="text/plain")


@app.get("/emails/stream-preview")
def emails_stream_preview():
    n = int(request.args.get("n", 25))
    page_token = request.args.get("page_token")
    skip_classified = request.args.get("skip_classified", "false").lower() == "true"
    use_before = request.args.get("use_before", "true").lower() == "true"
    if n <= 0 or n > 100:
        return jsonify({"error": "n must be 1..100"}), 400

    with Session(engine) as s:
        creds, user_email = get_current_user_creds(s)
        gmail = build("gmail", "v1", credentials=creds)

        before: str | None = None
        if skip_classified and use_before:
            last_id = s.execute(
                select(Message.gmail_id)
                .join(Classification)
                .order_by(Classification.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            if last_id:
                try:
                    msg = (
                        gmail.users()
                        .messages()
                        .get(userId="me", id=last_id, format="metadata")
                        .execute()
                    )
                    internal = msg.get("internalDate")
                    if internal:
                        dt = datetime.fromtimestamp(int(internal) / 1000, tz=timezone.utc)
                        before = (dt + timedelta(days=1)).strftime("%Y/%m/%d")
                except Exception:
                    before = None

        stream = GmailPreviewMessageStream(gmail, batch_size=n, before=before)
        stream._next_page_token = page_token  # seed token from client

        messages = []
        next_token = None
        while True:
            batch, next_token = stream.next_batch()
            if skip_classified and batch:
                ids = [m["id"] for m in batch]
                classified_ids = set(
                    s.execute(
                        select(Message.gmail_id)
                        .join(Classification)
                        .where(Message.gmail_id.in_(ids))
                    ).scalars()
                )
                batch = [m for m in batch if m["id"] not in classified_ids]
            if batch or not next_token:
                messages = batch
                break

        return jsonify({"messages": messages, "next_page_token": next_token})


@app.get("/categories")
def get_categories():
    """Return available categories: prefer trained model categories if available."""
    if _MODEL_CATEGORIES:
        cats = _MODEL_CATEGORIES
    else:
        with Session(engine) as s:
            cats = (
                s.execute(select(Classification.category).distinct()).scalars().all()
            )
    return jsonify({"categories": cats})


@app.post("/emails/classify")
def emails_classify():
    data = request.json or {}
    required = ["id", "subject", "sender_name", "sender_email", "content", "category"]
    if not all(k in data for k in required):
        return jsonify({"error": "missing fields"}), 400

    with Session(engine) as s:
        creds, user_email = get_current_user_creds(s)
        gmail = build("gmail", "v1", credentials=creds)

        msg = s.execute(
            select(Message).where(Message.gmail_id == data["id"])
        ).scalar_one_or_none()

        if msg is None:
            msg = Message(
                gmail_id=data["id"],
                subject=data.get("subject"),
                sender_name=data.get("sender_name"),
                sender_email=data.get("sender_email"),
                content=(data.get("content", "")[:1024]),
            )
            s.add(msg)
            s.flush()

        existing_classification = s.execute(
            select(Classification).where(Classification.message_id == msg.id)
        ).scalar_one_or_none()

        if existing_classification:
            return jsonify({"ok": True, "message": "Email already classified", "skipped": True})

        rec = Classification(
            message_id=msg.id,
            category=data.get("category")[:128],
        )
        s.add(rec)

        if data.get("delete"):
            gmail.users().messages().delete(userId="me", id=data["id"]).execute()

        s.commit()
        return jsonify({"ok": True})


@app.post("/emails/delete")
def emails_delete():
    data = request.json or {}
    if "id" not in data:
        return jsonify({"error": "missing fields"}), 400

    with Session(engine) as s:
        creds, user_email = get_current_user_creds(s)
        gmail = build("gmail", "v1", credentials=creds)
        gmail.users().messages().delete(userId="me", id=data["id"]).execute()
        return jsonify({"ok": True})


@app.post("/emails/move")
def emails_move():
    data = request.json or {}
    if "id" not in data or "label" not in data:
        return jsonify({"error": "missing fields"}), 400

    label_name = data.get("label", "").strip()
    if not label_name:
        return jsonify({"error": "missing label"}), 400

    with Session(engine) as s:
        creds, user_email = get_current_user_creds(s)
        gmail = build("gmail", "v1", credentials=creds)
        # Verify label exists
        labels_resp = gmail.users().labels().list(userId="me").execute()
        labels = labels_resp.get("labels", []) or []
        match = next((lbl for lbl in labels if lbl.get("name", "").lower() == label_name.lower()), None)
        if not match:
            return jsonify({"error": f"label '{label_name}' not found"}), 400

        # Move message out of INBOX into the given label
        gmail.users().messages().modify(
            userId="me",
            id=data["id"],
            body={"removeLabelIds": ["INBOX"], "addLabelIds": [match.get("id")]},
        ).execute()
        return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
