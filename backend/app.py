from __future__ import annotations
import json
from datetime import datetime, timezone
from urllib.parse import urlencode

from flask import Flask, request, jsonify, redirect, make_response
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
import requests

from config import Config
from models import Base, Token, Message, MarketingEmailClassification

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request as GoogleRequest
from googleapiclient.discovery import build

from gmail_stream import GmailMessageStream

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

        # List messages
        lst = gmail.users().messages().list(userId="me", maxResults=n).execute()
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
    if n <= 0 or n > 100:
        return jsonify({"error": "n must be 1..100"}), 400

    with Session(engine) as s:
        creds, user_email = get_current_user_creds(s)
        gmail = build("gmail", "v1", credentials=creds)
        stream = GmailMessageStream(gmail, batch_size=n)
        stream._next_page_token = page_token  # seed token from client
        messages, next_token = stream.next_batch()
        return jsonify({"messages": messages, "next_page_token": next_token})


@app.post("/emails/marketing")
def emails_marketing():
    data = request.json or {}
    required = ["id", "subject", "sender_name", "sender_email", "content", "is_marketing"]
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
                content=data.get("content"),
            )
            s.add(msg)
            s.flush()

        rec = MarketingEmailClassification(
            message_id=msg.id,
            is_marketing=bool(data.get("is_marketing")),
        )
        s.add(rec)

        if data.get("is_marketing"):
            gmail.users().messages().delete(userId="me", id=data["id"]).execute()

        s.commit()
        return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
