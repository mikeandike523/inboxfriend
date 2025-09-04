from flask import request, jsonify, Response
from sqlalchemy import select
import requests
from termcolor import colored
from googleapiclient.discovery import build
from datetime import datetime, timezone, timedelta

from app import Session, engine, get_current_user_creds, _map_pred, _ID2LABEL, Message, Classification
from gmail_stream import GmailMessageStream, GmailPreviewMessageStream


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

        before = None
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
                        .messages().get(userId="me", id=last_id, format="metadata")
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
                Session(engine).execute(
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


def emails_experiment_classify_marketing_newsletter_other():
    """Dry-run classification of emails into MARKETING/NEWSLETTER/OTHER using SetFit model."""
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
            probas_list = result.get("probabilities") or []
            classes_set = set(classes)
            for m, proba in zip(batch, probas_list):
                subject = m.get("subject") or ""
                snippet = m.get("snippet") or ""
                if proba is not None:
                    # flag deletion if any category probability exceeds threshold and is in target classes
                    high = [i for i, p in enumerate(proba) if p >= 0.95]
                    high_labels = { _ID2LABEL.get(i, str(i)).lower() for i in high }
                    intersect = high_labels & classes_set
                    if intersect:
                        # choose highest confidence among matching labels
                        conf_vals = [proba[i] for i in high if _ID2LABEL.get(i, str(i)).lower() in intersect]
                        max_conf = max(conf_vals)
                        label_desc = ", ".join(intersect)
                        if dry_run:
                            yield f"Would delete ({label_desc}, {max_conf:.2f}) | {subject}\n    {snippet}\n"
                        else:
                            gmail.users().messages().delete(userId="me", id=m["id"]).execute()
                            yield f"Deleted ({label_desc}, {max_conf:.2f}) | {subject}\n    {snippet}\n"
                        continue
                    # no deletion: show top prediction
                    top_idx = max(range(len(proba)), key=lambda i: proba[i])
                    pred = _map_pred(top_idx)
                    conf = proba[top_idx]
                    label = colored(pred, "green") if pred != "OTHER" else colored(pred, "cyan")
                    yield f"{label} ({conf:.2f}) | {subject}\n    {snippet}\n"
                else:
                    yield f"UNKNOWN | {subject}\n    {snippet}\n"

    return Response(generate(), mimetype="text/plain")


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

        before = None
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
                        .messages().get(userId="me", id=last_id, format="metadata")
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
                Session(engine).execute(
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


def emails_classify():
    data = request.json or {}
    required = ["id", "subject", "sender_name", "sender_email", "content", "category"]
    if not all(k in data for k in required):
        return jsonify({"error": "missing fields"}), 400

    with Session(engine) as s:
        creds, user_email = get_current_user_creds(s)
        gmail = build("gmail", "v1", credentials=creds)
        gmail.users().messages().delete(userId="me", id=data["id"]).execute()
        return jsonify({"ok": True})


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
