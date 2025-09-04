from flask import request, jsonify
from sqlalchemy import select
from googleapiclient.discovery import build
from datetime import datetime, timezone, timedelta

from preamble import Session, engine, get_current_user_creds
from gmail_stream import GmailMessageStream, GmailPreviewMessageStream
from models import Message, Classification


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


def emails_declutter():
    return jsonify({"message": "This endpoint is not yet implemented."}), 500


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
