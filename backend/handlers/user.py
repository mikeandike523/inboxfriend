from flask import jsonify
from googleapiclient.discovery import build

from app import Session, engine, get_current_user_creds

def me():
    with Session(engine) as s:
        creds, user_email = get_current_user_creds(s)
        # Use Gmail profile as a quick check
        gmail = build("gmail", "v1", credentials=creds)
        prof = gmail.users().getProfile(userId="me").execute()
    return jsonify({"email": prof.get("emailAddress"), "messagesTotal": prof.get("messagesTotal")})


def stats():
    """Return stats: total messages and inbox messages."""
    with Session(engine) as s:
        creds, user_email = get_current_user_creds(s)
        gmail = build("gmail", "v1", credentials=creds)
        prof = gmail.users().getProfile(userId="me").execute()
        total = prof.get("messagesTotal")
        inbox_label = gmail.users().labels().get(userId="me", id="INBOX").execute()
        inbox_total = inbox_label.get("messagesTotal")
    return jsonify({"total_messages": total, "inbox_messages": inbox_total})
