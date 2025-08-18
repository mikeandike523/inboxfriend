from __future__ import annotations
from typing import List, Dict, Optional
import base64
from email.utils import parseaddr

from googleapiclient.discovery import Resource


def _decode_body(payload: dict) -> str:
    """Recursively extract the first text/plain or text/html part."""
    if not payload:
        return ""
    mime = payload.get("mimeType", "")
    data = payload.get("body", {}).get("data")
    if mime in {"text/plain", "text/html"} and data:
        try:
            return base64.urlsafe_b64decode(data).decode("utf-8")
        except Exception:
            return ""
    for part in payload.get("parts", []) or []:
        text = _decode_body(part)
        if text:
            return text
    return ""


class GmailMessageStream:
    """Fetch Gmail messages in batches, newest first."""

    def __init__(
        self,
        gmail: Resource,
        batch_size: int = 25,
        label_ids: Optional[List[str]] = None,
        before: Optional[str] = None,
    ):
        self.gmail = gmail
        self.batch_size = batch_size
        self.label_ids = label_ids or ["INBOX"]
        self._next_page_token: Optional[str] = None
        self.before = before

    def next_batch(self) -> tuple[List[Dict], Optional[str]]:
        params: Dict[str, object] = {
            "userId": "me",
            "maxResults": self.batch_size,
            "labelIds": self.label_ids,
        }
        if self._next_page_token:
            params["pageToken"] = self._next_page_token
        if self.before:
            params["q"] = f"before:{self.before}"

        res = self.gmail.users().messages().list(**params).execute()
        self._next_page_token = res.get("nextPageToken")

        messages: List[Dict] = []
        for m in res.get("messages", []):
            msg = (
                self.gmail.users()
                .messages()
                .get(userId="me", id=m["id"], format="full")
                .execute()
            )
            headers = {
                h["name"].lower(): h["value"]
                for h in msg.get("payload", {}).get("headers", [])
            }
            from_hdr = headers.get("from", "")
            name, email = parseaddr(from_hdr)

            # Determine if this message is part of a thread with multiple messages
            thread_id = msg.get("threadId")
            thread = (
                self.gmail.users()
                .threads()
                .get(userId="me", id=thread_id, fields="messages/id")
                .execute()
            )
            in_thread = len(thread.get("messages", [])) > 1

            messages.append(
                {
                    "id": msg.get("id"),
                    "subject": headers.get("subject"),
                    "sender_name": name or None,
                    "sender_email": email or None,
                    "date": headers.get("date"),
                    "thread": in_thread,
                    "snippet": msg.get("snippet"),
                    "content": _decode_body(msg.get("payload", {})),
                }
            )
        return messages, self._next_page_token


class GmailPreviewMessageStream:
    """Fetch Gmail messages using only the snippet/preview text."""

    def __init__(
        self,
        gmail: Resource,
        batch_size: int = 25,
        label_ids: Optional[List[str]] = None,
        before: Optional[str] = None,
    ):
        self.gmail = gmail
        self.batch_size = batch_size
        self.label_ids = label_ids or ["INBOX"]
        self._next_page_token: Optional[str] = None
        self.before = before

    def next_batch(self) -> tuple[List[Dict], Optional[str]]:
        params: Dict[str, object] = {
            "userId": "me",
            "maxResults": self.batch_size,
            "labelIds": self.label_ids,
        }
        if self._next_page_token:
            params["pageToken"] = self._next_page_token
        if self.before:
            params["q"] = f"before:{self.before}"

        res = self.gmail.users().messages().list(**params).execute()
        self._next_page_token = res.get("nextPageToken")

        messages: List[Dict] = []
        for m in res.get("messages", []):
            msg = (
                self.gmail.users()
                .messages()
                .get(
                    userId="me",
                    id=m["id"],
                    format="metadata",
                    metadataHeaders=["Subject", "From", "Date"],
                )
                .execute()
            )
            headers = {
                h["name"].lower(): h["value"]
                for h in msg.get("payload", {}).get("headers", [])
            }
            from_hdr = headers.get("from", "")
            name, email = parseaddr(from_hdr)

            # Determine if this message is part of a thread with multiple messages
            thread_id = msg.get("threadId")
            thread = (
                self.gmail.users()
                .threads()
                .get(userId="me", id=thread_id, fields="messages/id")
                .execute()
            )
            in_thread = len(thread.get("messages", [])) > 1

            snippet = msg.get("snippet")

            messages.append(
                {
                    "id": msg.get("id"),
                    "subject": headers.get("subject"),
                    "sender_name": name or None,
                    "sender_email": email or None,
                    "date": headers.get("date"),
                    "thread": in_thread,
                    "snippet": snippet,
                    # Use snippet as content for downstream classification
                    "content": snippet,
                }
            )

        return messages, self._next_page_token
