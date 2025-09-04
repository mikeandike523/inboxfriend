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
            creds.expiry = token.expiry.astimezone(timezone.utc).replace(tzinfo=None)
        else:
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
    tokens = session.execute(select(Token)).scalars().all()
    if not tokens:
        raise ValueError("No authenticated user found. Please run 'login' first.")

    for token in tokens:
        try:
            creds = ensure_fresh_creds(session, token.user_id)
            return creds, token.user_id
        except Exception:
            continue

    raise ValueError("No valid credentials found. Please run 'login' first.")

import handlers

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
