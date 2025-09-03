from flask import request, jsonify, make_response

import requests
from googleapiclient.discovery import build

from app import app, build_flow, upsert_token, get_current_user_creds, Session, engine, Token


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
    """Exchange the code for tokens, store them, then show a simple success page."""
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
