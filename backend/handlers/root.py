from flask import jsonify

from app import app


@app.get("/")
def root():
    return jsonify({"ok": True, "service": "inbox-backend"})
