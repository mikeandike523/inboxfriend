from flask import jsonify

def root():
    return jsonify({"ok": True, "service": "inbox-backend"})
