from config import Config
from flask import Flask

app = Flask(__name__)
app.config.from_object(Config)

def register_handler(route, handler, methods=("GET",)):
    """Register a view function under `route` with given HTTP methods."""
    app.add_url_rule(route, endpoint=handler.__name__, view_func=handler, methods=list(methods))

# Explicitly import handlers and register routes
from handlers.auth import auth_login, auth_callback, auth_logout
from handlers.user import me, stats
from handlers.emails import (
    emails_recent,
    emails_stream,
    emails_declutter,
    emails_stream_preview,
    emails_classify,
    emails_move,
)
from handlers.categories import get_categories

register_handler("/auth/login", auth_login)
register_handler("/auth/callback", auth_callback)
register_handler("/auth/logout", auth_logout, methods=("POST",))
register_handler("/me", me)
register_handler("/stats", stats)
register_handler("/emails/recent", emails_recent)
register_handler("/emails/stream", emails_stream)
register_handler("/emails/declutter", emails_declutter)
register_handler("/emails/stream-preview", emails_stream_preview)
register_handler("/emails/classify", emails_classify, methods=("POST",))
register_handler("/emails/move", emails_move, methods=("POST",))
register_handler("/categories", get_categories)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
