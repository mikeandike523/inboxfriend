import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev")
    DB_URL = os.getenv("DB_URL")
    DB_URL_EXTERNAL = os.getenv("DB_URL_EXTERNAL")
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
    OAUTH_REDIRECT_URI = os.getenv("OAUTH_REDIRECT_URI", "http://localhost:5000/auth/callback")
    GOOGLE_SCOPES = os.getenv("GOOGLE_SCOPES", "").split()