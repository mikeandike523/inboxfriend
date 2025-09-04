import re

BASE_URL = "http://localhost:5000"
ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
