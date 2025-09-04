import os
import psutil

from cli_config import ANSI_ESCAPE

def is_running_in_git_bash():
    try:
        parent_process = psutil.Process(os.getppid())
        parent_process_name = parent_process.name()
        return "winpty-agent.exe" in parent_process_name or "bash.exe" in parent_process_name
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False

def pad_right(s: str, width: int) -> str:
    """Pad string s with spaces on the right to ensure its visible length is width."""
    stripped = ANSI_ESCAPE.sub('', s)
    pad_len = width - len(stripped)
    if pad_len <= 0:
        return s
    return s + ' ' * pad_len
