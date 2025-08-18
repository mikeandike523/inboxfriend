import argparse
import os
import shutil
import webbrowser
import requests
import psutil
import textwrap

def is_running_in_git_bash():
    try:
        parent_process = psutil.Process(os.getppid())
        parent_process_name = parent_process.name()
        return "winpty-agent.exe" in parent_process_name or "bash.exe" in parent_process_name
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


BASE_URL = "http://localhost:5000"


def cmd_login(args):
    r = requests.get(f"{BASE_URL}/auth/login")
    r.raise_for_status()
    url = r.json()["authorization_url"]
    print("\nPaste this URL into your browser to sign in:")
    print(url)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    print("\nAfter completing the flow, run `python cli.py me` or `recent-emails`.\n")


def cmd_logout(args):
    r = requests.post(f"{BASE_URL}/auth/logout")
    print(r.status_code, r.json())


def cmd_me(args):
    r = requests.get(f"{BASE_URL}/me")
    print(r.status_code, r.json())


def cmd_recent_emails(args):
    r = requests.get(f"{BASE_URL}/emails/recent", params={"n": args.n})
    r.raise_for_status()
    data = r.json()
    for i, m in enumerate(data.get("messages", []), 1):
        print(f"{i:2d}. {m.get('from')} | {m.get('subject')}\n    {m.get('snippet')}\n")


def cmd_classify(args):
    page_token = None
    batch = []
    while True:
        if not batch:
            params = {"n": args.n, "skip_classified": "true"}
            if page_token:
                params["page_token"] = page_token
            r = requests.get(f"{BASE_URL}/emails/stream", params=params)
            r.raise_for_status()
            data = r.json()
            batch = data.get("messages", [])
            page_token = data.get("next_page_token")
            if not batch:
                print("No more messages.")
                return

        msg = batch.pop(0)
        
        os.system("cls" if os.name == "nt" and not is_running_in_git_bash() else "clear")
        
        # Get terminal dimensions
        cols, rows = shutil.get_terminal_size((80, 20))
        max_lines = int(rows * 0.8)
        
        # Prepare header lines with proper wrapping
        from_line = "From: {} <{}>".format(msg.get("sender_name"), msg.get("sender_email"))
        subject_line = "Subject: " + str(msg.get("subject"))
        
        # Wrap header lines to terminal width
        wrapped_lines = []
        wrapped_lines.extend(textwrap.wrap(from_line, width=cols))
        wrapped_lines.extend(textwrap.wrap(subject_line, width=cols))
        
        # Calculate remaining lines for content
        available_lines = max_lines - len(wrapped_lines)
        
        # Process content lines with wrapping
        content = msg.get("content", "") or ""
        content_lines = content.splitlines()
        
        for content_line in content_lines:
            if available_lines <= 0:
                break
            
            if not content_line.strip():
                # Empty line
                wrapped_lines.append("")
                available_lines -= 1
            else:
                # Wrap the content line
                wrapped_content = textwrap.wrap(content_line, width=cols)
                if not wrapped_content:
                    wrapped_content = [""]
                
                # Check if we have enough space for this wrapped content
                if len(wrapped_content) <= available_lines:
                    wrapped_lines.extend(wrapped_content)
                    available_lines -= len(wrapped_content)
                else:
                    # Add as many lines as we can fit
                    wrapped_lines.extend(wrapped_content[:available_lines])
                    available_lines = 0
                    break
        
        print("\n".join(wrapped_lines))

        resp = input("> ").strip()
        if not resp or resp.lower() in {"n", "next", "skip", "s"}:
            continue
        if resp.lower() in {"q", "quit"}:
            print("Quitting.")
            return
        delete = False
        if resp.endswith(" DELETE"):
            delete = True
            resp = resp[:-7].strip()
        category = resp
        if not category:
            continue
        payload = {
            "id": msg["id"],
            "subject": msg.get("subject"),
            "sender_name": msg.get("sender_name"),
            "sender_email": msg.get("sender_email"),
            "content": msg.get("content"),
            "category": category,
            "delete": delete,
        }
        requests.post(f"{BASE_URL}/emails/classify", json=payload).raise_for_status()
        if delete:
            print(f"Categorized as {category} and deleted.")
        else:
            print(f"Categorized as {category}.")


def main():
    p = argparse.ArgumentParser(description="Inbox Tool CLI")
    sub = p.add_subparsers(dest="cmd")

    sub_login = sub.add_parser("login", help="Begin OAuth flow; prints a URL")
    sub_login.set_defaults(func=cmd_login)

    sub_logout = sub.add_parser("logout", help="Revoke and delete tokens")
    sub_logout.set_defaults(func=cmd_logout)

    sub_me = sub.add_parser("me", help="Show profile info; also refreshes token if expired")
    sub_me.set_defaults(func=cmd_me)

    sub_recent = sub.add_parser("recent-emails", help="List recent emails (sender, subject, snippet)")
    sub_recent.add_argument("-n", type=int, default=20)
    sub_recent.set_defaults(func=cmd_recent_emails)

    sub_classify = sub.add_parser("classify", help="Interactive email classifier")
    sub_classify.add_argument("-n", type=int, default=25)
    sub_classify.set_defaults(func=cmd_classify)

    args = p.parse_args()
    if not hasattr(args, "func"):
        p.print_help()
        return
    args.func(args)

if __name__ == "__main__":
    main()