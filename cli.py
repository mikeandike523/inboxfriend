import argparse
import sys
import time
import webbrowser
import requests

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
            params = {"n": args.n}
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
        print("\nFrom: {} <{}>".format(msg.get("sender_name"), msg.get("sender_email")))
        print("Subject:", msg.get("subject"))
        print(msg.get("content", "")[:500])
        choice = input("1.marketing 2.not 3.skip 4.quit > ")
        if choice in {"1", "2"}:
            msg["is_marketing"] = choice == "1"
            requests.post(f"{BASE_URL}/emails/marketing", json=msg).raise_for_status()
            if choice == "1":
                print("Marked as marketing and deleted.")
            else:
                print("Marked as not marketing.")
        elif choice == "4":
            print("Quitting.")
            return
        else:
            # skip just move on
            pass


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

    sub_classify = sub.add_parser("classify", help="Interactive marketing classifier")
    sub_classify.add_argument("-n", type=int, default=25)
    sub_classify.set_defaults(func=cmd_classify)

    args = p.parse_args()
    if not hasattr(args, "func"):
        p.print_help()
        return
    args.func(args)

if __name__ == "__main__":
    main()
