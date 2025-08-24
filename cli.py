import json
import os
import re
import shutil
import webbrowser
import requests
import psutil
import textwrap
import click
from termcolor import colored
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
import datetime
from email.utils import parsedate_to_datetime

def is_running_in_git_bash():
    try:
        parent_process = psutil.Process(os.getppid())
        parent_process_name = parent_process.name()
        return "winpty-agent.exe" in parent_process_name or "bash.exe" in parent_process_name
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


BASE_URL = "http://localhost:5000"
ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

def _pad_right(s: str, width: int) -> str:
    """Pad string s with spaces on the right to ensure its visible length is width."""
    # strip ANSI escape sequences for length calculation
    stripped = ANSI_ESCAPE.sub('', s)
    pad_len = width - len(stripped)
    if pad_len <= 0:
        return s
    return s + ' ' * pad_len

def _interactive_classify(n, preview=False):
    page_token = None
    batch = []
    r = requests.get(f"{BASE_URL}/categories")
    r.raise_for_status()
    categories = r.json().get("categories", [])
    category_set = set(c.lower() for c in categories)
    completer = WordCompleter(categories, ignore_case=True)
    while True:
        if not batch:
            click.echo("Loading more emails...")
            params = {"n": n, "skip_classified": "true"}
            if page_token:
                params["page_token"] = page_token
            endpoint = "emails/stream-preview" if preview else "emails/stream"
            r = requests.get(f"{BASE_URL}/{endpoint}", params=params)
            r.raise_for_status()
            data = r.json()
            batch = data.get("messages", [])
            page_token = data.get("next_page_token")
            if not batch:
                click.echo("No more messages.")
                return

        msg = batch.pop(0)

        os.system("cls" if os.name == "nt" and not is_running_in_git_bash() else "clear")

        cols, rows = shutil.get_terminal_size((80, 20))
        max_lines = int(rows * 0.8)

        from_line = "From: {} <{}>".format(msg.get("sender_name"), msg.get("sender_email"))
        subject_line = "Subject: " + str(msg.get("subject"))
        date_line = "Date: " + str(msg.get("date"))
        thread_line = "Thread: " + ("yes" if msg.get("thread") else "no")

        left_width = int(cols * 0.75)
        right_width = cols - left_width - 1

        left_lines = []
        left_lines.extend(textwrap.wrap(date_line, width=left_width))
        left_lines.extend(textwrap.wrap(from_line, width=left_width))
        left_lines.extend(textwrap.wrap(subject_line, width=left_width))
        left_lines.extend(textwrap.wrap(thread_line, width=left_width))

        available_lines = max_lines - len(left_lines)
        content = msg.get("content", "") or ""
        for content_line in content.splitlines():
            if available_lines <= 0:
                break
            if not content_line.strip():
                left_lines.append("")
                available_lines -= 1
            else:
                wrapped_content = textwrap.wrap(content_line, width=left_width)
                if not wrapped_content:
                    wrapped_content = [""]
                if len(wrapped_content) <= available_lines:
                    left_lines.extend(wrapped_content)
                    available_lines -= len(wrapped_content)
                else:
                    left_lines.extend(wrapped_content[:available_lines])
                    available_lines = 0
                    break

        right_lines = []
        for cat in categories:
            if len(cat) <= right_width:
                right_lines.append(cat)
            else:
                right_lines.extend(textwrap.wrap(cat, width=right_width))

        total_lines = min(max(len(left_lines), len(right_lines)), max_lines)
        for i in range(total_lines):
            left = left_lines[i] if i < len(left_lines) else ""
            right = right_lines[i] if i < len(right_lines) else ""
            click.echo(_pad_right(left, left_width) + ' ' + _pad_right(right, right_width))

        while True:
            resp = prompt("> ", completer=completer).strip()
            if not resp or resp.lower() in {"n", "next", "skip", "s"}:
                break
            if resp.lower() in {"q", "quit", "e", "end", "x", "exit", "c", "close", "a", "abort"}:
                click.echo("Quitting.")
                return
            # Immediate delete or move command takes precedence over categorization
            lresp = resp.lower()
            if lresp.startswith("move "):
                label = resp[5:].strip()
                if not label:
                    continue
                rmove = requests.post(
                    f"{BASE_URL}/emails/move",
                    json={"id": msg["id"], "label": label},
                )
                if rmove.status_code != 200:
                    click.echo(colored(f"Unknown label: {label}", "yellow"))
                else:
                    click.echo(f"Moved to {label}.")
                break
            if lresp == "delete":
                requests.post(f"{BASE_URL}/emails/delete", json={"id": msg["id"]}).raise_for_status()
                click.echo("Deleted.")
                break
            # Check for delete or move suffix after categorization
            delete = False
            move_label = None
            # suffix delete
            if resp.upper().endswith(" DELETE"):
                delete = True
                resp = resp[:-7].strip()
            # suffix move
            suffix_match = re.match(r"^(.*)\s+MOVE\s+(.+)$", resp, flags=re.IGNORECASE)
            if suffix_match:
                resp = suffix_match.group(1).strip()
                move_label = suffix_match.group(2).strip()
            category = resp
            if not category:
                continue
            if category.lower() not in category_set:
                if category.startswith("! "):
                    category = category[2:].strip()
                    if not category:
                        continue
                    categories.append(category)
                    category_set.add(category.lower())
                    completer = WordCompleter(categories, ignore_case=True)
                else:
                    click.echo(
                        colored(
                            "Unknown category. To force-add a category, type a ! followed by a space, and then the category.",
                            "yellow",
                        )
                    )
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
                click.echo(f"Categorized as {category} and deleted.")
            else:
                click.echo(f"Categorized as {category}.")
            # perform move if requested after categorization
            if move_label:
                rmove = requests.post(
                    f"{BASE_URL}/emails/move",
                    json={"id": msg["id"], "label": move_label},
                )
                if rmove.status_code != 200:
                    click.echo(colored(f"Unknown label: {move_label}", "yellow"))
                    continue
                else:
                    click.echo(f"Moved to {move_label}.")
            break


@click.group()
def cli():
    """Inbox Tool CLI"""
    pass


@cli.command()
def login():
    """Begin OAuth flow; prints a URL"""
    r = requests.get(f"{BASE_URL}/auth/login")
    r.raise_for_status()
    url = r.json()["authorization_url"]
    click.echo("\nPaste this URL into your browser to sign in:")
    click.echo(url)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    click.echo("\nAfter completing the flow, run `python cli.py me` or `recent-emails`.\n")


@cli.command()
def logout():
    """Revoke and delete tokens"""
    r = requests.post(f"{BASE_URL}/auth/logout")
    click.echo(f"{r.status_code} {r.json()}")


@cli.command()
def me():
    """Show profile info; also refreshes token if expired"""
    r = requests.get(f"{BASE_URL}/me")
    click.echo(f"{r.status_code} {r.json()}")


@cli.command("recent-emails")
@click.option("-n", default=20, help="Number of emails to fetch")
def recent_emails(n):
    """List recent emails (sender, subject, snippet)"""
    r = requests.get(f"{BASE_URL}/emails/recent", params={"n": n})
    r.raise_for_status()
    data = r.json()
    for i, m in enumerate(data.get("messages", []), 1):
        click.echo(f"{i:2d}. {m.get('from')} | {m.get('subject')}\n    {m.get('snippet')}\n")


@cli.command()
@click.option("-n", default=25, help="Number of emails to process per batch")
@click.option("--list", "list_categories", is_flag=True,
              help="List available categories and exit")
def classify(n, list_categories):
    """Interactive email classifier"""
    if list_categories:
        r = requests.get(f"{BASE_URL}/categories")
        r.raise_for_status()
        for i, cat in enumerate(r.json().get("categories", [])):
            click.echo(f"{i}: {cat}")
        return
    _interactive_classify(n, preview=False)


@cli.command("classify-preview")
@click.option("-n", default=25, help="Number of emails to process per batch")
@click.option("--list", "list_categories", is_flag=True,
              help="List available categories and exit")
def classify_preview(n, list_categories):
    """Interactive classifier using Gmail previews"""
    if list_categories:
        r = requests.get(f"{BASE_URL}/categories")
        r.raise_for_status()
        for i, cat in enumerate(r.json().get("categories", [])):
            click.echo(f"{i}: {cat}")
        return
    _interactive_classify(n, preview=True)


@cli.command("classify-auto")
@click.argument("rules", type=click.Path(exists=True))
@click.option("-n", default=25, help="Number of emails to process per batch")
@click.option("--use-before-date", is_flag=True, 
              help="Enable before-date optimization when skipping classified emails")
def classify_auto(rules, n, use_before_date):
    """Automatically classify emails using regex rules"""
    with open(rules) as f:
        rule_data = json.load(f)

    compiled_rules = []
    for rule in rule_data:
        patterns = {}
        if rule.get("sender_name"):
            patterns["sender_name"] = re.compile(rule["sender_name"], re.IGNORECASE)
        if rule.get("sender_email"):
            patterns["sender_email"] = re.compile(rule["sender_email"], re.IGNORECASE)
        if rule.get("subject"):
            patterns["subject"] = re.compile(rule["subject"], re.IGNORECASE)
        if rule.get("preview"):
            patterns["content"] = re.compile(rule["preview"], re.IGNORECASE)
        if not patterns:
            continue
        compiled_rules.append({
            "patterns": patterns,
            "category": rule.get("category"),
            # delete may be a boolean or special indicator string (e.g. "before-this-year")
            "delete": rule.get("delete", False),
        })

    page_token = None
    batch = []
    try:
        while True:
            if not batch:
                click.echo("Loading more emails...")
                params = {
                    "n": n,
                    "skip_classified": "true",
                    "use_before": "true" if use_before_date else "false",
                }
                if page_token:
                    params["page_token"] = page_token
                r = requests.get(f"{BASE_URL}/emails/stream-preview", params=params)
                r.raise_for_status()
                data = r.json()
                batch = data.get("messages", [])
                page_token = data.get("next_page_token")
                if not batch:
                    click.echo("No more messages.")
                    return

            msg = batch.pop(0)
            matched = False
            for rule in compiled_rules:
                patterns = rule["patterns"]
                if all(patterns[field].search(msg.get(field, "") or "") for field in patterns):
                    # Determine whether to delete based on delete config
                    delete_cfg = rule["delete"]
                    if isinstance(delete_cfg, bool):
                        do_delete = delete_cfg
                    elif isinstance(delete_cfg, str) and delete_cfg == "before-this-year":
                        # parse email date header and delete if before current year
                        try:
                            dt = parsedate_to_datetime(msg.get("date", ""))
                            do_delete = dt.year < datetime.datetime.now(dt.tzinfo).year
                        except Exception:
                            do_delete = False
                    else:
                        do_delete = False

                    payload = {
                        "id": msg["id"],
                        "subject": msg.get("subject"),
                        "sender_name": msg.get("sender_name"),
                        "sender_email": msg.get("sender_email"),
                        "content": msg.get("content"),
                        "category": rule["category"],
                        "delete": do_delete,
                    }
                    requests.post(f"{BASE_URL}/emails/classify", json=payload).raise_for_status()
                    if do_delete:
                        click.echo(
                            f"Categorized as {rule['category']} and deleted: {msg.get('subject')}"
                        )
                    else:
                        click.echo(
                            f"Categorized as {rule['category']}: {msg.get('subject')}"
                        )
                    matched = True
                    break
            if not matched:
                click.echo(f"No rule matched: {msg.get('subject')}")
    except KeyboardInterrupt:
        click.echo("Stopping automatic classification.")


@cli.command()
@click.argument("classes", nargs=-1)
@click.option("--list", "list_classes", is_flag=True,
              help="List available clutter classes and exit")
@click.option("--before-this-year", "before_this_year", is_flag=True,
              help="Only process emails from before the current year")
@click.option("-n", default=25, help="Number of emails to process per batch")
@click.option("--dry-run", is_flag=True,
              help="Dry run: show which emails would be deleted without actually deleting them")
def declutter(n, classes, dry_run, list_classes, before_this_year):
    """Preview and delete clutter emails (marketing/newsletter/etc.); use --dry-run to preview only"""
    default_classes = ("marketing", "newsletter", "notification")
    if list_classes:
        to_list = classes or default_classes
        for cls in to_list:
            click.echo(cls)
        return
    classes = classes or default_classes
    params = {"n": n, "classes": list(classes)}
    if dry_run:
        params["dry_run"] = "true"
        click.echo("Dry run mode: no messages will be deleted.")
    if before_this_year:
        params["before_this_year"] = "true"
        click.echo("Only processing emails from before the current year.")
    r = requests.get(f"{BASE_URL}/emails/declutter", params=params, stream=True)
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if line:
            click.echo(line)


if __name__ == "__main__":
    cli()
