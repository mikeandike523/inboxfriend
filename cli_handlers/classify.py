import json
import re
import os
import shutil
import datetime
import textwrap

import click
import requests
from termcolor import colored
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from email.utils import parsedate_to_datetime

from cli_config import BASE_URL
from cli_helpers.utils import is_running_in_git_bash, pad_right

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

        from_line = f"From: {msg.get('sender_name')} <{msg.get('sender_email')}>"
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
            click.echo(pad_right(left, left_width) + ' ' + pad_right(right, right_width))

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
            if resp.upper().endswith(" DELETE"):
                delete = True
                resp = resp[:-7].strip()
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

@click.command()
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

@click.command("classify-preview")
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

@click.command("classify-auto")
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
