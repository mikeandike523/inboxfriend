import click
import requests

from cli_config import BASE_URL

@click.command("recent-emails")
@click.option("-n", default=20, help="Number of emails to fetch")
def recent_emails(n):
    """List recent emails (sender, subject, snippet)"""
    r = requests.get(f"{BASE_URL}/emails/recent", params={"n": n})
    r.raise_for_status()
    data = r.json()
    for i, m in enumerate(data.get("messages", []), 1):
        click.echo(f"{i:2d}. {m.get('from')} | {m.get('subject')}\n    {m.get('snippet')}\n")
