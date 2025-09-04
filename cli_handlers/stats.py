import click
import requests

from cli_config import BASE_URL

@click.command()
def stats():
    """Show message stats (total messages and inbox messages)."""
    r = requests.get(f"{BASE_URL}/stats")
    r.raise_for_status()
    data = r.json()
    click.echo(f"Total messages: {data.get('total_messages')}")
    click.echo(f"Inbox messages: {data.get('inbox_messages')}")
