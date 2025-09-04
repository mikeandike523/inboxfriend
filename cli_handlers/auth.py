import click
import requests
import webbrowser

from cli_config import BASE_URL

@click.command()
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

@click.command()
def logout():
    """Revoke and delete tokens"""
    r = requests.post(f"{BASE_URL}/auth/logout")
    click.echo(f"{r.status_code} {r.json()}")

@click.command()
def me():
    """Show profile info; also refreshes token if expired"""
    r = requests.get(f"{BASE_URL}/me")
    click.echo(f"{r.status_code} {r.json()}")
