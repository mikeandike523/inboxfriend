import click

from cli_handlers.auth import login, logout, me
from cli_handlers.stats import stats
from cli_handlers.recent import recent_emails
from cli_handlers.classify import classify, classify_preview, classify_auto
from cli_handlers.declutter import declutter


@click.group()
def cli():
    """Inbox Tool CLI"""
    pass


cli.add_command(login)
cli.add_command(logout)
cli.add_command(me)
cli.add_command(stats)
cli.add_command(recent_emails)
cli.add_command(classify)
cli.add_command(classify_preview)
cli.add_command(classify_auto)
cli.add_command(declutter)


if __name__ == "__main__":
    cli()
