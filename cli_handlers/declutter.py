import click
import requests
from termcolor import colored

from cli_config import BASE_URL

@click.command()
@click.argument("classes", nargs=-1, required=True)
@click.option("--list", "--list-available-classes", "list_classes", is_flag=True,
              help="List available clutter classes and exit")
@click.option("--before-this-year", "before_this_year", is_flag=True,
              help="Only process emails from before the current year")
@click.option("-n", default=25, help="Number of emails to process per batch")
@click.option("--dry-run", is_flag=True,
              help="Dry run: show which emails would be deleted without actually deleting them")
def declutter(n, classes, dry_run, list_classes, before_this_year):
    """Preview and delete clutter emails (marketing/newsletter/etc.); use --dry-run to preview only"""
    if list_classes:
        resp = requests.get(f"{BASE_URL}/categories?model=true")
        resp.raise_for_status()
        for cls in resp.json().get("categories", []):
            click.echo(cls)
        return

    # Validate provided classes against available pretrained model classes
    resp = requests.get(f"{BASE_URL}/categories?model=true")
    resp.raise_for_status()
    available_classes = resp.json().get("categories", [])
    available_classes_lower = [cls.lower() for cls in available_classes]

    invalid_classes = []
    for cls in classes:
        if cls.lower() not in available_classes_lower:
            invalid_classes.append(cls)

    if invalid_classes:
        click.echo(colored(f"Error: Invalid classes specified: {', '.join(invalid_classes)}", "red"))
        click.echo(colored(f"Available classes are: {', '.join(available_classes)}", "cyan"))
        return

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
