"""a python cli command to be updated for training and inference."""

import click


@click.command()
@click.option(
    "--option", default="default_value", help="Description for this option"
)
def main(option: str) -> None:
    """Your main function logic here."""
    click.echo(f"Option: {option}")


if __name__ == "__main__":
    main()
