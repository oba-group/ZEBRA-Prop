"""Module entry point for `python -m zebra_prop`."""

from .cli import main


if __name__ == "__main__":
    # Delegate all command handling to the CLI module.
    main()
