"""Command-line entry point: ``opengasspec <subcommand>``."""

from __future__ import annotations

import sys

from . import __version__


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] in {"-h", "--help"}:
        print(
            "OpenGasSpec CLI\n\nsubcommands:\n"
            "  validate FILES...   validate records against the schema\n"
            "  --version           print version"
        )
        return 0
    if args[0] == "--version":
        print(__version__)
        return 0
    if args[0] == "validate":
        from .validate import main as validate_main

        return validate_main(args[1:])
    print(f"unknown subcommand: {args[0]}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
