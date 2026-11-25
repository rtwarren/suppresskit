"""Command-line interface: suppress a CSV table and write the result plus an audit log."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from suppresskit.suppression import suppress_table


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="suppresskit",
        description="Apply small-cell suppression to a two-way count table in CSV form.",
    )
    p.add_argument("input", type=Path, help="CSV file; first column holds the row labels")
    p.add_argument("-o", "--output", type=Path, help="where to write the suppressed table")
    p.add_argument("--log", type=Path, help="where to write the audit log CSV")
    p.add_argument("-t", "--threshold", type=int, default=5, help="minimum publishable count")
    p.add_argument("--suppress-zeros", action="store_true", help="treat zeros as sensitive")
    p.add_argument(
        "--margins",
        choices=["both", "rows", "columns", "none"],
        default="both",
        help="which totals will be published",
    )
    p.add_argument("--symbol", default="x", help="placeholder for suppressed cells")
    p.add_argument("--no-totals", action="store_true", help="omit totals from the output")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    table = pd.read_csv(args.input, index_col=0)
    result = suppress_table(
        table,
        threshold=args.threshold,
        suppress_zeros=args.suppress_zeros,
        margins=args.margins,
    )
    rendered = result.render(symbol=args.symbol, include_totals=not args.no_totals)

    if args.output:
        rendered.to_csv(args.output)
    else:
        rendered.to_csv(sys.stdout)
    if args.log:
        result.log().to_csv(args.log, index=False)

    primary = sum(r.kind == "primary" for r in result.records)
    print(
        f"Suppressed {result.n_suppressed} cells "
        f"({primary} primary, {result.n_suppressed - primary} complementary).",
        file=sys.stderr,
    )
    for e in result.unresolved:
        print(
            f"WARNING: ({e.row}, {e.column}) can be recovered from {e.axis} total "
            f"'{e.label}' and there is no other cell to suppress. Suppress that total too.",
            file=sys.stderr,
        )
    return 1 if result.unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
