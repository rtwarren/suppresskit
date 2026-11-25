"""Primary and complementary small-cell suppression for two-way count tables."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

Margins = Literal["both", "rows", "columns", "none"]
Axis = Literal["row", "column"]


@dataclass(frozen=True)
class SuppressionRecord:
    """One suppressed cell and the reason it was suppressed."""

    row: Hashable
    column: Hashable
    value: int
    kind: Literal["primary", "complementary"]
    reason: str


@dataclass(frozen=True)
class Exposure:
    """A row or column whose published total lets a suppressed cell be recovered."""

    axis: Axis
    label: Hashable
    row: Hashable
    column: Hashable


@dataclass
class SuppressionResult:
    """Output of :func:`suppress_table`."""

    original: pd.DataFrame
    mask: pd.DataFrame
    records: list[SuppressionRecord]
    threshold: int
    margins: Margins
    unresolved: list[Exposure] = field(default_factory=list)

    @property
    def n_suppressed(self) -> int:
        return int(self.mask.to_numpy().sum())

    @property
    def table(self) -> pd.DataFrame:
        """The table with suppressed cells set to ``pd.NA`` (nullable Int64)."""
        return self.original.astype("Int64").mask(self.mask)

    def render(self, symbol: str = "x", include_totals: bool = True) -> pd.DataFrame:
        """Return a publishable table of strings with suppressed cells replaced by ``symbol``.

        Totals are the true totals from the original table, added only for the margins
        that ``suppress_table`` was told would be published.
        """
        out = self.original.astype(object).astype(str).mask(self.mask, symbol)
        if not include_totals:
            return out
        if self.margins in ("both", "rows"):
            out["Total"] = self.original.sum(axis=1).astype(str)
        if self.margins in ("both", "columns"):
            col_totals = self.original.sum(axis=0).astype(str)
            if self.margins == "both":
                col_totals["Total"] = str(int(self.original.to_numpy().sum()))
            out.loc["Total"] = col_totals
        return out

    def log(self) -> pd.DataFrame:
        """Audit log of every suppressed cell, one row per cell."""
        return pd.DataFrame(
            [
                {
                    "row": r.row,
                    "column": r.column,
                    "value": r.value,
                    "kind": r.kind,
                    "reason": r.reason,
                }
                for r in self.records
            ],
            columns=["row", "column", "value", "kind", "reason"],
        )


def _validate(table: pd.DataFrame, threshold: int) -> pd.DataFrame:
    if not isinstance(table, pd.DataFrame):
        raise TypeError("table must be a pandas DataFrame")
    if table.empty:
        raise ValueError("table is empty")
    if threshold < 1:
        raise ValueError("threshold must be at least 1")
    if not table.index.is_unique or not table.columns.is_unique:
        raise ValueError("row and column labels must be unique")
    if table.isna().to_numpy().any():
        raise ValueError("table contains missing values; fill or drop them first")
    try:
        numeric = table.astype("int64")
    except (TypeError, ValueError) as exc:
        raise ValueError("table must contain whole-number counts") from exc
    if not (numeric == table).to_numpy().all():
        raise ValueError("table must contain whole-number counts")
    if (numeric < 0).to_numpy().any():
        raise ValueError("counts cannot be negative")
    return numeric


def find_exposed_cells(mask: pd.DataFrame, margins: Margins = "both") -> list[Exposure]:
    """Find rows or columns containing exactly one suppressed cell.

    If a row (or column) total is published and only one cell in it is suppressed,
    that cell equals the total minus the visible cells, so it is not protected.
    """
    exposures: list[Exposure] = []
    if margins in ("both", "rows"):
        for row_label, row in mask.iterrows():
            hidden = row[row]
            if len(hidden) == 1:
                exposures.append(Exposure("row", row_label, row_label, hidden.index[0]))
    if margins in ("both", "columns"):
        for col_label in mask.columns:
            col = mask[col_label]
            hidden = col[col]
            if len(hidden) == 1:
                exposures.append(Exposure("column", col_label, hidden.index[0], col_label))
    return exposures


def _pick_complement(
    values: pd.DataFrame, mask: pd.DataFrame, exposure: Exposure
) -> tuple[Hashable, Hashable] | None:
    """Choose the cell to suppress alongside an exposed one.

    Preference order: non-zero cells over zeros (suppressing a zero hides little),
    cells whose crossing line already has a suppression (stops cascades),
    then the smallest value (least information lost).
    """
    if exposure.axis == "row":
        line = values.loc[exposure.label]
        hidden = mask.loc[exposure.label]
        candidates = [c for c in line.index if not hidden[c]]
        cells = [(exposure.label, c) for c in candidates]
        crossing_has_hidden = [bool(mask[c].any()) for c in candidates]
    else:
        line = values[exposure.label]
        hidden = mask[exposure.label]
        candidates = [r for r in line.index if not hidden[r]]
        cells = [(r, exposure.label) for r in candidates]
        crossing_has_hidden = [bool(mask.loc[r].any()) for r in candidates]

    if not cells:
        return None

    def sort_key(i: int) -> tuple[int, int, int, int]:
        value = int(line[candidates[i]])
        return (value == 0, not crossing_has_hidden[i], value, i)

    best = min(range(len(cells)), key=sort_key)
    return cells[best]


def suppress_table(
    table: pd.DataFrame,
    threshold: int = 5,
    *,
    suppress_zeros: bool = False,
    complementary: bool = True,
    margins: Margins = "both",
    max_iterations: int = 1000,
) -> SuppressionResult:
    """Apply small-cell suppression to a two-way table of counts.

    Parameters
    ----------
    table:
        Counts with categories as the index and columns. Do not include total rows or
        columns; totals are derived from the cells.
    threshold:
        Cells with ``0 < value < threshold`` are suppressed (primary suppression).
    suppress_zeros:
        Also treat zero cells as sensitive.
    complementary:
        Add complementary suppressions so no suppressed cell can be recovered by
        subtracting visible cells from a published row or column total.
    margins:
        Which totals will be published alongside the table: ``"both"``, ``"rows"``,
        ``"columns"`` or ``"none"``. Complementary suppression only protects against
        totals you say will be published.
    max_iterations:
        Safety limit on the complementary suppression loop.

    Notes
    -----
    Complementary suppression uses a greedy heuristic. It guarantees that no row or
    column published with a total contains exactly one suppressed cell, which blocks
    simple subtraction. It does not solve the full optimal suppression problem and
    does not guarantee protection against interval (linear programming) attacks.
    """
    values = _validate(table, threshold)
    mask = (values < threshold) & ((values > 0) | suppress_zeros)

    records: list[SuppressionRecord] = []
    for row_label, col_label in zip(*mask.to_numpy().nonzero()):
        r = values.index[row_label]
        c = values.columns[col_label]
        v = int(values.loc[r, c])
        records.append(
            SuppressionRecord(r, c, v, "primary", f"value {v} is below threshold {threshold}")
        )

    unresolved: list[Exposure] = []
    if complementary and margins != "none":
        for _ in range(max_iterations):
            exposures = [
                e
                for e in find_exposed_cells(mask, margins)
                if e not in unresolved
            ]
            if not exposures:
                break
            exposure = exposures[0]
            choice = _pick_complement(values, mask, exposure)
            if choice is None:
                unresolved.append(exposure)
                continue
            r, c = choice
            mask.loc[r, c] = True
            v = int(values.loc[r, c])
            records.append(
                SuppressionRecord(
                    r,
                    c,
                    v,
                    "complementary",
                    f"protects ({exposure.row}, {exposure.column}) "
                    f"from recovery via {exposure.axis} total '{exposure.label}'",
                )
            )
        else:
            raise RuntimeError("complementary suppression did not converge")

    return SuppressionResult(
        original=values,
        mask=mask,
        records=records,
        threshold=threshold,
        margins=margins,
        unresolved=unresolved,
    )
