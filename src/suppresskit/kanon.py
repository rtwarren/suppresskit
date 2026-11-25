"""k-anonymity checks for record-level data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass
class KAnonymityReport:
    """Result of :func:`k_anonymity`."""

    k: int
    quasi_identifiers: list[str]
    n_records: int
    min_class_size: int
    classes: pd.DataFrame
    violations: pd.DataFrame

    @property
    def satisfied(self) -> bool:
        return bool(self.violations.empty)

    @property
    def records_at_risk(self) -> int:
        return int(self.violations["count"].sum())

    def summary(self) -> str:
        status = "satisfy" if self.satisfied else "do NOT satisfy"
        return (
            f"{self.n_records} records {status} {self.k}-anonymity on "
            f"{', '.join(self.quasi_identifiers)}. "
            f"Smallest equivalence class: {self.min_class_size}. "
            f"Classes below k: {len(self.violations)} "
            f"({self.records_at_risk} records)."
        )


def k_anonymity(
    records: pd.DataFrame, quasi_identifiers: Sequence[str], k: int = 5
) -> KAnonymityReport:
    """Check whether each combination of quasi-identifiers appears at least ``k`` times.

    Missing values are treated as their own category, because a blank field can
    still single someone out.
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    qis = list(quasi_identifiers)
    if not qis:
        raise ValueError("at least one quasi-identifier is required")
    missing = [q for q in qis if q not in records.columns]
    if missing:
        raise KeyError(f"columns not found: {missing}")
    if records.empty:
        raise ValueError("records is empty")

    classes = (
        records[qis]
        .astype(object)
        .fillna("<missing>")
        .value_counts(sort=False)
        .rename("count")
        .reset_index()
        .sort_values(["count", *qis])
        .reset_index(drop=True)
    )
    violations = classes[classes["count"] < k].reset_index(drop=True)
    return KAnonymityReport(
        k=k,
        quasi_identifiers=qis,
        n_records=len(records),
        min_class_size=int(classes["count"].min()),
        classes=classes,
        violations=violations,
    )
