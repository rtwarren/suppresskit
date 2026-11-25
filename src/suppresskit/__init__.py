"""suppresskit: statistical disclosure control for aggregate tables.

Tools for preparing small-population health data for public release:

- ``suppress_table``: primary and complementary small-cell suppression with an audit log
- ``find_exposed_cells``: checks a suppression pattern for cells recoverable from margins
- ``round_to_base`` and ``random_round``: base rounding of counts
- ``k_anonymity``: equivalence-class checks on record-level data
"""

from suppresskit.kanon import KAnonymityReport, k_anonymity
from suppresskit.rounding import random_round, round_to_base
from suppresskit.suppression import (
    SuppressionRecord,
    SuppressionResult,
    find_exposed_cells,
    suppress_table,
)

__all__ = [
    "KAnonymityReport",
    "SuppressionRecord",
    "SuppressionResult",
    "find_exposed_cells",
    "k_anonymity",
    "random_round",
    "round_to_base",
    "suppress_table",
]

__version__ = "0.1.0"
