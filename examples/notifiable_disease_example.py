"""Worked example: preparing a small-population surveillance table for public release.

The counts are synthetic. They are shaped like a typical territorial release
(cases by region and age group) so the suppression behaviour is realistic.
"""

import pandas as pd

from suppresskit import k_anonymity, random_round, suppress_table

cases = pd.DataFrame(
    {
        "0-19": [14, 3, 22, 0, 7],
        "20-39": [41, 9, 35, 4, 18],
        "40-64": [38, 6, 29, 2, 15],
        "65+": [11, 1, 8, 5, 3],
    },
    index=["Region 1", "Region 2", "Region 3", "Region 4", "Region 5"],
)

print("Original counts (not for release):")
print(cases, "\n")

result = suppress_table(cases, threshold=5)
print("Publishable table:")
print(result.render(symbol="x"), "\n")

print("Audit log:")
print(result.log().to_string(index=False), "\n")

print("Alternative: random rounding to base 5 (seed recorded for reproducibility):")
print(random_round(cases, base=5, seed=2026), "\n")

# Record-level check before releasing a line list.
line_list = pd.DataFrame(
    {
        "age_group": ["20-39"] * 8 + ["40-64"] * 6 + ["65+"] * 2,
        "sex": ["F"] * 4 + ["M"] * 4 + ["F"] * 6 + ["M", "F"],
        "region": ["Region 1"] * 8 + ["Region 3"] * 6 + ["Region 2"] * 2,
    }
)
report = k_anonymity(line_list, ["age_group", "sex", "region"], k=5)
print(report.summary())
print(report.violations.to_string(index=False))
