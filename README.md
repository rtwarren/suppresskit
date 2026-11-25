# suppresskit

Small-cell suppression, rounding and k-anonymity checks for publishing aggregate health data.

Small populations make disclosure risk a real problem. In a jurisdiction where a region has a few hundred residents, a published count of 2 cases in one age group can identify a person. Hiding that cell is not enough on its own: if row and column totals are published, the hidden value can be recovered by subtraction. `suppresskit` handles both steps and records why every cell was hidden, so a release can be reviewed and defended.

## Features

- **Primary suppression** of counts below a threshold (zeros optional)
- **Complementary suppression** so no hidden cell can be recovered from published row or column totals
- **Audit log** with one row per suppressed cell and the reason it was suppressed
- **Exposure checker** to validate any suppression pattern, including ones made by hand
- **Base rounding**, deterministic or unbiased random rounding with a recorded seed
- **k-anonymity report** for record-level data, listing every equivalence class below k
- **Command-line tool** for CSV in, CSV out workflows

## Install

```bash
pip install git+https://github.com/rtwarren/suppresskit.git
```

For development:

```bash
git clone https://github.com/rtwarren/suppresskit.git
cd suppresskit
pip install -e ".[dev]"
pytest
```

## Quick start

```python
import pandas as pd
from suppresskit import suppress_table

cases = pd.DataFrame(
    {
        "0-19": [14, 3, 22, 0, 7],
        "20-39": [41, 9, 35, 4, 18],
        "40-64": [38, 6, 29, 2, 15],
        "65+": [11, 1, 8, 5, 3],
    },
    index=["Region 1", "Region 2", "Region 3", "Region 4", "Region 5"],
)

result = suppress_table(cases, threshold=5)
print(result.render(symbol="x"))
```

```
         0-19 20-39 40-64 65+ Total
Region 1   14    41    38  11   104
Region 2    x     x     x   x    19
Region 3   22    35    29   8    94
Region 4    0     x     x   5    11
Region 5    x    18    15   x    43
Total      46   107    90  28   271
```

Five cells were below the threshold. Three more were suppressed so none of the five can be worked out from the totals:

```python
print(result.log())
```

```
     row column  value          kind                                                            reason
Region 2   0-19      3       primary                                      value 3 is below threshold 5
Region 2    65+      1       primary                                      value 1 is below threshold 5
Region 4  20-39      4       primary                                      value 4 is below threshold 5
Region 4  40-64      2       primary                                      value 2 is below threshold 5
Region 5    65+      3       primary                                      value 3 is below threshold 5
Region 5   0-19      7 complementary   protects (Region 5, 65+) from recovery via row total 'Region 5'
Region 2  20-39      9 complementary protects (Region 4, 20-39) from recovery via column total '20-39'
Region 2  40-64      6 complementary protects (Region 4, 40-64) from recovery via column total '40-64'
```

The full example, including rounding and a k-anonymity check, is in [`examples/notifiable_disease_example.py`](examples/notifiable_disease_example.py). All data in this repository is synthetic.

## Command line

```bash
suppresskit cases.csv --threshold 5 -o cases_public.csv --log suppression_log.csv
```

The first CSV column is used as row labels. Use `--margins rows|columns|none` if you will not publish both sets of totals, and `--suppress-zeros` if zeros are sensitive. The command exits with code 1 if a cell cannot be protected (for example, a single-column table where the row total equals the cell), so it can gate a release pipeline.

## API

| Function | Purpose |
| --- | --- |
| `suppress_table(table, threshold=5, *, suppress_zeros=False, complementary=True, margins="both")` | Returns a `SuppressionResult` with `.mask`, `.table`, `.render()`, `.log()` and `.unresolved` |
| `find_exposed_cells(mask, margins="both")` | Lists rows or columns with exactly one hidden cell |
| `round_to_base(table, base=5)` | Rounds each count to the nearest multiple of `base` |
| `random_round(table, base=5, seed=None)` | Unbiased random rounding; expected value equals the original |
| `k_anonymity(records, quasi_identifiers, k=5)` | Returns a `KAnonymityReport` with `.satisfied`, `.violations` and `.summary()` |

## How complementary suppression works

For each row or column whose total will be published, the table must never have exactly one hidden cell. The algorithm repeats until that holds:

1. Find a row or column with exactly one hidden cell.
2. Choose another cell in that line to hide, preferring in order:
   - non-zero cells, since hiding a zero protects little
   - cells whose crossing line already has a hidden cell, which closes both lines at once and stops cascades
   - the smallest value, to lose as little information as possible
3. Record the choice and the cell it protects.

### Limitations

This is a greedy heuristic. It blocks recovery by simple subtraction, which is the most common attack on published tables. It does not find the minimum number of suppressions, and it does not guarantee protection against interval attacks, where linear programming narrows a hidden cell to a tight range. For high-risk releases, review the output or use a dedicated tool such as tau-ARGUS or sdcTable.

## Project layout

```
src/suppresskit/
  suppression.py   primary and complementary suppression, exposure checks
  rounding.py      base and random rounding
  kanon.py         k-anonymity report
  cli.py           command-line interface
tests/             pytest suite, including randomized tables
examples/          worked example on synthetic data
```

## License

MIT
