import numpy as np
import pandas as pd
import pytest

from suppresskit import find_exposed_cells, suppress_table


@pytest.fixture
def table() -> pd.DataFrame:
    return pd.DataFrame(
        {"0-19": [12, 3, 40], "20-64": [55, 21, 18], "65+": [9, 14, 2]},
        index=["Region A", "Region B", "Region C"],
    )


def test_primary_suppression_flags_small_nonzero_cells(table: pd.DataFrame) -> None:
    result = suppress_table(table, threshold=5, complementary=False)
    assert result.mask.loc["Region B", "0-19"]
    assert result.mask.loc["Region C", "65+"]
    assert result.n_suppressed == 2
    assert all(r.kind == "primary" for r in result.records)


def test_zeros_not_suppressed_by_default() -> None:
    t = pd.DataFrame({"a": [0, 10], "b": [10, 10]}, index=["r1", "r2"])
    assert suppress_table(t, complementary=False).n_suppressed == 0
    assert suppress_table(t, complementary=False, suppress_zeros=True).n_suppressed == 1


def test_complementary_leaves_no_exposed_cells(table: pd.DataFrame) -> None:
    result = suppress_table(table, threshold=5)
    assert find_exposed_cells(result.mask, "both") == []
    assert any(r.kind == "complementary" for r in result.records)


def test_complementary_prefers_cells_that_close_both_lines(table: pd.DataFrame) -> None:
    # Primary cells are (B, 0-19) and (C, 65+). Suppressing (B, 65+) and (C, 0-19)
    # forms a closed rectangle that protects both with only two extra cells.
    result = suppress_table(table, threshold=5)
    assert result.n_suppressed == 4
    assert result.mask.loc["Region B", "65+"]
    assert result.mask.loc["Region C", "0-19"]


def test_complementary_skips_zero_cells_when_possible() -> None:
    t = pd.DataFrame({"a": [2, 50], "b": [0, 60], "c": [30, 70]}, index=["r1", "r2"])
    result = suppress_table(t, threshold=5)
    assert not result.mask.loc["r1", "b"]
    assert find_exposed_cells(result.mask) == []


def test_margins_none_means_no_complementary_suppression(table: pd.DataFrame) -> None:
    result = suppress_table(table, threshold=5, margins="none")
    assert result.n_suppressed == 2


def test_rows_only_margins_protect_rows_only() -> None:
    t = pd.DataFrame({"a": [2, 50], "b": [40, 60]}, index=["r1", "r2"])
    result = suppress_table(t, threshold=5, margins="rows")
    assert find_exposed_cells(result.mask, "rows") == []
    assert result.n_suppressed == 2


def test_single_column_table_reports_unresolved_exposure() -> None:
    t = pd.DataFrame({"count": [2, 50, 60]}, index=["r1", "r2", "r3"])
    result = suppress_table(t, threshold=5, margins="rows")
    assert len(result.unresolved) == 1
    assert result.unresolved[0].row == "r1"


@pytest.mark.parametrize("seed", range(25))
def test_random_tables_never_leave_exposed_cells(seed: int) -> None:
    rng = np.random.default_rng(seed)
    shape = rng.integers(2, 7, size=2)
    data = rng.integers(0, 30, size=shape)
    t = pd.DataFrame(data)
    result = suppress_table(t, threshold=5)
    remaining = [e for e in find_exposed_cells(result.mask) if e not in result.unresolved]
    assert remaining == []
    # Every primary cell stays suppressed.
    primary = (t < 5) & (t > 0)
    assert (result.mask | ~primary).to_numpy().all()


def test_render_uses_symbol_and_true_totals(table: pd.DataFrame) -> None:
    out = suppress_table(table, threshold=5).render(symbol="<5")
    assert out.loc["Region B", "0-19"] == "<5"
    assert out.loc["Region A", "Total"] == str(12 + 55 + 9)
    assert out.loc["Total", "Total"] == str(int(table.to_numpy().sum()))


def test_table_property_hides_values(table: pd.DataFrame) -> None:
    masked = suppress_table(table, threshold=5).table
    assert pd.isna(masked.loc["Region B", "0-19"])
    assert masked.loc["Region A", "20-64"] == 55


def test_log_has_one_row_per_suppressed_cell(table: pd.DataFrame) -> None:
    result = suppress_table(table, threshold=5)
    log = result.log()
    assert len(log) == result.n_suppressed
    assert set(log["kind"]) == {"primary", "complementary"}
    assert log["reason"].str.len().gt(0).all()


@pytest.mark.parametrize(
    "bad, message",
    [
        (pd.DataFrame({"a": [1.5, 2]}), "whole-number"),
        (pd.DataFrame({"a": [-1, 2]}), "negative"),
        (pd.DataFrame({"a": [None, 2]}), "missing"),
        (pd.DataFrame(), "empty"),
    ],
)
def test_invalid_input_rejected(bad: pd.DataFrame, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        suppress_table(bad)


def test_threshold_must_be_positive(table: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="threshold"):
        suppress_table(table, threshold=0)
