from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from suppresskit import k_anonymity, random_round, round_to_base
from suppresskit.cli import main


def test_round_to_base() -> None:
    t = pd.DataFrame({"a": [0, 2, 3, 7, 8, 12, 13]})
    assert round_to_base(t, 5)["a"].tolist() == [0, 0, 5, 5, 10, 10, 15]


def test_round_to_even_base_rounds_halves_up() -> None:
    t = pd.DataFrame({"a": [4, 5, 15]})
    assert round_to_base(t, 10)["a"].tolist() == [0, 10, 20]


def test_random_round_outputs_multiples_and_stays_close() -> None:
    t = pd.DataFrame(np.arange(0, 100).reshape(10, 10))
    out = random_round(t, 5, seed=1)
    assert (out % 5 == 0).to_numpy().all()
    assert ((out - t).abs() < 5).to_numpy().all()


def test_random_round_is_unbiased_on_average() -> None:
    t = pd.DataFrame({"a": [7] * 20000})
    assert random_round(t, 5, seed=42)["a"].mean() == pytest.approx(7, abs=0.1)


def test_random_round_reproducible_with_seed() -> None:
    t = pd.DataFrame({"a": [1, 2, 3, 4, 6, 7]})
    assert random_round(t, seed=3).equals(random_round(t, seed=3))


def test_rounding_rejects_bad_base() -> None:
    with pytest.raises(ValueError):
        round_to_base(pd.DataFrame({"a": [1]}), base=1)


@pytest.fixture
def records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age_group": ["20-29"] * 6 + ["30-39"] * 5 + ["80+"],
            "sex": ["F"] * 6 + ["M"] * 5 + ["F"],
            "community": ["A"] * 6 + ["B"] * 5 + [None],
        }
    )


def test_k_anonymity_finds_small_classes(records: pd.DataFrame) -> None:
    report = k_anonymity(records, ["age_group", "sex", "community"], k=5)
    assert not report.satisfied
    assert report.min_class_size == 1
    assert report.records_at_risk == 1
    assert report.violations.iloc[0]["community"] == "<missing>"
    assert "do NOT satisfy" in report.summary()


def test_k_anonymity_passes_when_all_classes_large(records: pd.DataFrame) -> None:
    report = k_anonymity(records.iloc[:11], ["age_group", "sex"], k=5)
    assert report.satisfied
    assert report.min_class_size == 5


def test_k_anonymity_validates_columns(records: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        k_anonymity(records, ["postal_code"])


def test_cli_writes_table_and_log(tmp_path: Path) -> None:
    src = tmp_path / "in.csv"
    pd.DataFrame(
        {"a": [12, 3, 40], "b": [55, 21, 18], "c": [9, 14, 2]}, index=["r1", "r2", "r3"]
    ).to_csv(src)
    out, log = tmp_path / "out.csv", tmp_path / "log.csv"
    assert main([str(src), "-o", str(out), "--log", str(log)]) == 0
    published = pd.read_csv(out, index_col=0, dtype=str)
    assert published.loc["r2", "a"] == "x"
    assert len(pd.read_csv(log)) == 4


def test_cli_returns_error_code_for_unresolved(tmp_path: Path) -> None:
    src = tmp_path / "in.csv"
    pd.DataFrame({"count": [2, 50]}, index=["r1", "r2"]).to_csv(src)
    assert main([str(src), "--margins", "rows", "-o", str(tmp_path / "o.csv")]) == 1
