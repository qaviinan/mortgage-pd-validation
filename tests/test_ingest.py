"""End-to-end ingest on a synthetic zip in the exact Freddie layout, plus layout drift check."""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from pdval.ingest import extract, find_sample_zips, ingest_vintage, write_manifest
from pdval.layout import ORIG_COLUMNS, PERF_COLUMNS, verify_layout
from pdval.synth import make_vintage
from pdval.target import build_target

LAYOUT = Path(__file__).resolve().parents[1] / "docs" / "reference" / "file_layout_july_2026.xlsx"


def test_layout_matches_freddie_spreadsheet():
    assert LAYOUT.exists(), "file_layout_july_2026.xlsx missing from docs/reference/"
    n = verify_layout(LAYOUT)
    assert n == {"orig": len(ORIG_COLUMNS), "perf": len(PERF_COLUMNS)} == {"orig": 31, "perf": 35}


def test_no_performance_columns_leak_into_features():
    from pdval.layout import ORIG_COLUMNS, PERF_COLUMNS
    shared = set(ORIG_COLUMNS) & set(PERF_COLUMNS)
    assert shared == {"loan_id"}


@pytest.fixture
def synthetic_repo(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw"
    make_vintage(2016, 300, raw, seed=1)
    cfg = {
        "data": {
            "raw_dir": str(raw),
            "interim_dir": str(tmp_path / "data" / "interim"),
            "processed_dir": str(tmp_path / "data" / "processed"),
        }
    }
    monkeypatch.chdir(tmp_path)
    return raw, cfg


def test_ingest_roundtrip(synthetic_repo):
    raw, cfg = synthetic_repo
    zips = find_sample_zips(raw)
    assert list(zips) == [2016]
    m = write_manifest(raw, raw.parent / "manifest.json")
    assert m["files"][0]["sha256"] and m["files"][0]["bytes"] > 0

    con = duckdb.connect()
    r = ingest_vintage(2016, zips[2016], cfg, con)
    assert r["orig"]["rows"] == 300
    assert r["perf"]["rows"] > 300
    assert r["perf_loans_not_in_orig"] == 0 and r["orig_loans_not_in_perf"] == 0

    proc = Path(cfg["data"]["processed_dir"])
    orig = con.execute(f"SELECT * FROM '{proc / 'orig_2016.parquet'}'").df()
    perf = con.execute(f"SELECT * FROM '{proc / 'perf_2016.parquet'}'").df()
    assert list(orig.columns) == ORIG_COLUMNS and list(perf.columns) == PERF_COLUMNS
    assert orig["fico"].between(300, 850).all()
    assert perf["period"].dtype.kind == "i"
    assert not pd.api.types.is_numeric_dtype(perf["dlq_status"])  # stays text: '00'..'99', 'RA', 'XX'

    t = build_target(orig, perf, window=36)
    assert len(t) == 300
    assert t["y"].isin([0, 1]).sum() > 0
    assert (t["y"] == 1).sum() > 0  # synthetic default rate ≈ 5%


def test_extract_rejects_bad_zip(tmp_path):
    import zipfile
    bad = tmp_path / "sample_1999.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("readme.txt", "nope")
    with pytest.raises(RuntimeError):
        extract(bad, tmp_path / "x")
