"""Ingest Freddie Mac sample files: zip -> pipe-delimited txt -> typed parquet.

Usage:
    uv run python -m pdval.ingest --config configs/default.yaml [--years 2016 2017]

Side effects:
    data/manifest.json          sha256, size, mtime of every raw zip (data lineage)
    data/interim/<year>/        extracted txt files
    data/processed/orig_<year>.parquet, perf_<year>.parquet
    outputs/ingest_log.json     row counts and column checks per vintage
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import yaml

from pdval.layout import (
    ORIG_CASTS,
    ORIG_COLUMNS,
    PERF_CASTS,
    PERF_COLUMNS,
    verify_layout,
)

SAMPLE_ZIP_RE = re.compile(r"^sample_(\d{4})\.zip$")


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_sample_zips(raw_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in sorted(raw_dir.glob("sample_*.zip")):
        m = SAMPLE_ZIP_RE.match(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def write_manifest(raw_dir: Path, manifest_path: Path) -> dict:
    entries = []
    for year, p in find_sample_zips(raw_dir).items():
        st = p.stat()
        entries.append(
            {
                "vintage": year,
                "file": p.name,
                "bytes": st.st_size,
                "sha256": sha256_of(p),
                "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=UTC).isoformat(),
            }
        )
    manifest = {
        "source": "Freddie Mac Single-Family Loan-Level Dataset, Sample files (Clarity Data Intelligence)",
        "generated_utc": datetime.now(tz=UTC).isoformat(),
        "files": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def extract(zip_path: Path, interim_dir: Path) -> tuple[Path, Path]:
    """Extract sample_orig_YYYY.txt and sample_perf_YYYY.txt; return their paths."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        txts = [n for n in names if n.lower().endswith(".txt")]
        orig = [n for n in txts if "orig" in n.lower()]
        # Freddie names the performance file sample_svcg_YYYY.txt ("servicing"); the user
        # guide calls it sample_perf_YYYY.txt. Accept either: the non-orig txt is performance.
        perf = [n for n in txts if n not in orig]
        if len(orig) != 1 or len(perf) != 1:
            raise RuntimeError(f"{zip_path.name}: expected one orig and one perf/svcg txt, found {names}")
        z.extract(orig[0], interim_dir)
        z.extract(perf[0], interim_dir)
    return interim_dir / orig[0], interim_dir / perf[0]


def _select_with_casts(columns: list[str], casts: dict[str, str]) -> str:
    parts = []
    for c in columns:
        if c in casts:
            parts.append(f"TRY_CAST(NULLIF(TRIM({c}), '') AS {casts[c]}) AS {c}")
        else:
            parts.append(f"NULLIF(TRIM({c}), '') AS {c}")
    return ",\n    ".join(parts)


def load_pipe_file(con: duckdb.DuckDBPyConnection, txt: Path, columns: list[str], casts: dict[str, str], out_parquet: Path) -> dict:
    """Read a header-less pipe-delimited file with the canonical column list; write parquet."""
    # Peek at the column count so a layout change fails with a clear message.
    with txt.open("r", encoding="utf-8", errors="replace") as f:
        first = f.readline().rstrip("\n")
    n_found = first.count("|") + 1
    if n_found != len(columns):
        raise RuntimeError(
            f"{txt.name}: {n_found} pipe-delimited fields but layout defines {len(columns)}. "
            "Check docs/reference/file_layout_*.xlsx against this release."
        )
    col_spec = "{" + ", ".join(f"'{c}': 'VARCHAR'" for c in columns) + "}"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY (
          SELECT
            {_select_with_casts(columns, casts)}
          FROM read_csv('{txt.as_posix()}', delim='|', header=false, quote='', escape='',
                        columns={col_spec}, null_padding=true, ignore_errors=false)
        ) TO '{out_parquet.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    n_rows = con.execute(f"SELECT count(*) FROM '{out_parquet.as_posix()}'").fetchone()[0]
    return {"file": txt.name, "rows": int(n_rows), "columns": len(columns), "parquet": out_parquet.name}


def ingest_vintage(year: int, zip_path: Path, cfg: dict, con: duckdb.DuckDBPyConnection, *, keep_interim: bool = False) -> dict:
    interim = Path(cfg["data"]["interim_dir"]) / str(year)
    processed = Path(cfg["data"]["processed_dir"])
    orig_txt, perf_txt = extract(zip_path, interim)
    try:
        o = load_pipe_file(con, orig_txt, ORIG_COLUMNS, ORIG_CASTS, processed / f"orig_{year}.parquet")
        p = load_pipe_file(con, perf_txt, PERF_COLUMNS, PERF_CASTS, processed / f"perf_{year}.parquet")
    finally:
        # The extracted text is ~0.5 GB per vintage and fully regenerable from the zip:
        # drop it as soon as the parquet exists so ingestion fits in a few hundred MB of headroom.
        if not keep_interim:
            for f in (orig_txt, perf_txt):
                f.unlink(missing_ok=True)
    # Sanity: every perf loan appears in orig and vice versa
    orphan = con.execute(
        f"""
        SELECT
          (SELECT count(*) FROM (SELECT DISTINCT loan_id FROM '{processed / f"perf_{year}.parquet"}') p
             WHERE loan_id NOT IN (SELECT loan_id FROM '{processed / f"orig_{year}.parquet"}')) AS perf_not_in_orig,
          (SELECT count(*) FROM '{processed / f"orig_{year}.parquet"}'
             WHERE loan_id NOT IN (SELECT DISTINCT loan_id FROM '{processed / f"perf_{year}.parquet"}')) AS orig_not_in_perf
        """
    ).fetchone()
    return {
        "vintage": year,
        "orig": o,
        "perf": p,
        "perf_loans_not_in_orig": int(orphan[0]),
        "orig_loans_not_in_perf": int(orphan[1]),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--years", nargs="*", type=int, help="vintages to ingest (default: every sample_YYYY.zip present)")
    ap.add_argument("--raw-dir", default=None, help="override data.raw_dir (used by `make smoke`)")
    ap.add_argument("--interim-dir", default=None)
    ap.add_argument("--processed-dir", default=None)
    ap.add_argument("--manifest", default="data/manifest.json")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    for key in ("raw_dir", "interim_dir", "processed_dir"):
        if getattr(args, key):
            cfg["data"][key] = getattr(args, key)
    raw_dir = Path(cfg["data"]["raw_dir"])
    layout = verify_layout(Path(cfg["data"]["layout_xlsx"]))
    print(f"layout verified: orig={layout['orig']} fields, perf={layout['perf']} fields")

    zips = find_sample_zips(raw_dir)
    if not zips:
        raise SystemExit(
            f"No sample_YYYY.zip files in {raw_dir}/. Download them from Freddie Mac Clarity "
            "(Data Download -> Sample) and drop them there. See SPEC.md §4.1."
        )
    years = args.years or sorted(zips)
    missing = [y for y in years if y not in zips]
    if missing:
        raise SystemExit(f"Missing sample zips for vintages: {missing}")

    manifest = write_manifest(raw_dir, Path(args.manifest))
    print(f"manifest: {len(manifest['files'])} raw files hashed -> {args.manifest}")

    con = duckdb.connect()
    con.execute("SET threads TO 8")
    con.execute(f"SET temp_directory = '{Path(cfg['data']['interim_dir']).as_posix()}/duckdb_tmp'")
    log = []
    for y in years:
        r = ingest_vintage(y, zips[y], cfg, con)
        log.append(r)
        print(
            f"  {y}: orig {r['orig']['rows']:>8,} loans | perf {r['perf']['rows']:>11,} loan-months"
            f" | orphans perf->orig {r['perf_loans_not_in_orig']} orig->perf {r['orig_loans_not_in_perf']}"
        )
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/ingest_log.json").write_text(json.dumps({"layout": layout, "vintages": log}, indent=2))


if __name__ == "__main__":
    main()
