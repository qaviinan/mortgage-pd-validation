"""Synthetic Freddie-layout fixtures for tests and smoke runs (never for results).

Writes sample_orig_<year>.txt and sample_svcg_<year>.txt in the 31/35-column pipe-delimited
layout, zipped as sample_<year>.zip by default so the real ingest path is exercised.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from pdval.layout import ORIG_COLUMNS, PERF_COLUMNS


def _ym_add(ym: int, k: int) -> int:
    y, m = divmod(ym, 100)
    idx = y * 12 + (m - 1) + k
    return (idx // 12) * 100 + (idx % 12) + 1


def make_vintage(year: int, n_loans: int, out_dir: Path, *, seed: int = 0, default_rate: float = 0.05,
                 prepay_rate: float = 0.30, horizon_months: int = 48, zip_it: bool = True) -> Path:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    orig_lines, perf_lines = [], []
    for i in range(n_loans):
        q = rng.integers(1, 5)
        loan_id = f"F{year % 100:02d}Q{q}{i:07d}"
        fpd = _ym_add(year * 100 + 1, int((q - 1) * 3 + rng.integers(0, 3)))
        fico = int(np.clip(rng.normal(740, 45), 600, 850))
        ltv = int(np.clip(rng.normal(75, 15), 20, 97))
        dti = int(np.clip(rng.normal(34, 9), 10, 50))
        risk = 1 / (1 + np.exp(-(-3.5 - 0.02 * (fico - 740) + 0.03 * (ltv - 75) + 0.03 * (dti - 34))))
        row = {c: "" for c in ORIG_COLUMNS}
        row.update(
            fico=fico, first_payment_date=fpd, first_time_homebuyer=rng.choice(["Y", "N"]),
            maturity_date=_ym_add(fpd, 359), msa=rng.choice([12060, 31080, 35620, ""]),
            mi_pct=0 if ltv <= 80 else 25, num_units=1, occupancy=rng.choice(["P", "P", "P", "I", "S"]),
            cltv=ltv, dti=dti, orig_upb=int(rng.integers(80, 600)) * 1000, ltv=ltv,
            orig_rate=round(float(rng.normal(4.5, 0.6)), 3), channel=rng.choice(["R", "B", "C"]),
            ppm_flag="N", amortization_type="FRM", state=rng.choice(["CA", "TX", "FL", "NY", "OH"]),
            property_type=rng.choice(["SF", "PU", "CO"]), zip3=f"{rng.integers(100, 999)}", loan_id=loan_id,
            loan_purpose=rng.choice(["P", "C", "N"]), orig_term=360, num_borrowers=rng.choice([1, 2]),
            seller_name="Other sellers", super_conforming="N", special_program="9", harp_indicator="",
            valuation_method=2, io_flag="N", vantagescore4="",
        )
        orig_lines.append("|".join(str(row[c]) for c in ORIG_COLUMNS))

        u = rng.random()
        default_month = int(rng.integers(6, horizon_months)) if u < default_rate * risk / 0.05 else None
        prepay_month = int(rng.integers(3, horizon_months)) if (default_month is None and rng.random() < prepay_rate) else None
        end = min(x for x in (default_month, prepay_month, horizon_months) if x is not None)
        upb = row["orig_upb"]
        for age in range(1, end + 1):
            period = _ym_add(fpd, age - 1)
            status = "00"
            zb, zbd = "", ""
            if default_month is not None and age >= default_month - 3:
                status = f"{min(age - (default_month - 3), 6):02d}"
            if prepay_month is not None and age == prepay_month:
                zb, zbd = "01", str(period)
            if default_month is not None and age == default_month and rng.random() < 0.2:
                zb, zbd = "09", str(period)   # REO disposition alongside 90+ DPD
            prow = {c: "" for c in PERF_COLUMNS}
            prow.update(
                loan_id=loan_id, period=period, current_upb=round(upb * (1 - age / 400), 2),
                dlq_status=status, loan_age=age, months_to_maturity=360 - age, modification_flag="",
                zero_balance_code=zb, zero_balance_date=zbd, current_rate=row["orig_rate"],
                current_nib_upb=0, disaster_flag="", assistance_plan="", servicer_name="Other servicers",
            )
            perf_lines.append("|".join(str(prow[c]) for c in PERF_COLUMNS))

    orig_txt = out_dir / f"sample_orig_{year}.txt"
    perf_txt = out_dir / f"sample_svcg_{year}.txt"
    orig_txt.write_text("\n".join(orig_lines) + "\n")
    perf_txt.write_text("\n".join(perf_lines) + "\n")
    if not zip_it:
        return out_dir
    zpath = out_dir / f"sample_{year}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(orig_txt, orig_txt.name)
        z.write(perf_txt, perf_txt.name)
    orig_txt.unlink()
    perf_txt.unlink()
    return zpath
