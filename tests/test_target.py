"""Hand-built loan histories covering every branch of the target definition."""

import pandas as pd
import pytest

from pdval.target import build_target, waterfall

FPD = 201501  # first payment Jan 2015


def _perf(loan_id, statuses, zb=None, zb_at=None, plan=None, plan_at=None, disaster_at=None):
    """statuses: list of dlq codes month by month starting at age 1; plan applies at age plan_at."""
    rows = []
    for i, s in enumerate(statuses):
        y, m = divmod(FPD, 100)
        idx = y * 12 + (m - 1) + i
        period = (idx // 12) * 100 + (idx % 12) + 1
        rows.append(
            {
                "loan_id": loan_id,
                "period": period,
                "dlq_status": s,
                "zero_balance_code": (zb if zb_at is not None and i + 1 == zb_at else None),
                "assistance_plan": (plan if plan and i + 1 == plan_at else None),
                "disaster_flag": ("Y" if disaster_at is not None and i + 1 == disaster_at else None),
            }
        )
    return rows


@pytest.fixture
def frames():
    cases = {
        "A_default_m12": _perf("A", ["00"] * 8 + ["01", "02", "03", "04"]),           # D90 at age 11
        "B_prepay_m20": _perf("B", ["00"] * 20, zb=1, zb_at=20),                      # payoff, observed 0
        "C_active_36": _perf("C", ["00"] * 36),                                       # survives window
        "D_active_30_cutoff": _perf("D", ["00"] * 30),                                # window incomplete
        "E_censored_zb15": _perf("E", ["00"] * 20, zb=15, zb_at=20),                  # whole-loan sale
        "F_default_m40": _perf("F", ["00"] * 37 + ["01", "02", "03"]),                # D90 at 40 -> 0
        "G_reo_ra_m10": _perf("G", ["00"] * 9 + ["RA"]),                              # RA at age 10
        "H_zb03_no_dpd": _perf("H", ["00"] * 8, zb=3, zb_at=8),                       # short sale w/o D90
        "I_defect": _perf("I", ["00"] * 5, zb=96, zb_at=5),                           # excluded
        "J_forbearance": _perf("J", ["00"] * 8 + ["01", "02", "03"] + ["00"] * 25, plan="F", plan_at=11),  # D90 in plan, then cures
        "K_60dpd_only": _perf("K", ["00"] * 30 + ["01", "02", "01", "00", "00", "00"]),  # never 90+
        "L_post_forb_lag": _perf("L", ["00"] * 8 + ["01", "02", "03", "04", "05"] + ["00"] * 23, plan="F", plan_at=10),  # plan ends age 10, D90 at 11-13
        "M_forb_long_ago": _perf("M", ["00"] * 4 + ["01"] + ["00"] * 7 + ["01", "02", "03"] + ["00"] * 21, plan="F", plan_at=5),  # plan at 5, D90 at 15
        "N_disaster_d90": _perf("N", ["00"] * 8 + ["01", "02", "03"] + ["00"] * 25, disaster_at=11),   # D90 flagged disaster
        "O_disaster_then_d90_later": _perf("O", ["00"] * 4 + ["01"] + ["00"] * 7 + ["01", "02", "03"] + ["00"] * 21, disaster_at=5),
    }
    perf = pd.DataFrame([r for rows in cases.values() for r in rows])
    orig = pd.DataFrame({"loan_id": sorted({r["loan_id"] for r in perf.to_dict("records")})})
    orig["first_payment_date"] = FPD
    return orig, perf


def test_primary_definition(frames):
    orig, perf = frames
    t = build_target(orig, perf, window=36).set_index("loan_id")

    assert t.loc["A", "y"] == 1 and t.loc["A", "default_age"] == 11 and t.loc["A", "default_type"] == "dpd"
    assert t.loc["B", "y"] == 0 and t.loc["B", "payoff_age"] == 20
    assert t.loc["C", "y"] == 0 and t.loc["C", "max_age"] == 36
    assert pd.isna(t.loc["D", "y"]) and t.loc["D", "exclusion_reason"] == "incomplete_window"
    assert pd.isna(t.loc["E", "y"]) and t.loc["E", "exclusion_reason"] == "censored"
    assert t.loc["F", "y"] == 0 and t.loc["F", "default_age"] == 40
    assert t.loc["G", "y"] == 1 and t.loc["G", "default_age"] == 10
    assert t.loc["H", "y"] == 1 and t.loc["H", "default_type"] == "zb"
    assert pd.isna(t.loc["I", "y"]) and t.loc["I", "exclusion_reason"] == "defect"
    assert t.loc["J", "y"] == 0 and pd.isna(t.loc["J", "default_age"])   # D90 under forbearance: not a default
    assert t.loc["K", "y"] == 0
    assert t.loc["L", "y"] == 0 and pd.isna(t.loc["L", "default_age"])   # D90 within 3m after plan: guarded
    assert t.loc["M", "y"] == 1 and t.loc["M", "default_age"] == 15       # plan long before: counts
    assert t.loc["N", "y"] == 0 and pd.isna(t.loc["N", "default_age"])   # disaster-flagged D90: guarded
    assert t.loc["O", "y"] == 1 and t.loc["O", "default_age"] == 15
    assert not bool(t.loc["A", "default_in_forbearance"])


def test_disaster_guard_off(frames):
    orig, perf = frames
    t = build_target(orig, perf, window=36, exclude_disaster_dpd=False).set_index("loan_id")
    assert t.loc["N", "y"] == 1 and bool(t.loc["N", "default_in_disaster"])
    assert t.loc["J", "y"] == 0


def test_lookback_zero_counts_post_plan_dpd(frames):
    orig, perf = frames
    t = build_target(orig, perf, window=36, forbearance_lookback=0).set_index("loan_id")
    assert t.loc["L", "y"] == 1 and t.loc["L", "default_age"] == 11
    assert t.loc["J", "y"] == 0


def test_pure_dpd_sensitivity_counts_forbearance(frames):
    orig, perf = frames
    t = build_target(orig, perf, window=36, exclude_forbearance_dpd=False).set_index("loan_id")
    assert t.loc["J", "y"] == 1 and bool(t.loc["J", "default_in_forbearance"])
    assert t.loc["A", "y"] == 1 and not bool(t.loc["A", "default_in_forbearance"])


def test_sensitivities(frames):
    orig, perf = frames
    t24 = build_target(orig, perf, window=24).set_index("loan_id")
    assert t24.loc["F", "y"] == 0 and t24.loc["D", "y"] == 0        # D now observable at 24
    assert pd.isna(t24.loc["E", "y"]) and t24.loc["E", "exclusion_reason"] == "censored"  # ZB15 at 20 < 24

    t60 = build_target(orig, perf, window=36, dpd_threshold=60).set_index("loan_id")
    assert t60.loc["K", "y"] == 1 and t60.loc["K", "default_age"] == 32
    assert t60.loc["A", "default_age"] == 10


def test_cutoff_truncates_history(frames):
    orig, perf = frames
    # Cut performance at age 9 (Sep 2015): A has not yet defaulted; only H (short sale at
    # age 8) has a resolved outcome, everyone else is incomplete or excluded.
    t = build_target(orig, perf, window=36, performance_cutoff=201509).set_index("loan_id")
    assert pd.isna(t.loc["A", "default_age"])
    assert t.loc["H", "y"] == 1
    others = t.drop(index="H")
    assert others["y"].isna().all()
    assert (others["exclusion_reason"].fillna("") != "").all()


def test_waterfall_adds_up(frames):
    orig, perf = frames
    t = build_target(orig, perf, window=36)
    w = waterfall(t).set_index("step")["loans"]
    assert w["loans in vintage sample"] == 15
    assert w["observable loans"] == 12         # A B C F G H J K L M N O
    assert w["  of which default (y=1)"] == 5  # A G H M O
