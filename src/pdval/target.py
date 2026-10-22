"""Default-target construction with explicit censoring.

Given one vintage's origination + performance tables, produce one row per loan:

    loan_id, first_payment_date, max_age, default_age, default_type, payoff_age,
    censor_age, defect, default_in_forbearance, y, exclusion_reason

`y` is 1/0 for loans observable over the full window and NULL for excluded loans, with
`exclusion_reason` in {"defect", "censored", "incomplete_window"}.

Loan age is computed from First Payment Date (months between, +1), not from the Loan Age
field, because Freddie resets Loan Age on modification.

The logic is one DuckDB query, so the same code runs on a 15M-row vintage and on the
unit-test fixtures.
"""

from __future__ import annotations

import duckdb
import pandas as pd


def _sql(window: int, dpd_code_min: int, credit_zb: list[int], payoff_zb: list[int],
         censor_zb: list[int], defect_zb: list[int], cutoff: int | None,
         exclude_forbearance_dpd: bool, forbearance_lookback: int, exclude_disaster_dpd: bool) -> str:
    credit = ", ".join(map(str, credit_zb))
    payoff = ", ".join(map(str, payoff_zb))
    censor = ", ".join(map(str, censor_zb))
    defect = ", ".join(map(str, defect_zb))
    cutoff_filter = f"AND period <= {cutoff}" if cutoff else ""
    # A 90+ DPD month counts as a default only if the loan is not in a forbearance plan
    # (Borrower Assistance Plan = 'F') that month. Credit-event terminations (ZB 02/03/09)
    # always count.
    # forbearance_lookback extends the guard to N months after a plan ends. In the COVID data
    # the plan flag drops a month or two before the deferral is booked, so the loan shows 90+
    # with no plan and then cures.
    # exclude_disaster_dpd applies the same guard to "Delinquency Due to Disaster" months, which
    # servicers used for COVID hardship when no formal plan was booked.
    hardship = []
    if exclude_forbearance_dpd:
        hardship.append("pf.assistance_plan = 'F'")
    if exclude_disaster_dpd:
        hardship.append("pf.disaster_flag = 'Y'")
    hardship_expr = " OR ".join(hardship) if hardship else "FALSE"
    forb_guard = "AND NOT in_or_after_hardship" if hardship else ""
    return f"""
    WITH p0 AS (
      SELECT
        pf.loan_id,
        o.first_payment_date,
        pf.period,
        ((pf.period // 100) * 12 + (pf.period % 100))
          - ((o.first_payment_date // 100) * 12 + (o.first_payment_date % 100)) + 1 AS age,
        pf.dlq_status,
        pf.zero_balance_code,
        pf.assistance_plan,
        pf.disaster_flag,
        COALESCE(max(CASE WHEN {hardship_expr} THEN 1 ELSE 0 END) OVER (
            PARTITION BY pf.loan_id ORDER BY pf.period
            ROWS BETWEEN {forbearance_lookback} PRECEDING AND CURRENT ROW), 0) = 1 AS in_or_after_hardship
      FROM perf pf
      JOIN orig o USING (loan_id)
      WHERE pf.period IS NOT NULL {cutoff_filter}
    ),
    p AS (
      SELECT *,
        CASE
          WHEN dlq_status = 'RA' THEN TRUE
          WHEN TRY_CAST(dlq_status AS INTEGER) >= {dpd_code_min} {forb_guard} THEN TRUE
          ELSE FALSE
        END AS is_dpd_event,
        zero_balance_code IN ({credit}) AS is_credit_zb
      FROM p0
    ),
    ev AS (
      SELECT
        loan_id,
        any_value(first_payment_date) AS first_payment_date,
        max(age) AS max_age,
        min(CASE WHEN is_dpd_event OR is_credit_zb THEN age END) AS default_age,
        min(CASE WHEN zero_balance_code IN ({payoff}) THEN age END) AS payoff_age,
        min(CASE WHEN zero_balance_code IN ({censor}) THEN age END) AS censor_age,
        COALESCE(bool_or(zero_balance_code IN ({defect})), FALSE) AS defect
      FROM p GROUP BY loan_id
    ),
    typed AS (
      SELECT e.*,
        (SELECT CASE WHEN bool_or(is_dpd_event) THEN 'dpd' ELSE 'zb' END
           FROM p WHERE p.loan_id = e.loan_id AND p.age = e.default_age) AS default_type,
        COALESCE((SELECT bool_or(assistance_plan = 'F')
           FROM p WHERE p.loan_id = e.loan_id AND p.age = e.default_age), FALSE) AS default_in_forbearance,
        COALESCE((SELECT bool_or(disaster_flag = 'Y')
           FROM p WHERE p.loan_id = e.loan_id AND p.age = e.default_age), FALSE) AS default_in_disaster
      FROM ev e
    )
    SELECT *,
      CASE
        WHEN defect THEN NULL
        WHEN default_age IS NOT NULL AND default_age <= {window} THEN 1
        WHEN max_age >= {window} THEN 0
        WHEN payoff_age IS NOT NULL AND payoff_age <= {window} THEN 0
        ELSE NULL
      END AS y,
      CASE
        WHEN defect THEN 'defect'
        WHEN default_age IS NOT NULL AND default_age <= {window} THEN NULL
        WHEN max_age >= {window} THEN NULL
        WHEN payoff_age IS NOT NULL AND payoff_age <= {window} THEN NULL
        WHEN censor_age IS NOT NULL AND censor_age <= {window} THEN 'censored'
        ELSE 'incomplete_window'
      END AS exclusion_reason
    FROM typed
    ORDER BY loan_id
    """


def build_target(
    orig: pd.DataFrame | str,
    perf: pd.DataFrame | str,
    *,
    window: int = 36,
    dpd_threshold: int = 90,
    credit_event_zb_codes: list[int] = (2, 3, 9),
    payoff_zb_codes: list[int] = (1,),
    censor_zb_codes: list[int] = (15, 16),
    defect_zb_codes: list[int] = (96,),
    performance_cutoff: int | None = None,
    exclude_forbearance_dpd: bool = True,
    forbearance_lookback: int = 3,
    exclude_disaster_dpd: bool = True,
    con: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Return the per-loan target frame.

    `orig` / `perf` may be pandas DataFrames or parquet paths. `dpd_threshold` is in days;
    Freddie codes 30-day buckets, so 90 -> status code >= 3, 60 -> >= 2.
    `exclude_forbearance_dpd=True` (primary) ignores DPD months in a forbearance plan or within
    `forbearance_lookback` months after one. Turning off both guards gives the pure ever-90+
    sensitivity.
    """
    own = con is None
    con = con or duckdb.connect()
    try:
        for name, obj in (("orig", orig), ("perf", perf)):
            if isinstance(obj, pd.DataFrame):
                con.register(name, obj)
            else:
                con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM '{obj}'")
        q = _sql(
            window=window,
            dpd_code_min=max(1, dpd_threshold // 30),
            credit_zb=list(credit_event_zb_codes),
            payoff_zb=list(payoff_zb_codes),
            censor_zb=list(censor_zb_codes),
            defect_zb=list(defect_zb_codes),
            cutoff=performance_cutoff,
            exclude_forbearance_dpd=exclude_forbearance_dpd,
            forbearance_lookback=max(0, int(forbearance_lookback)),
            exclude_disaster_dpd=exclude_disaster_dpd,
        )
        return con.execute(q).df()
    finally:
        if own:
            con.close()


def waterfall(target: pd.DataFrame) -> pd.DataFrame:
    """Row-count waterfall for the report appendix."""
    n = len(target)
    rows = [("loans in vintage sample", n)]
    for reason in ("defect", "censored", "incomplete_window"):
        k = int((target["exclusion_reason"] == reason).sum())
        rows.append((f"  minus {reason}", -k))
        n -= k
    rows.append(("observable loans", n))
    rows.append(("  of which default (y=1)", int((target["y"] == 1).sum())))
    return pd.DataFrame(rows, columns=["step", "loans"])
