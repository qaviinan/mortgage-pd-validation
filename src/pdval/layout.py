"""Column layout for the Freddie Mac Single-Family Loan-Level Dataset.

The raw files are pipe-delimited with no header row. Column order comes from Freddie's
file-layout spreadsheet (docs/reference/file_layout_july_2026.xlsx). Each position gets a
snake_case name that is checked against the spreadsheet, so a layout change in a future
release raises an error instead of mislabelling columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# (position, name, prefix of the spreadsheet's "Attribute Name", matched case-insensitively)
ORIG_FIELDS: list[tuple[int, str, str]] = [
    (1, "fico", "Classic FICO"),
    (2, "first_payment_date", "First Payment Date"),
    (3, "first_time_homebuyer", "First Time Homebuyer"),
    (4, "maturity_date", "Maturity Date"),
    (5, "msa", "Metropolitan Statistical Area"),
    (6, "mi_pct", "Mortgage Insurance Percentage"),
    (7, "num_units", "Number of Units"),
    (8, "occupancy", "Occupancy Status"),
    (9, "cltv", "Original Combined Loan-to-Value"),
    (10, "dti", "Original Debt-to-Income"),
    (11, "orig_upb", "Original UPB"),
    (12, "ltv", "Original Loan-to-Value"),
    (13, "orig_rate", "Original Interest Rate"),
    (14, "channel", "Channel"),
    (15, "ppm_flag", "Prepayment Penalty"),
    (16, "amortization_type", "Amortization Type"),
    (17, "state", "Property State"),
    (18, "property_type", "Property Type"),
    (19, "zip3", "Postal Code"),
    (20, "loan_id", "Loan Identifier"),
    (21, "loan_purpose", "Loan Purpose"),
    (22, "orig_term", "Original Loan Term"),
    (23, "num_borrowers", "Number of Borrowers"),
    (24, "seller_name", "Seller Name"),
    (25, "super_conforming", "Super Conforming"),
    (26, "pre_harp_loan_id", "Pre-HARP Loan Sequence"),
    (27, "special_program", "Special Eligibility Program"),
    (28, "harp_indicator", "HARP Indicator"),
    (29, "valuation_method", "Property Valuation Method"),
    (30, "io_flag", "Interest Only"),
    (31, "vantagescore4", "VantageScore"),
]

PERF_FIELDS: list[tuple[int, str, str]] = [
    (1, "loan_id", "Loan Identifier"),
    (2, "period", "Period"),
    (3, "current_upb", "Current Actual UPB"),
    (4, "dlq_status", "Current Loan Delinquency Status"),
    (5, "loan_age", "Loan Age"),
    (6, "months_to_maturity", "Remaining Months to Legal Maturity"),
    (7, "defect_settlement_date", "Underwriting Defect"),
    (8, "modification_flag", "Modification Flag"),
    (9, "zero_balance_code", "Zero Balance Code"),
    (10, "zero_balance_date", "Zero Balance Effective Date"),
    (11, "current_rate", "Current Interest Rate"),
    (12, "current_nib_upb", "Current Non-Interest Bearing UPB"),
    (13, "ddlpi", "Due Date of Last Paid Installment"),
    (14, "mi_recoveries", "MI Recoveries"),
    (15, "net_sales_proceeds", "Net Sales Proceeds"),
    (16, "non_mi_recoveries", "Non MI Recoveries"),
    (17, "total_expenses", "Total Expenses"),
    (18, "legal_costs", "Legal Costs"),
    (19, "maintenance_costs", "Maintenance and Preservation"),
    (20, "taxes_insurance", "Taxes and Insurance"),
    (21, "misc_expenses", "Miscellaneous Expenses"),
    (22, "actual_loss", "Actual Loss"),
    (23, "cum_modification_costs", "Cumulative Modification Costs"),
    (24, "step_rate_flag", "Interest Rate Step"),
    (25, "payment_deferral_flag", "Payment Deferral"),
    (26, "eltv", "Estimated Loan-to-Value"),
    (27, "zb_removal_upb", "Zero Balance Removal UPB"),
    (28, "delinquent_accrued_interest", "Delinquent Accrued Interest"),
    (29, "disaster_flag", "Delinquency Due to Disaster"),
    (30, "assistance_plan", "Borrower Assistance Plan"),
    (31, "current_period_mod_costs", "Current Period Modification Costs"),
    (32, "current_ib_upb", "Current Interest Bearing UPB"),
    (33, "mi_cancellation_flag", "Mortgage Insurance Cancellation"),
    (34, "servicer_name", "Servicer Name"),
    (35, "bankruptcy_cramdown_costs", "Bankruptcy Cramdown"),
]

ORIG_COLUMNS = [name for _, name, _ in ORIG_FIELDS]
PERF_COLUMNS = [name for _, name, _ in PERF_FIELDS]

# Typed columns in the processed parquet. Everything else stays VARCHAR so a stray value
# can't abort ingestion.
ORIG_CASTS: dict[str, str] = {
    "fico": "INTEGER",
    "first_payment_date": "INTEGER",   # YYYYMM
    "maturity_date": "INTEGER",
    "msa": "INTEGER",
    "mi_pct": "INTEGER",
    "num_units": "INTEGER",
    "cltv": "INTEGER",
    "dti": "INTEGER",
    "orig_upb": "DOUBLE",
    "ltv": "INTEGER",
    "orig_rate": "DOUBLE",
    "orig_term": "INTEGER",
    "num_borrowers": "INTEGER",
    "valuation_method": "INTEGER",
    "vantagescore4": "INTEGER",
}
PERF_CASTS: dict[str, str] = {
    "period": "INTEGER",
    "current_upb": "DOUBLE",
    "loan_age": "INTEGER",
    "months_to_maturity": "INTEGER",
    "defect_settlement_date": "INTEGER",
    "zero_balance_code": "INTEGER",
    "zero_balance_date": "INTEGER",
    "current_rate": "DOUBLE",
    "current_nib_upb": "DOUBLE",
    "ddlpi": "INTEGER",
    "actual_loss": "DOUBLE",
    "eltv": "INTEGER",
    "zb_removal_upb": "DOUBLE",
    "current_ib_upb": "DOUBLE",
}


@dataclass(frozen=True)
class LayoutMismatch(Exception):
    file: str
    position: int
    expected_prefix: str
    found: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return (
            f"{self.file} position {self.position}: expected attribute starting with "
            f"{self.expected_prefix!r}, spreadsheet says {self.found!r}"
        )


def _read_sheet(xlsx: Path, sheet: str) -> list[tuple[int, str]]:
    import openpyxl

    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    ws = wb[sheet]
    out: list[tuple[int, str]] = []
    for row in ws.iter_rows(values_only=True):
        if not row or row[0] is None:
            continue
        try:
            pos = int(row[0])
        except (TypeError, ValueError):
            continue  # title / header rows
        out.append((pos, str(row[1]).strip()))
    return out


def verify_layout(xlsx: Path) -> dict[str, int]:
    """Check the canonical field lists against Freddie's spreadsheet.

    Returns {"orig": n_fields, "perf": n_fields}. Raises LayoutMismatch on any drift.
    """
    result: dict[str, int] = {}
    for sheet, fields, key in (
        ("Origination Data File", ORIG_FIELDS, "orig"),
        ("Monthly Performance Data File", PERF_FIELDS, "perf"),
    ):
        found = dict(_read_sheet(xlsx, sheet))
        for pos, _, prefix in fields:
            actual = found.get(pos, "<missing>")
            if not actual.lower().startswith(prefix.lower()):
                raise LayoutMismatch(sheet, pos, prefix, actual)
        extra = sorted(set(found) - {p for p, _, _ in fields})
        if extra:
            raise LayoutMismatch(sheet, extra[0], "<no canonical field>", found[extra[0]])
        result[key] = len(fields)
    return result
