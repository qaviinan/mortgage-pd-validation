"""Findings register: a validation result becomes a Finding when it breaches a configured threshold."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

SEVERITY_DEFS = {
    "High": "Materially affects fitness for intended use; must be remediated or use restricted before reliance.",
    "Medium": "Limits use or reliability in identifiable conditions; remediate within one cycle, monitor meanwhile.",
    "Low": "Documentation, minor deviation, or observation; address in next redevelopment.",
}


@dataclass
class Finding:
    id: str
    area: str
    severity: str
    title: str
    evidence: str
    threshold: str
    recommendation: str
    refs: list[str] = field(default_factory=list)
    status: str = "Open"


class Register:
    def __init__(self):
        self.items: list[Finding] = []
        self._n = 0

    def add(self, area: str, severity: str, title: str, evidence: str, threshold: str, recommendation: str,
            refs: list[str] | None = None, status: str = "Open") -> Finding:
        assert severity in SEVERITY_DEFS
        self._n += 1
        f = Finding(f"F{self._n:02d}", area, severity, title, evidence, threshold, recommendation, refs or [], status)
        self.items.append(f)
        return f

    def to_frame(self) -> pd.DataFrame:
        order = {"High": 0, "Medium": 1, "Low": 2}
        df = pd.DataFrame([asdict(f) for f in self.items])
        if df.empty:
            return df
        df["refs"] = df["refs"].apply("; ".join)
        return df.sort_values(["severity", "id"], key=lambda s: s.map(order) if s.name == "severity" else s)

    def counts(self) -> dict:
        return {k: sum(1 for f in self.items if f.severity == k) for k in SEVERITY_DEFS}
