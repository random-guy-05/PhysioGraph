"""Audit logging for ETL pipeline transforms.

Records row counts, stay counts, and metadata at every pipeline step
to enable reproducibility checks and parity validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AuditLogger:
    """Records structured audit entries for each pipeline transform step.

    Each entry captures the step name, dataset, row/stay counts,
    and optional details. The full log is serialized as audit.json
    at pipeline completion.

    Attributes:
        dataset: Dataset identifier (e.g., "mimic", "eicu").
        entries: Accumulated list of audit entry dictionaries.
    """

    dataset: str
    entries: list[dict[str, object]] = field(default_factory=list)

    def log(
        self,
        step: str,
        *,
        row_count: int | None = None,
        stay_count: int | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        """Append an audit entry with the current UTC timestamp.

        Args:
            step: Human-readable step identifier (e.g.,
                "mimic_pressors_extracted").
            row_count: Number of rows at this step.
            stay_count: Number of unique stays at this step.
            details: Arbitrary additional metadata.
        """
        entry: dict[str, object] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": self.dataset,
            "step": step,
        }
        if row_count is not None:
            entry["row_count"] = int(row_count)
        if stay_count is not None:
            entry["stay_count"] = int(stay_count)
        if details:
            entry["details"] = details
        self.entries.append(entry)
