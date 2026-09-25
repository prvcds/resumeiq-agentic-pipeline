"""Resume ingestion and normalization.

Loads a mock resume from JSON and normalizes it into a typed Candidate object.
Structured as a single-module seam: to support real PDF/OCR ingestion, replace
the `load_candidate()` function body while keeping its return type unchanged.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from backend.models import (
    Candidate,
    EmploymentGap,
    EmploymentTimeline,
)

# Path to mock data — resolved relative to this file so it works regardless
# of the working directory used to start the server.
_MOCK_DATA_DIR = Path(__file__).parent / "mock_data"
_DEFAULT_RESUME = _MOCK_DATA_DIR / "candidate_resume.json"


def load_candidate(source: Path | str | None = None) -> Candidate:
    """Load and validate a candidate resume into the Candidate schema.

    Parameters
    ----------
    source : Path, str, or None
        Path to a JSON file, a JSON string, or None to use the built-in mock.
        **Seam for future extension**: swap this function's body with a
        PDF-parsing / OCR pipeline; the rest of the application depends only
        on the returned ``Candidate`` object.

    Returns
    -------
    Candidate
        Fully validated and typed candidate data.
    """
    if source is None:
        source = _DEFAULT_RESUME

    path = Path(source)
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        # Treat as a raw JSON string (useful for future upload endpoint)
        raw = json.loads(str(source))

    return Candidate.model_validate(raw)


# ---------------------------------------------------------------------------
# Employment timeline computation (date arithmetic in code, not LLM)
# ---------------------------------------------------------------------------

def _parse_month(date_str: str) -> datetime:
    """Parse a 'YYYY-MM' string into a datetime (first of the month)."""
    return datetime.strptime(date_str, "%Y-%m")


def _months_between(start: datetime, end: datetime) -> int:
    """Compute the number of whole months between two dates."""
    return (end.year - start.year) * 12 + (end.month - start.month)


def compute_timeline(candidate: Candidate) -> EmploymentTimeline:
    """Build a pre-computed employment timeline with gaps.

    This is the single most important reliability decision in the system:
    date arithmetic is handled here in deterministic Python code, NOT by the
    LLM. The resulting timeline is injected into prompts as verified fact.
    """
    # Sort employment entries chronologically by start date
    sorted_entries = sorted(
        candidate.employment,
        key=lambda e: _parse_month(e.start_date),
    )

    # Compute total experience
    total_months = 0
    for entry in sorted_entries:
        start = _parse_month(entry.start_date)
        end = _parse_month(entry.end_date)
        total_months += _months_between(start, end)

    total_years = round(total_months / 12, 1)

    # Detect gaps between consecutive entries
    gaps: list[EmploymentGap] = []
    for i in range(len(sorted_entries) - 1):
        current_end = _parse_month(sorted_entries[i].end_date)
        next_start = _parse_month(sorted_entries[i + 1].start_date)
        gap_months = _months_between(current_end, next_start)

        # Only flag gaps longer than 2 months (allow normal transition time)
        if gap_months > 2:
            gaps.append(
                EmploymentGap(
                    gap_start=sorted_entries[i].end_date,
                    gap_end=sorted_entries[i + 1].start_date,
                    duration_months=gap_months,
                    preceding_job=f"{sorted_entries[i].company} — {sorted_entries[i].title}",
                    following_job=f"{sorted_entries[i + 1].company} — {sorted_entries[i + 1].title}",
                )
            )

    return EmploymentTimeline(
        entries_chronological=sorted_entries,
        total_years_experience=total_years,
        gaps=gaps,
    )
