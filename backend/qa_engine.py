"""Natural-language Q&A engine over a single candidate's resume.

The engine grounds every answer in the structured Candidate data and a
pre-computed employment timeline. It never asks the LLM to perform date
arithmetic or infer information beyond what is explicitly stated.

Anti-hallucination strategy (see README > Prompt Strategy for rationale):
  1. Candidate data is serialized as structured JSON into the prompt context.
  2. Employment gaps and total experience are pre-computed in Python and
     injected as verified facts.
  3. The system prompt explicitly forbids outside knowledge and requires
     "not stated" refusals for missing information.
  4. A few-shot refusal example anchors the refusal behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.llm import get_llm_response
from backend.models import Candidate, EmploymentTimeline, QARequest, QAResponse
from backend.parsing import compute_timeline

_PROMPT_DIR = Path(__file__).parent / "prompts"
_QA_SYSTEM_PROMPT_PATH = _PROMPT_DIR / "qa_system_prompt.txt"


def _load_system_prompt() -> str:
    return _QA_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _serialize_candidate(candidate: Candidate) -> str:
    """Serialize the Candidate into structured JSON for prompt injection.

    Using structured JSON (not a re-narrated paragraph) so the model can
    reference specific fields and values directly, reducing paraphrasing
    drift that leads to hallucination.
    """
    return candidate.model_dump_json(indent=2)


def _serialize_timeline(timeline: EmploymentTimeline) -> str:
    """Serialize the pre-computed timeline for prompt injection."""
    lines = [
        f"Total years of professional experience: {timeline.total_years_experience}",
        "",
        "Employment entries (chronological):",
    ]
    for i, entry in enumerate(timeline.entries_chronological, 1):
        lines.append(
            f"  {i}. {entry.company} — {entry.title} "
            f"({entry.start_date} to {entry.end_date}), {entry.location}"
        )

    if timeline.gaps:
        lines.append("")
        lines.append("Detected employment gaps (computed via date arithmetic):")
        for gap in timeline.gaps:
            lines.append(
                f"  - GAP: {gap.duration_months} months "
                f"({gap.gap_start} to {gap.gap_end}), "
                f"between [{gap.preceding_job}] and [{gap.following_job}]"
            )
    else:
        lines.append("")
        lines.append("No significant employment gaps detected.")

    return "\n".join(lines)


def ask(candidate: Candidate, question: str) -> QAResponse:
    """Answer a recruiter's free-text question about the candidate.

    Parameters
    ----------
    candidate : Candidate
        The normalized candidate data.
    question : str
        The recruiter's natural-language question.

    Returns
    -------
    QAResponse
        Grounded answer plus metadata.
    """
    timeline = compute_timeline(candidate)

    system_prompt = _load_system_prompt().format(
        candidate_data=_serialize_candidate(candidate),
        timeline_data=_serialize_timeline(timeline),
    )

    answer = get_llm_response(system_prompt, question)

    return QAResponse(
        question=question,
        answer=answer.strip(),
        grounded=True,
    )
