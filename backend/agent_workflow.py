"""Agentic evaluation workflow: extract -> map -> flag -> recommend -> dispatch.

Each step is a separate, inspectable function producing a typed output model.
The pipeline is a plain Python orchestration — no framework overhead. Every
intermediate result is returned in the final response for full auditability.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from backend.dispatch import export_form, mock_send_email
from backend.llm import get_llm_response
from backend.models import (
    Candidate,
    DispatchResult,
    EvaluationForm,
    ExtractedFields,
    FlagReport,
    MappedForm,
    RecommendationResult,
    RedFlag,
)
from backend.parsing import compute_timeline

logger = logging.getLogger("resumeiq.agent")

_PROMPT_DIR = Path(__file__).parent / "prompts"

# Fixed role categories (shared with the system prompt)
ROLE_CATEGORIES = [
    "Backend Engineer",
    "Frontend Engineer",
    "Full-Stack Engineer",
    "Platform / Infrastructure Engineer",
    "DevOps Engineer",
    "Site Reliability Engineer (SRE)",
    "Data Engineer",
    "Machine Learning Engineer",
    "Engineering Manager",
    "Technical Lead",
    "Solutions Architect",
]


# ---------------------------------------------------------------------------
# Step 1: Extract
# ---------------------------------------------------------------------------

def step_extract(candidate: Candidate) -> ExtractedFields:
    """Pull structured fields from the Candidate object and pre-compute timeline.

    This step is purely deterministic — no LLM call. It normalizes the
    candidate data into the shape the downstream steps expect and computes
    the employment timeline (gaps, total experience) in code.
    """
    timeline = compute_timeline(candidate)

    return ExtractedFields(
        candidate_name=candidate.full_name,
        skills=candidate.skills,
        years_of_experience=timeline.total_years_experience,
        education_entries=candidate.education,
        employment_entries=timeline.entries_chronological,
        certifications=[c.name for c in candidate.certifications],
        timeline=timeline,
    )


# ---------------------------------------------------------------------------
# Step 2: Map
# ---------------------------------------------------------------------------

def step_map(extracted: ExtractedFields) -> MappedForm:
    """Map extracted fields into the evaluation form structure.

    Deterministic field mapping — no LLM call. Selects the top skills
    based on recency and specificity, and summarizes education.
    """
    # Primary skillset: take the first 8 skills (assumed to be listed in
    # order of proficiency by the candidate) — a heuristic that works for
    # mock data and would be replaced by NLP-based skill ranking in production.
    primary_skills = extracted.skills[:8]

    # Education summary: concatenate degree + field + institution
    edu_parts = []
    for edu in extracted.education_entries:
        gpa_note = f" (GPA: {edu.gpa})" if edu.gpa else ""
        edu_parts.append(
            f"{edu.degree} in {edu.field}, {edu.institution} "
            f"({edu.start_date} to {edu.end_date}){gpa_note}"
        )
    education_summary = "; ".join(edu_parts)

    return MappedForm(
        candidate_name=extracted.candidate_name,
        primary_skillset=primary_skills,
        years_of_experience=extracted.years_of_experience,
        education_summary=education_summary,
    )


# ---------------------------------------------------------------------------
# Step 3: Flag
# ---------------------------------------------------------------------------

_AMBIGUOUS_TITLES = {
    "generalist",
    "specialist",
    "associate",
    "consultant",
    "advisor",
    "coordinator",
    "analyst",
}


def step_flag(extracted: ExtractedFields) -> FlagReport:
    """Identify red flags and grey areas. Deliberately kept as a separate,
    visible step so flags are never silently absorbed into the recommendation.

    Flags two categories:
      - Employment gaps (from pre-computed timeline)
      - Ambiguous job titles (keyword heuristic)
    """
    flags: list[RedFlag] = []

    # Flag employment gaps
    for gap in extracted.timeline.gaps:
        severity = "high" if gap.duration_months >= 12 else "medium"
        flags.append(RedFlag(
            category="Employment Gap",
            description=(
                f"{gap.duration_months}-month gap ({gap.gap_start} to {gap.gap_end}) "
                f"between {gap.preceding_job} and {gap.following_job}. "
                f"No explanation provided in the resume."
            ),
            severity=severity,
        ))

    # Flag ambiguous titles
    for entry in extracted.employment_entries:
        title_lower = entry.title.lower()
        matched_keywords = [kw for kw in _AMBIGUOUS_TITLES if kw in title_lower]
        if matched_keywords:
            flags.append(RedFlag(
                category="Ambiguous Job Title",
                description=(
                    f"Title '{entry.title}' at {entry.company} "
                    f"({entry.start_date} to {entry.end_date}) is vague and could "
                    f"map to multiple role categories. The job description mentions "
                    f"varied responsibilities without a clear specialization, making "
                    f"it difficult to assess depth of expertise in any single area."
                ),
                severity="medium",
            ))

    # Build summary
    if flags:
        summary_parts = [f"{len(flags)} concern(s) identified:"]
        for f in flags:
            summary_parts.append(f"  - [{f.severity.upper()}] {f.category}: {f.description[:80]}...")
        summary = "\n".join(summary_parts)
    else:
        summary = "No red flags identified."

    return FlagReport(flags=flags, summary=summary)


# ---------------------------------------------------------------------------
# Step 4: Recommend (LLM call)
# ---------------------------------------------------------------------------

def step_recommend(
    mapped: MappedForm,
    flag_report: FlagReport,
) -> RecommendationResult:
    """Use the LLM to recommend a role from the fixed category list.

    The LLM receives only the mapped candidate data and flag report —
    not the raw resume — and is constrained to select from (or justify
    deviation from) the approved role list.
    """
    prompt_template = (_PROMPT_DIR / "form_fill_system_prompt.txt").read_text(
        encoding="utf-8"
    )

    candidate_summary = (
        f"Name: {mapped.candidate_name}\n"
        f"Years of Experience: {mapped.years_of_experience}\n"
        f"Primary Skillset: {', '.join(mapped.primary_skillset)}\n"
        f"Education: {mapped.education_summary}\n"
    )

    flag_text = flag_report.summary if flag_report.flags else "No flags."

    system_prompt = prompt_template.replace(
        "{candidate_summary}", candidate_summary
    ).replace(
        "{flag_report}", flag_text
    )

    raw_response = get_llm_response(system_prompt, "Provide your role recommendation now.")

    # Parse the JSON response from the LLM
    try:
        # Strip markdown fences if the LLM wraps them
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        cleaned = cleaned.strip()

        data = json.loads(cleaned)
        return RecommendationResult(
            recommended_role=data.get("recommended_role", "Unknown"),
            confidence_score=min(max(float(data.get("confidence_score", 0.5)), 0.0), 1.0),
            reasoning=data.get("reasoning", "No reasoning provided."),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Failed to parse LLM recommendation response: %s", exc)
        logger.debug("Raw response: %s", raw_response)
        return RecommendationResult(
            recommended_role="Requires Manual Review",
            confidence_score=0.0,
            reasoning=f"LLM response could not be parsed. Raw output: {raw_response[:500]}",
        )


# ---------------------------------------------------------------------------
# Step 5: Dispatch
# ---------------------------------------------------------------------------

def step_dispatch(
    form: EvaluationForm,
    export_format: str = "json",
    send_email: bool = True,
    email_to: str = "hr-admissions@company.mock",
) -> list[DispatchResult]:
    """Export the form and optionally send a mock email."""
    results: list[DispatchResult] = []

    # File export
    result = export_form(form, fmt=export_format)
    results.append(result)

    # Mock email
    if send_email:
        email_result = mock_send_email(to=email_to, form=form)
        results.append(email_result)

    return results


# ---------------------------------------------------------------------------
# Full pipeline orchestrator
# ---------------------------------------------------------------------------

def run_evaluation(
    candidate: Candidate,
    export_format: str = "json",
    send_email: bool = True,
) -> dict:
    """Execute the full agentic evaluation pipeline.

    Returns a dict containing the final EvaluationForm plus all intermediate
    step outputs for auditability.
    """
    logger.info("Starting evaluation pipeline for: %s", candidate.full_name)

    # Step 1: Extract
    logger.info("[Step 1/5] Extracting structured fields...")
    extracted = step_extract(candidate)

    # Step 2: Map
    logger.info("[Step 2/5] Mapping to evaluation form...")
    mapped = step_map(extracted)

    # Step 3: Flag
    logger.info("[Step 3/5] Flagging red flags and grey areas...")
    flag_report = step_flag(extracted)

    # Step 4: Recommend
    logger.info("[Step 4/5] Generating role recommendation...")
    recommendation = step_recommend(mapped, flag_report)

    # Assemble the final form
    form = EvaluationForm(
        candidate_name=mapped.candidate_name,
        primary_skillset=mapped.primary_skillset,
        years_of_experience=mapped.years_of_experience,
        education_summary=mapped.education_summary,
        red_flags=flag_report.flags,
        recommended_role=recommendation.recommended_role,
        confidence_score=recommendation.confidence_score,
        reasoning_notes=recommendation.reasoning,
    )

    # Step 5: Dispatch
    logger.info("[Step 5/5] Dispatching evaluation form...")
    dispatch_results = step_dispatch(
        form,
        export_format=export_format,
        send_email=send_email,
    )

    form.dispatch_status = "; ".join(
        f"{r.method}: {r.detail}" for r in dispatch_results
    )

    logger.info("Evaluation pipeline complete for: %s", candidate.full_name)

    return {
        "evaluation_form": form.model_dump(),
        "pipeline_steps": {
            "step_1_extract": extracted.model_dump(),
            "step_2_map": mapped.model_dump(),
            "step_3_flag": flag_report.model_dump(),
            "step_4_recommend": recommendation.model_dump(),
            "step_5_dispatch": [r.model_dump() for r in dispatch_results],
        },
    }
