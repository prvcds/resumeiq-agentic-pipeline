"""Pydantic schemas for ResumeIQ: candidate data, evaluation forms, and API contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


# ---------------------------------------------------------------------------
# Resume / Candidate schemas
# ---------------------------------------------------------------------------

class EducationEntry(BaseModel):
    institution: str
    degree: str
    field: str
    start_date: str = Field(description="YYYY-MM format")
    end_date: str = Field(description="YYYY-MM format")
    gpa: Optional[str] = None
    notes: Optional[str] = None


class EmploymentEntry(BaseModel):
    company: str
    title: str
    start_date: str = Field(description="YYYY-MM format")
    end_date: str = Field(description="YYYY-MM format")
    location: str
    description: str


class Certification(BaseModel):
    name: str
    issuer: str
    date: str = Field(description="YYYY-MM format")


class Project(BaseModel):
    name: str
    description: str
    technologies: list[str]


class Candidate(BaseModel):
    """Normalized representation of a candidate's resume/biodata.

    Every downstream module (Q&A, evaluation, dispatch) works against this
    schema rather than raw dicts, ensuring a single source of truth.
    """
    full_name: str
    email: str
    phone: str
    location: str
    linkedin: Optional[str] = None
    summary: str
    education: list[EducationEntry]
    skills: list[str]
    employment: list[EmploymentEntry]
    certifications: list[Certification] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Computed timeline structures
# ---------------------------------------------------------------------------

class EmploymentGap(BaseModel):
    """A gap between two consecutive employment entries, computed via date
    arithmetic (not LLM inference)."""
    gap_start: str
    gap_end: str
    duration_months: int
    preceding_job: str = Field(description="Company — Title of the earlier role")
    following_job: str = Field(description="Company — Title of the later role")


class EmploymentTimeline(BaseModel):
    """Pre-computed employment timeline injected into LLM prompts so the model
    never needs to perform its own date calculations."""
    entries_chronological: list[EmploymentEntry]
    total_years_experience: float = Field(description="Sum of all employment durations in years, rounded to 1 decimal")
    gaps: list[EmploymentGap]


# ---------------------------------------------------------------------------
# Q&A API contracts
# ---------------------------------------------------------------------------

class QARequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class QAResponse(BaseModel):
    question: str
    answer: str
    grounded: bool = Field(
        default=True,
        description="True when the answer is derived solely from candidate data"
    )


# ---------------------------------------------------------------------------
# Evaluation form and red flags
# ---------------------------------------------------------------------------

class RedFlag(BaseModel):
    category: str = Field(description="e.g. Employment Gap, Ambiguous Title, Missing Credential")
    description: str
    severity: str = Field(description="low | medium | high")


class EvaluationForm(BaseModel):
    """Corporate Hiring Evaluation Form — the output of the agentic pipeline."""
    candidate_name: str
    primary_skillset: list[str]
    years_of_experience: float
    education_summary: str
    red_flags: list[RedFlag]
    recommended_role: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning_notes: str
    dispatch_status: Optional[str] = None


# ---------------------------------------------------------------------------
# Agentic pipeline step outputs (for auditability)
# ---------------------------------------------------------------------------

class ExtractedFields(BaseModel):
    """Raw extracted fields from the Candidate object — Step 1 output."""
    candidate_name: str
    skills: list[str]
    years_of_experience: float
    education_entries: list[EducationEntry]
    employment_entries: list[EmploymentEntry]
    certifications: list[str]
    timeline: EmploymentTimeline


class MappedForm(BaseModel):
    """Partially-filled evaluation form before flagging/recommendation — Step 2 output."""
    candidate_name: str
    primary_skillset: list[str]
    years_of_experience: float
    education_summary: str


class FlagReport(BaseModel):
    """Dedicated flag output — Step 3. Deliberately kept separate from the
    recommendation step so red flags are visible and auditable."""
    flags: list[RedFlag]
    summary: str


class RecommendationResult(BaseModel):
    """LLM-generated role recommendation — Step 4 output."""
    recommended_role: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class DispatchResult(BaseModel):
    method: str = Field(description="email | file_json | file_pdf")
    destination: str
    success: bool
    detail: str
