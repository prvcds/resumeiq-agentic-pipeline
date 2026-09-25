"""Tests for ResumeIQ: Q&A grounding, gap detection, and agentic pipeline.

Run with: pytest tests/ -v
"""

from __future__ import annotations

import json

import pytest

from backend.agent_workflow import (
    run_evaluation,
    step_extract,
    step_flag,
    step_map,
    step_recommend,
)
from backend.models import Candidate
from backend.parsing import compute_timeline, load_candidate
from backend.qa_engine import ask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def candidate() -> Candidate:
    return load_candidate()


# ---------------------------------------------------------------------------
# Task 1: Parsing and timeline computation
# ---------------------------------------------------------------------------

class TestParsing:
    def test_load_candidate_returns_typed_object(self, candidate: Candidate):
        assert isinstance(candidate, Candidate)
        assert candidate.full_name == "Arjun Mehta"

    def test_candidate_has_required_fields(self, candidate: Candidate):
        assert len(candidate.employment) >= 3
        assert len(candidate.skills) > 0
        assert len(candidate.education) > 0

    def test_timeline_total_experience(self, candidate: Candidate):
        timeline = compute_timeline(candidate)
        # 3 jobs: ~1.8 + ~1.8 + ~3.4 = ~7.0 years (exact depends on rounding)
        assert 6.5 <= timeline.total_years_experience <= 8.0

    def test_timeline_detects_employment_gap(self, candidate: Candidate):
        """The mock data has a ~23-month gap between InnovateLabs and ScaleGrid."""
        timeline = compute_timeline(candidate)
        assert len(timeline.gaps) >= 1

        big_gap = max(timeline.gaps, key=lambda g: g.duration_months)
        assert big_gap.duration_months >= 20  # ~23 months
        assert "InnovateLabs" in big_gap.preceding_job
        assert "ScaleGrid" in big_gap.following_job

    def test_timeline_entries_are_chronological(self, candidate: Candidate):
        timeline = compute_timeline(candidate)
        dates = [e.start_date for e in timeline.entries_chronological]
        assert dates == sorted(dates)


# ---------------------------------------------------------------------------
# Task 2: Q&A engine
# ---------------------------------------------------------------------------

class TestQAEngine:
    def test_cloud_and_gaps_question(self, candidate: Candidate):
        """Query type 1: cloud deployments + employment gaps."""
        response = ask(
            candidate,
            "Does this candidate have experience with cloud deployments, "
            "and are there any unexplained employment gaps?",
        )
        answer_lower = response.answer.lower()
        # Should mention cloud-related skills or experience
        assert any(kw in answer_lower for kw in ["aws", "cloud", "eks", "kubernetes"])
        # Should mention the gap
        assert any(kw in answer_lower for kw in ["gap", "23", "month"])
        assert response.grounded is True

    def test_backend_fit_question(self, candidate: Candidate):
        """Query type 2: skills matching for backend engineering."""
        response = ask(
            candidate,
            "Would this candidate be a fit for a backend engineering role?",
        )
        answer_lower = response.answer.lower()
        # Should reference backend-relevant skills
        assert any(kw in answer_lower for kw in ["java", "python", "backend", "api"])
        assert response.grounded is True

    def test_absent_information_refusal(self, candidate: Candidate):
        """Query type 3: information not in the resume triggers refusal."""
        response = ask(
            candidate,
            "What are the candidate's salary expectations?",
        )
        answer_lower = response.answer.lower()
        # Should contain a refusal phrase
        assert any(phrase in answer_lower for phrase in [
            "not stated",
            "cannot be determined",
            "not present",
            "not address",
        ])
        assert response.grounded is True


# ---------------------------------------------------------------------------
# Task 3: Agentic pipeline steps
# ---------------------------------------------------------------------------

class TestAgentPipeline:
    def test_step_extract(self, candidate: Candidate):
        extracted = step_extract(candidate)
        assert extracted.candidate_name == "Arjun Mehta"
        assert extracted.years_of_experience > 0
        assert len(extracted.skills) > 0
        assert len(extracted.timeline.gaps) >= 1

    def test_step_map(self, candidate: Candidate):
        extracted = step_extract(candidate)
        mapped = step_map(extracted)
        assert mapped.candidate_name == "Arjun Mehta"
        assert len(mapped.primary_skillset) > 0
        assert mapped.years_of_experience > 0
        assert len(mapped.education_summary) > 0

    def test_step_flag_detects_gap(self, candidate: Candidate):
        """The flag step must explicitly list the employment gap."""
        extracted = step_extract(candidate)
        flag_report = step_flag(extracted)
        gap_flags = [f for f in flag_report.flags if f.category == "Employment Gap"]
        assert len(gap_flags) >= 1
        assert any("23" in f.description or "month" in f.description for f in gap_flags)

    def test_step_flag_detects_ambiguous_title(self, candidate: Candidate):
        """The flag step must detect the 'Technology Generalist' title."""
        extracted = step_extract(candidate)
        flag_report = step_flag(extracted)
        title_flags = [f for f in flag_report.flags if f.category == "Ambiguous Job Title"]
        assert len(title_flags) >= 1
        assert any("Generalist" in f.description for f in title_flags)

    def test_step_recommend(self, candidate: Candidate):
        extracted = step_extract(candidate)
        mapped = step_map(extracted)
        flag_report = step_flag(extracted)
        rec = step_recommend(mapped, flag_report)
        assert len(rec.recommended_role) > 0
        assert 0.0 <= rec.confidence_score <= 1.0
        assert len(rec.reasoning) > 0

    def test_full_pipeline(self, candidate: Candidate):
        """End-to-end pipeline produces a complete evaluation form."""
        result = run_evaluation(candidate, export_format="json", send_email=False)
        form = result["evaluation_form"]
        assert form["candidate_name"] == "Arjun Mehta"
        assert form["years_of_experience"] > 0
        assert len(form["red_flags"]) >= 2  # gap + ambiguous title
        assert len(form["recommended_role"]) > 0
        assert 0.0 <= form["confidence_score"] <= 1.0

        # Verify pipeline steps are present for auditability
        steps = result["pipeline_steps"]
        assert "step_1_extract" in steps
        assert "step_2_map" in steps
        assert "step_3_flag" in steps
        assert "step_4_recommend" in steps
        assert "step_5_dispatch" in steps
