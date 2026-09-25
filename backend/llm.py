"""Thin LLM client abstraction.

Supports Google Gemini (primary), OpenAI (alternative), and a built-in mock
for running the full application without any API key. The mock is always
available and produces structured, deterministic responses suitable for
demos and testing.

Configuration via environment variables:
    GOOGLE_API_KEY  -> uses Gemini 1.5 Flash
    OPENAI_API_KEY  -> uses GPT-4o-mini
    (neither set)   -> falls back to mock mode automatically
    LLM_PROVIDER    -> force a provider: "gemini" | "openai" | "mock"
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger("resumeiq.llm")

# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

def _can_import(module_name: str) -> bool:
    """Check if a module is importable without side effects."""
    import importlib
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def _active_provider() -> str:
    """Determine which LLM backend to use.

    Checks both that the API key is set AND the library is importable,
    falling back to mock mode if either condition is missing.
    """
    forced = os.environ.get("LLM_PROVIDER", "").lower()
    if forced == "mock":
        return "mock"
    if forced == "gemini" and _can_import("google.generativeai"):
        return "gemini"
    if forced == "openai" and _can_import("openai"):
        return "openai"
    if forced:
        logger.warning("Forced provider '%s' unavailable, falling back to mock.", forced)
        return "mock"

    if os.environ.get("GOOGLE_API_KEY") and _can_import("google.generativeai"):
        return "gemini"
    if os.environ.get("OPENAI_API_KEY") and _can_import("openai"):
        return "openai"
    return "mock"


def get_llm_response(system_prompt: str, user_prompt: str) -> str:
    """Send a prompt to the active LLM and return the text response.

    Parameters
    ----------
    system_prompt : str
        Instructions and context for the model.
    user_prompt : str
        The user-facing query or task.

    Returns
    -------
    str
        The model's text response.
    """
    provider = _active_provider()
    logger.info("LLM provider: %s", provider)

    if provider == "gemini":
        return _gemini(system_prompt, user_prompt)
    elif provider == "openai":
        return _openai(system_prompt, user_prompt)
    else:
        return _mock(system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

def _gemini(system_prompt: str, user_prompt: str) -> str:
    try:
        import google.generativeai as genai

        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system_prompt,
        )
        response = model.generate_content(user_prompt)
        return response.text
    except Exception as exc:
        logger.warning("Gemini API call failed (%s), falling back to mock.", exc)
        return _mock(system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

def _openai(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Mock LLM — keyword-driven, deterministic, zero external dependencies
# ---------------------------------------------------------------------------

def _mock(system_prompt: str, user_prompt: str) -> str:
    """Return realistic template responses based on keyword detection.

    The mock inspects both the system prompt (to identify the task type) and
    the user prompt (to identify the topic) and returns pre-written responses
    that exercise the same code paths as a real LLM would.
    """
    query = user_prompt.lower()

    # --- Role recommendation (returns JSON) ---
    if "recommended_role" in system_prompt.lower() or "approved role categories" in system_prompt.lower():
        return json.dumps({
            "recommended_role": "Platform / Infrastructure Engineer",
            "confidence_score": 0.78,
            "reasoning": (
                "The candidate has 3.4 years of direct platform engineering experience at "
                "ScaleGrid Systems leading Kubernetes and EKS infrastructure. Combined with "
                "AWS and GCP certifications, Terraform/Helm proficiency, and a track record "
                "of cross-team IaC adoption, Platform / Infrastructure Engineer is the "
                "strongest fit. Confidence is moderated by the 23-month employment gap "
                "(2019-04 to 2021-03) and the ambiguous 'Technology Generalist' title at "
                "InnovateLabs Consulting which makes it harder to verify depth of prior "
                "infrastructure work."
            ),
        })

    # --- Q&A: cloud deployment + gaps ---
    if "cloud" in query and ("gap" in query or "employment" in query):
        return (
            "Based on the resume data:\n\n"
            "**Cloud deployment experience:** Yes. The candidate lists AWS (EC2, S3, Lambda, "
            "EKS, CloudFormation) and Google Cloud Platform (GKE, BigQuery, Cloud Run) in "
            "their skills. At ScaleGrid Systems (2021-03 to 2024-08), they designed and "
            "implemented a zero-downtime migration from EC2-based deployments to EKS and "
            "drove Infrastructure-as-Code adoption using Terraform and Helm across 12 "
            "engineering teams. They also hold an AWS Certified Solutions Architect - "
            "Associate credential (2021-06) and a Certified Kubernetes Administrator "
            "certification (2022-01).\n\n"
            "**Employment gaps:** The pre-computed timeline identifies one significant gap "
            "of 23 months, from 2019-04 (end of InnovateLabs Consulting) to 2021-03 (start "
            "of ScaleGrid Systems). The resume does not state a reason for this gap. During "
            "this period, the candidate completed a Machine Learning Specialization from "
            "Stanford University (Online via Coursera) between 2020-03 and 2020-09, which "
            "partially overlaps the gap but does not fully explain the remaining months. "
            "The reason for the gap is not stated in the resume and cannot be determined "
            "from the provided data."
        )

    # --- Q&A: backend fit ---
    if "backend" in query or "fit" in query:
        return (
            "Based on the resume data, the candidate has relevant backend engineering "
            "experience:\n\n"
            "- At Nexus Technologies (2015-07 to 2017-05), they developed backend services "
            "using Java and Spring Boot, implemented RESTful APIs for 50+ enterprise clients, "
            "and contributed to PostgreSQL query optimization (40% response time reduction).\n"
            "- Their listed skills include Python, Java, Go, PostgreSQL, Redis, Apache Kafka, "
            "FastAPI, and gRPC — all commonly used in backend engineering roles.\n"
            "- At ScaleGrid Systems (2021-03 to 2024-08), while the title was 'Senior "
            "Platform Engineer,' the work included building internal developer tooling and "
            "deployment pipelines, which overlaps with backend engineering scope.\n\n"
            "However, the candidate's most recent and senior experience (3.4 years) is "
            "focused on platform/infrastructure rather than application-level backend "
            "development. Whether this constitutes a strong fit depends on how the hiring "
            "team defines the 'backend engineering' role — pure application development "
            "vs. platform-adjacent backend work."
        )

    # --- Q&A: absent information (refusal) ---
    if any(kw in query for kw in ["salary", "compensation", "expect", "motivation", "why did", "reason for leaving", "hobbies", "interests", "management style", "references"]):
        return (
            "This information is not stated in the resume and cannot be determined from the "
            "provided data. The candidate's resume includes employment history, skills, "
            "education, certifications, and project details, but does not address this topic."
        )

    # --- Q&A: skills/technology questions ---
    if any(kw in query for kw in ["skill", "python", "java", "kubernetes", "docker", "aws", "react", "terraform"]):
        return (
            "Based on the candidate's listed skills and employment history:\n\n"
            "The candidate lists the following relevant technologies: Python, Java, Go, "
            "TypeScript, AWS (EC2, S3, Lambda, EKS, CloudFormation), Google Cloud Platform "
            "(GKE, BigQuery, Cloud Run), Docker, Kubernetes, Terraform, Helm, PostgreSQL, "
            "Redis, Apache Kafka, React, FastAPI, and gRPC.\n\n"
            "Their employment history demonstrates practical application of these skills "
            "across three roles spanning approximately 7.3 years of total experience. "
            "The most recent role at ScaleGrid Systems (2021-03 to 2024-08) emphasizes "
            "Kubernetes, AWS EKS, Terraform, and Helm in a production environment serving "
            "2M+ daily active users."
        )

    # --- Q&A: experience / years ---
    if "experience" in query or "years" in query:
        return (
            "According to the pre-computed employment timeline, the candidate has "
            "approximately 7.3 years of total professional experience across three roles:\n\n"
            "1. Nexus Technologies — Software Engineer (2015-07 to 2017-05): ~1.8 years\n"
            "2. InnovateLabs Consulting — Technology Generalist (2017-06 to 2019-04): ~1.8 years\n"
            "3. ScaleGrid Systems — Senior Platform Engineer (2021-03 to 2024-08): ~3.4 years\n\n"
            "Note: there is a 23-month gap between roles 2 and 3 (2019-04 to 2021-03) "
            "which is not counted toward professional experience."
        )

    # --- Generic fallback ---
    return (
        "Based on the available resume data, here is what can be determined:\n\n"
        "The candidate, Arjun Mehta, is a platform engineer based in San Francisco, CA "
        "with approximately 7.3 years of experience across three positions. Their strongest "
        "technical areas are cloud infrastructure (AWS, GCP), container orchestration "
        "(Kubernetes, Docker), and backend development (Python, Java, Go). Notable aspects "
        "include a 23-month unexplained employment gap (2019-04 to 2021-03) and an "
        "ambiguous 'Technology Generalist' title at InnovateLabs Consulting. For more "
        "specific information, please ask a targeted question about the candidate's "
        "skills, experience, education, or employment history."
    )
