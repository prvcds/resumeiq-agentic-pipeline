"""Evaluation form dispatch: mock SMTP email and file export (JSON / PDF).

Every function in this module is a mock — no real network calls or production
file-system writes occur. Each mock is explicit, self-contained, and designed
to be replaced by a real implementation without changing the call signature.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fpdf import FPDF

from backend.models import DispatchResult, EvaluationForm

logger = logging.getLogger("resumeiq.dispatch")

_OUTPUT_DIR = Path(__file__).parent.parent / "output"


# ---------------------------------------------------------------------------
# File export
# ---------------------------------------------------------------------------

def export_form(form: EvaluationForm, fmt: str = "json") -> DispatchResult:
    """Write the completed evaluation form to the output directory.

    Parameters
    ----------
    form : EvaluationForm
        The fully populated evaluation form.
    fmt : str
        ``"json"`` or ``"pdf"``.

    Returns
    -------
    DispatchResult
        Metadata about the export operation.
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = form.candidate_name.replace(" ", "_").lower()

    if fmt == "pdf":
        return _export_pdf(form, safe_name, timestamp)
    return _export_json(form, safe_name, timestamp)


def _export_json(form: EvaluationForm, safe_name: str, ts: str) -> DispatchResult:
    filename = f"evaluation_{safe_name}_{ts}.json"
    path = _OUTPUT_DIR / filename
    path.write_text(
        form.model_dump_json(indent=2),
        encoding="utf-8",
    )
    logger.info("Exported JSON evaluation: %s", path)
    return DispatchResult(
        method="file_json",
        destination=str(path),
        success=True,
        detail=f"Evaluation form written to {path}",
    )


def _export_pdf(form: EvaluationForm, safe_name: str, ts: str) -> DispatchResult:
    filename = f"evaluation_{safe_name}_{ts}.pdf"
    path = _OUTPUT_DIR / filename

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Corporate Hiring Evaluation Form", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    # Metadata
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {ts}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Sections
    _pdf_section(pdf, "Candidate Name", form.candidate_name)
    _pdf_section(pdf, "Years of Experience", str(form.years_of_experience))
    _pdf_section(pdf, "Primary Skillset", ", ".join(form.primary_skillset))
    _pdf_section(pdf, "Education Summary", form.education_summary)
    _pdf_section(pdf, "Recommended Role", form.recommended_role)
    _pdf_section(pdf, "Confidence Score", f"{form.confidence_score:.0%}")
    _pdf_section(pdf, "Reasoning Notes", form.reasoning_notes)

    # Red flags
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Red Flags / Concerns", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    if form.red_flags:
        for flag in form.red_flags:
            pdf.multi_cell(
                0, 6,
                f"[{flag.severity.upper()}] {flag.category}: {flag.description}",
            )
            pdf.ln(1)
    else:
        pdf.cell(0, 6, "None identified.", new_x="LMARGIN", new_y="NEXT")

    pdf.output(str(path))
    logger.info("Exported PDF evaluation: %s", path)
    return DispatchResult(
        method="file_pdf",
        destination=str(path),
        success=True,
        detail=f"Evaluation form written to {path}",
    )


def _pdf_section(pdf: FPDF, label: str, value: str) -> None:
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, f"{label}:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, value)
    pdf.ln(2)


# ---------------------------------------------------------------------------
# Mock SMTP dispatch
# ---------------------------------------------------------------------------

def mock_send_email(
    to: str,
    form: EvaluationForm,
    from_addr: str = "resumeiq-system@company.mock",
) -> DispatchResult:
    """Simulate an SMTP email dispatch with realistic log output.

    No real network call is made. The function logs a full SMTP-style
    transaction (EHLO, MAIL FROM, RCPT TO, DATA, 250 OK) for inspection.

    Parameters
    ----------
    to : str
        Recipient email address.
    form : EvaluationForm
        The completed evaluation form (serialized into the email body).
    from_addr : str
        Sender address for the simulated email.
    """
    timestamp = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
    subject = f"Hiring Evaluation: {form.candidate_name} — {form.recommended_role}"

    body_lines = [
        f"Candidate: {form.candidate_name}",
        f"Recommended Role: {form.recommended_role}",
        f"Confidence: {form.confidence_score:.0%}",
        f"Years of Experience: {form.years_of_experience}",
        f"Primary Skills: {', '.join(form.primary_skillset)}",
        f"Education: {form.education_summary}",
        "",
        "Red Flags:",
    ]
    for flag in form.red_flags:
        body_lines.append(f"  [{flag.severity.upper()}] {flag.category}: {flag.description}")
    body_lines.extend(["", "Reasoning:", form.reasoning_notes])
    body = "\n".join(body_lines)

    # Simulate SMTP transaction log
    smtp_log = (
        f"SMTP Transaction Log (MOCK)\n"
        f"{'='*50}\n"
        f">>> EHLO resumeiq.local\n"
        f"<<< 250-smtp.company.mock Hello resumeiq.local\n"
        f"<<< 250-SIZE 35882577\n"
        f"<<< 250 OK\n"
        f">>> MAIL FROM:<{from_addr}>\n"
        f"<<< 250 OK\n"
        f">>> RCPT TO:<{to}>\n"
        f"<<< 250 OK\n"
        f">>> DATA\n"
        f"<<< 354 Start mail input; end with <CRLF>.<CRLF>\n"
        f">>> From: {from_addr}\n"
        f">>> To: {to}\n"
        f">>> Subject: {subject}\n"
        f">>> Date: {timestamp}\n"
        f">>> MIME-Version: 1.0\n"
        f">>> Content-Type: text/plain; charset=utf-8\n"
        f">>>\n"
    )
    for line in body.split("\n"):
        smtp_log += f">>> {line}\n"
    smtp_log += (
        f">>> .\n"
        f"<<< 250 OK id=mock-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}\n"
        f">>> QUIT\n"
        f"<<< 221 smtp.company.mock closing connection\n"
        f"{'='*50}"
    )

    logger.info("Mock SMTP dispatch:\n%s", smtp_log)
    # Also print to stdout for visibility in demo runs
    print(smtp_log)

    return DispatchResult(
        method="email",
        destination=to,
        success=True,
        detail=f"Mock email sent to {to} with subject: {subject}",
    )
