"""Report generation backed by Anthropic's Messages API.

The client uses the official ``anthropic`` SDK and talks to Claude directly,
which avoids the escaping and request-shape mismatches of the
OpenAI-compatible shim. Credentials and model are resolved from the
environment: ``ANTHROPIC_API_KEY`` for auth and ``GENERATOR_MODEL`` for the
model id.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import anthropic

from core.schema import (
    ClinicalFindings,
    GeneratedReport,
    GeneratedSection,
    PatientContext,
    ReportContext,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a clinical report writer. You produce precise, measured prose for "
    "a clinician audience. Ground every clinical statement in the supplied "
    "evidence and cite it inline as [n], where n is the evidence index. If the "
    "evidence does not support a claim, say so plainly rather than inventing "
    "detail. Never restate the patient's identity. Output only the section "
    "body, with no heading."
)


class LLMClient:
    """Minimal synchronous client for Anthropic's Messages API."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.model = model
        # Falls back to the SDK's own ANTHROPIC_API_KEY env resolution when
        # api_key is None.
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout)

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            block.text for block in message.content if block.type == "text"
        ).strip()


class ReportGenerator:
    """Turn findings plus retrieved evidence into a structured report."""

    def __init__(
        self,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        client: Optional[LLMClient] = None,
    ):
        self.model = model or os.getenv("GENERATOR_MODEL") or "claude-sonnet-4-6"
        self.max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "2000"))
        self.temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("TEMPERATURE", "0.3"))
        )
        # Injectable so tests can pass a fake and skip the network entirely.
        self.client = client or LLMClient(self.model)

    def generate_section(
        self,
        section_title: str,
        findings: ClinicalFindings,
        patient: PatientContext,
        retrieved_chunks: list[RetrievedChunk],
        prompt_template: str,
    ) -> GeneratedSection:
        """Generate one section.

        Args:
            section_title: Heading the section will carry.
            findings: The structured findings to describe.
            patient: Anonymous demographic context.
            retrieved_chunks: Evidence to ground and cite. May be empty.
            prompt_template: Adapter-supplied template. Supports the
                placeholders ``{section}``, ``{findings}``, ``{patient}`` and
                ``{evidence}``; unknown placeholders are left untouched.

        Returns:
            A populated :class:`GeneratedSection`. Its ``confidence_score`` is a
            coarse proxy: the mean relevance of the evidence it was given.
        """
        evidence_block = _format_evidence(retrieved_chunks)
        user_prompt = _safe_format(
            prompt_template,
            section=section_title,
            findings=_format_findings(findings),
            patient=_format_patient(patient),
            evidence=evidence_block,
        )
        content = self.client.complete(
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        confidence = (
            sum(c.relevance_score for c in retrieved_chunks) / len(retrieved_chunks)
            if retrieved_chunks
            else 0.5
        )
        return GeneratedSection(
            title=section_title,
            content=content,
            supporting_chunks=retrieved_chunks,
            confidence_score=round(confidence, 3),
        )

    def generate_report(
        self,
        context: ReportContext,
        retrieved_chunks: dict[str, list[RetrievedChunk]],
        prompt_templates: dict[str, str],
    ) -> GeneratedReport:
        """Generate every requested section and assemble the report.

        Sections without a matching template fall back to a generic one, so a
        misconfigured adapter degrades gracefully instead of crashing.
        """
        sections: list[GeneratedSection] = []
        for title in context.report_sections:
            template = prompt_templates.get(title, _DEFAULT_TEMPLATE)
            chunks = retrieved_chunks.get(title, [])
            sections.append(
                self.generate_section(
                    title, context.findings, context.patient, chunks, template
                )
            )

        sources = sorted(
            {c.source for chunks in retrieved_chunks.values() for c in chunks}
        )
        overall = (
            sum(s.confidence_score for s in sections) / len(sections)
            if sections
            else 0.0
        )
        return GeneratedReport(
            sections=sections,
            overall_confidence=round(overall, 3),
            sources_used=sources,
            generation_metadata={
                "model": self.model,
                "domain": context.findings.domain,
                "temperature": self.temperature,
                "section_count": len(sections),
            },
        )


_DEFAULT_TEMPLATE = (
    "Write the '{section}' section of a clinical report.\n\n"
    "Patient context:\n{patient}\n\n"
    "Findings:\n{findings}\n\n"
    "Evidence:\n{evidence}\n\n"
    "Write 2-4 short paragraphs. Cite evidence inline as [n]."
)


def _safe_format(template: str, **kwargs: Any) -> str:
    """``str.format`` that leaves unrecognised ``{placeholders}`` alone."""
    for key, value in kwargs.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def _format_findings(findings: ClinicalFindings) -> str:
    lines = []
    for f in findings.findings:
        unit = f" {f.unit}" if f.unit else ""
        ref = f" (ref {f.reference_range})" if f.reference_range else ""
        lines.append(f"- {f.name}: {f.value}{unit}{ref} [{f.status}]")
    return "\n".join(lines) if lines else "(no findings provided)"


def _format_patient(patient: PatientContext) -> str:
    parts = []
    if patient.age is not None:
        parts.append(f"age {patient.age}")
    parts.append(f"sex {patient.sex}")
    if patient.age_group:
        parts.append(f"group {patient.age_group}")
    return ", ".join(parts)


def _format_evidence(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no supporting evidence retrieved)"
    return "\n\n".join(
        f"[{i + 1}] ({c.citation()}) {c.content}" for i, c in enumerate(chunks)
    )
