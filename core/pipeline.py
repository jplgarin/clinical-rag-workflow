"""End-to-end orchestration: raw payload in, verified report out.

The pipeline is intentionally thin. It wires the adapter, retriever,
generator, and verifier together and logs each hand-off. All the domain
knowledge lives in the adapter; all the heavy lifting lives in the components.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from core.generator import ReportGenerator
from core.retriever import Retriever
from core.schema import GeneratedReport, PatientContext, ReportContext
from core.verifier import ReportVerifier

if TYPE_CHECKING:
    from adapters.base import BaseAdapter

logger = logging.getLogger(__name__)


class ClinicalReportPipeline:
    """Compose the four stages into a single ``run`` call."""

    def __init__(
        self,
        retriever: Retriever,
        generator: ReportGenerator,
        verifier: ReportVerifier,
        adapter: "BaseAdapter",
    ):
        self.retriever = retriever
        self.generator = generator
        self.verifier = verifier
        self.adapter = adapter

    def run(self, raw_findings: dict, patient_context: dict) -> GeneratedReport:
        """Run the full pipeline.

        Args:
            raw_findings: The domain-specific payload, as the adapter expects.
            patient_context: Demographic context. Keys not understood by
                :class:`~core.schema.PatientContext` land in
                ``additional_context``.

        Returns:
            A verified :class:`~core.schema.GeneratedReport`.
        """
        started = time.perf_counter()
        domain = self.adapter.get_domain()
        logger.info("pipeline start | domain=%s", domain)

        findings = self.adapter.format_findings(raw_findings)
        logger.info("step 1 format_findings | %d findings", len(findings.findings))

        patient = _build_patient_context(patient_context)
        sections = self.adapter.get_report_sections()
        chunks = self.retriever.retrieve_for_sections(sections)
        retrieved_total = sum(len(v) for v in chunks.values())
        logger.info(
            "step 2 retrieve | %d sections, %d chunks", len(sections), retrieved_total
        )

        context = ReportContext(
            patient=patient, findings=findings, report_sections=sections
        )
        report = self.generator.generate_report(
            context, chunks, self.adapter.get_prompt_templates()
        )
        logger.info("step 3 generate | %d sections written", len(report.sections))

        report = self.verifier.verify_report(report)
        logger.info("step 4 verify | %d warnings", len(report.warnings))

        elapsed = time.perf_counter() - started
        report.generation_metadata.update(
            {
                "elapsed_seconds": round(elapsed, 3),
                "report_metadata": self.adapter.get_report_metadata(),
            }
        )
        logger.info("pipeline done | domain=%s | %.2fs", domain, elapsed)
        return report


def _build_patient_context(raw: dict) -> PatientContext:
    known = {"age", "sex", "age_group", "additional_context"}
    extra = {k: v for k, v in raw.items() if k not in known}
    payload = {k: v for k, v in raw.items() if k in known}
    if extra:
        payload.setdefault("additional_context", {}).update(extra)
    return PatientContext(**payload)
