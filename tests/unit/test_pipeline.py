from pathlib import Path
from unittest.mock import MagicMock

from adapters.base import BaseAdapter
from core.generator import LLMClient, ReportGenerator
from core.pipeline import ClinicalReportPipeline, _build_patient_context
from core.retriever import Retriever, VectorStore
from core.schema import ClinicalFinding, ClinicalFindings, RetrievedChunk
from core.verifier import ReportVerifier


class StubAdapter(BaseAdapter):
    def get_domain(self):
        return "stub"

    def get_knowledge_base_path(self):
        return Path(".")

    def format_findings(self, raw):
        return ClinicalFindings(
            domain="stub",
            findings=[ClinicalFinding(name="m", value=raw.get("m", 1.0))],
        )

    def get_prompt_templates(self):
        return {"Summary": "{evidence}"}

    def get_report_sections(self):
        return ["Summary"]

    def get_report_metadata(self):
        return {"display_name": "Stub"}


def build_pipeline(llm_text="Body text [1]."):
    client = MagicMock(spec=LLMClient)
    client.complete.return_value = llm_text
    generator = ReportGenerator(model="fake", client=client)

    store = MagicMock(spec=VectorStore)
    store.search.return_value = [
        RetrievedChunk(
            content="Body text supporting evidence.",
            source="s.txt",
            relevance_score=0.7,
            chunk_id="s::0",
        )
    ]
    retriever = Retriever([store])
    return ClinicalReportPipeline(retriever, generator, ReportVerifier(), StubAdapter())


def test_build_patient_context_routes_extras():
    ctx = _build_patient_context({"age": 12, "sex": "M", "weird": "value"})
    assert ctx.age == 12
    assert ctx.additional_context["weird"] == "value"


def test_pipeline_run_produces_report():
    pipeline = build_pipeline()
    report = pipeline.run({"m": 2.0}, {"age": 12, "sex": "M"})
    assert [s.title for s in report.sections] == ["Summary"]
    assert report.sources_used == ["s.txt"]
    assert "elapsed_seconds" in report.generation_metadata
    assert report.generation_metadata["report_metadata"]["display_name"] == "Stub"


def test_pipeline_calls_each_stage_in_order():
    pipeline = build_pipeline()
    pipeline.adapter = MagicMock(wraps=StubAdapter())
    pipeline.run({"m": 1.0}, {})
    pipeline.adapter.format_findings.assert_called_once()
    pipeline.adapter.get_report_sections.assert_called()
    pipeline.adapter.get_prompt_templates.assert_called_once()
