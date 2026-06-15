from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.generator import LLMClient, ReportGenerator, _safe_format
from core.schema import (
    ClinicalFinding,
    ClinicalFindings,
    PatientContext,
    ReportContext,
    RetrievedChunk,
)


@pytest.fixture
def findings():
    return ClinicalFindings(
        domain="demo",
        findings=[
            ClinicalFinding(name="TBR_Fz", value=3.2, unit="ratio", status="abnormal")
        ],
    )


@pytest.fixture
def chunks():
    return [
        RetrievedChunk(
            content="Elevated theta beta ratio is associated with ADHD.",
            source="paper1.txt",
            relevance_score=0.9,
            chunk_id="p1::0",
        )
    ]


def fake_client(text="Generated body [1]."):
    client = MagicMock(spec=LLMClient)
    client.complete.return_value = text
    return client


def test_safe_format_leaves_unknown_placeholders():
    out = _safe_format("{known} and {unknown}", known="X")
    assert out == "X and {unknown}"


def test_generate_section_uses_template(findings, chunks):
    client = fake_client()
    gen = ReportGenerator(model="fake", client=client)
    section = gen.generate_section(
        "Summary", findings, PatientContext(age=12), chunks, "{section}: {evidence}"
    )
    assert section.title == "Summary"
    assert section.content == "Generated body [1]."
    assert section.supporting_chunks == chunks
    # confidence proxies mean chunk relevance
    assert section.confidence_score == pytest.approx(0.9)

    user_prompt = client.complete.call_args.kwargs["user"]
    assert "Summary" in user_prompt
    assert "paper1" in user_prompt


def test_generate_section_without_chunks(findings):
    gen = ReportGenerator(model="fake", client=fake_client())
    section = gen.generate_section(
        "S", findings, PatientContext(), [], "{evidence}"
    )
    assert section.confidence_score == 0.5


def test_generate_report_assembles_sections(findings, chunks):
    gen = ReportGenerator(model="fake", client=fake_client())
    context = ReportContext(
        patient=PatientContext(age=12),
        findings=findings,
        report_sections=["Summary", "Recommendations"],
    )
    report = gen.generate_report(
        context,
        {"Summary": chunks, "Recommendations": []},
        {"Summary": "{evidence}"},
    )
    assert [s.title for s in report.sections] == ["Summary", "Recommendations"]
    assert report.sources_used == ["paper1.txt"]
    assert report.generation_metadata["section_count"] == 2
    assert 0 <= report.overall_confidence <= 1


def test_generate_report_falls_back_to_default_template(findings):
    gen = ReportGenerator(model="fake", client=fake_client())
    context = ReportContext(
        patient=PatientContext(), findings=findings, report_sections=["Unmapped"]
    )
    report = gen.generate_report(context, {}, {})
    assert len(report.sections) == 1


def test_llm_client_calls_anthropic_messages():
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="  hello  ")]
        )

    client = LLMClient(model="m", api_key="secret")
    client.client.messages.create = fake_create

    out = client.complete("sys", "user", max_tokens=10, temperature=0.2)

    assert out == "hello"
    assert captured["model"] == "m"
    assert captured["system"] == "sys"
    assert captured["max_tokens"] == 10
    assert captured["temperature"] == 0.2
    assert captured["messages"] == [{"role": "user", "content": "user"}]
