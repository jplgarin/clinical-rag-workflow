import pytest
from pydantic import ValidationError

from core.schema import (
    ClinicalFinding,
    ClinicalFindings,
    FindingStatus,
    GeneratedReport,
    GeneratedSection,
    PatientContext,
    RetrievedChunk,
)


def test_finding_defaults():
    f = ClinicalFinding(name="x", value=1.0)
    assert f.status == FindingStatus.NORMAL.value
    assert f.confidence == 1.0
    assert f.unit is None


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        ClinicalFinding(name="x", value=1.0, confidence=1.5)


@pytest.mark.parametrize(
    "age,expected",
    [
        (4, "early_childhood"),
        (10, "child"),
        (15, "adolescent"),
        (30, "adult"),
        (70, "older_adult"),
    ],
)
def test_age_group_derivation(age, expected):
    assert PatientContext(age=age).age_group == expected


def test_age_group_not_overwritten():
    assert PatientContext(age=10, age_group="custom").age_group == "custom"


def test_age_group_none_without_age():
    assert PatientContext().age_group is None


def test_findings_abnormal_and_by_status():
    findings = ClinicalFindings(
        domain="d",
        findings=[
            ClinicalFinding(name="a", value=1, status=FindingStatus.NORMAL),
            ClinicalFinding(name="b", value=2, status=FindingStatus.ABNORMAL),
            ClinicalFinding(name="c", value=3, status=FindingStatus.CRITICAL),
        ],
    )
    assert len(findings.abnormal) == 2
    assert len(findings.by_status(FindingStatus.CRITICAL)) == 1


def test_chunk_citation_prefers_metadata():
    c = RetrievedChunk(
        content="x", source="paper.txt", relevance_score=0.5, chunk_id="p::0"
    )
    assert c.citation() == "paper.txt"
    c.metadata["citation"] = "Smith 2020"
    assert c.citation() == "Smith 2020"


def test_report_section_lookup():
    report = GeneratedReport(
        sections=[GeneratedSection(title="Summary", content="...")]
    )
    assert report.section("Summary") is not None
    assert report.section("Missing") is None
