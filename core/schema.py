"""Domain-agnostic data models for the clinical report pipeline.

Nothing in here knows about a specific clinical domain. Adapters are
responsible for mapping their own raw payloads onto these structures, which
keeps the core pipeline reusable across domains.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class FindingStatus(str, Enum):
    """Clinical interpretation of a single measurement."""

    NORMAL = "normal"
    BORDERLINE = "borderline"
    ABNORMAL = "abnormal"
    CRITICAL = "critical"


class Sex(str, Enum):
    MALE = "M"
    FEMALE = "F"
    OTHER = "O"
    UNKNOWN = "U"


class ClinicalFinding(BaseModel):
    """A single biomarker, measurement, or derived feature.

    Attributes:
        name: Short identifier for the measurement (e.g. ``"TBR_Fz"``).
        value: Measured value. Kept as ``float`` because the pipeline only
            ever reasons about numeric findings.
        unit: Unit of measure, or ``None`` when the value is dimensionless.
        reference_range: Human-readable normal range, e.g. ``"1.8 - 2.4"``.
        status: Clinical interpretation of the value.
        confidence: Model or instrument confidence in ``[0, 1]``.
        metadata: Free-form extras an adapter wants to carry through (SHAP
            values, channel names, z-scores, and so on).
    """

    model_config = ConfigDict(use_enum_values=True)

    name: str
    value: float
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    status: FindingStatus = FindingStatus.NORMAL
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClinicalFindings(BaseModel):
    """A bundle of findings produced by an upstream model or instrument.

    ``summary_stats`` is intentionally untyped beyond ``dict`` so each domain
    can attach whatever aggregate it finds useful without us guessing the
    shape ahead of time.
    """

    domain: str
    findings: list[ClinicalFinding] = Field(default_factory=list)
    summary_stats: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utcnow)
    source_model: Optional[str] = None
    version: str = "1.0"

    @property
    def abnormal(self) -> list[ClinicalFinding]:
        """Findings that are anything other than ``normal``."""
        return [f for f in self.findings if f.status != FindingStatus.NORMAL.value]

    def by_status(self, status: FindingStatus) -> list[ClinicalFinding]:
        return [f for f in self.findings if f.status == status.value]


class PatientContext(BaseModel):
    """Anonymous demographic context.

    No direct identifiers live here by design. ``age_group`` is derived on
    construction when it is not supplied, so callers can pass just an age.
    """

    age: Optional[int] = Field(default=None, ge=0, le=130)
    sex: Sex = Sex.UNKNOWN
    age_group: Optional[str] = None
    additional_context: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(use_enum_values=True)

    @model_validator(mode="after")
    def _derive_age_group(self) -> "PatientContext":
        if self.age_group is not None or self.age is None:
            return self
        age = self.age
        if age < 6:
            self.age_group = "early_childhood"
        elif age < 13:
            self.age_group = "child"
        elif age < 18:
            self.age_group = "adolescent"
        elif age < 65:
            self.age_group = "adult"
        else:
            self.age_group = "older_adult"
        return self


class ReportContext(BaseModel):
    """Everything the generator needs to write a report."""

    patient: PatientContext
    findings: ClinicalFindings
    report_sections: list[str]
    language: str = "en"
    verbosity_level: str = "standard"  # one of: brief, standard, detailed


class RetrievedChunk(BaseModel):
    """A passage pulled from a knowledge base during retrieval."""

    content: str
    source: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    chunk_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def citation(self) -> str:
        """Short citation label used for inline references."""
        return self.metadata.get("citation", self.source)


class GeneratedSection(BaseModel):
    """One rendered section of the final report."""

    title: str
    content: str
    supporting_chunks: list[RetrievedChunk] = Field(default_factory=list)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)


class GeneratedReport(BaseModel):
    """The complete pipeline output."""

    sections: list[GeneratedSection] = Field(default_factory=list)
    overall_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    sources_used: list[str] = Field(default_factory=list)
    generation_metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    def section(self, title: str) -> Optional[GeneratedSection]:
        return next((s for s in self.sections if s.title == title), None)
