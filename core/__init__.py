"""Domain-agnostic clinical report pipeline core."""

from core.generator import LLMClient, ReportGenerator
from core.pipeline import ClinicalReportPipeline
from core.retriever import Retriever, VectorStore
from core.schema import (
    ClinicalFinding,
    ClinicalFindings,
    FindingStatus,
    GeneratedReport,
    GeneratedSection,
    PatientContext,
    ReportContext,
    RetrievedChunk,
    Sex,
)
from core.verifier import ReportVerifier, VerificationResult

__all__ = [
    "ClinicalFinding",
    "ClinicalFindings",
    "ClinicalReportPipeline",
    "FindingStatus",
    "GeneratedReport",
    "GeneratedSection",
    "LLMClient",
    "PatientContext",
    "ReportContext",
    "ReportGenerator",
    "Retriever",
    "RetrievedChunk",
    "ReportVerifier",
    "Sex",
    "VectorStore",
    "VerificationResult",
]

__version__ = "0.1.0"
