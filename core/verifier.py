"""A lightweight hallucination guard.

This is deliberately simple and explainable rather than clever. We break each
section into sentence-level claims and check whether the retrieved evidence
plausibly supports them, using a blend of token overlap and (optionally)
embedding similarity. It will not catch subtle fabrication, but it reliably
flags claims with no grounding at all, which is the failure mode that matters
most for a clinical report.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from pydantic import BaseModel, Field

from core.schema import GeneratedReport, GeneratedSection, RetrievedChunk

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "the a an and or of to in is are was were be this that these those with for "
    "on at by from as it its their his her patient findings report section may "
    "be can could should would which who whom whose than then there here".split()
)

# A claim needs at least this fraction of its content tokens to appear in some
# single evidence chunk before we call it supported on overlap alone.
_OVERLAP_THRESHOLD = 0.4
_SEMANTIC_THRESHOLD = 0.55


class VerificationResult(BaseModel):
    claim: str
    is_supported: bool
    supporting_source: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS}


def _overlap(claim_tokens: set[str], chunk_tokens: set[str]) -> float:
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & chunk_tokens) / len(claim_tokens)


class ReportVerifier:
    """Check generated claims against the evidence used to produce them.

    Args:
        embedder: Optional callable mapping a list of strings to a list of
            vectors (lists of floats). When supplied, semantic similarity is
            used as a fallback for claims that fail the lexical check. When
            omitted, verification is purely lexical, which keeps the dependency
            footprint and runtime low.
    """

    def __init__(self, embedder=None, min_claim_tokens: int = 4):
        self._embedder = embedder
        self.min_claim_tokens = min_claim_tokens

    def verify_section(
        self, section: GeneratedSection, source_chunks: list[RetrievedChunk]
    ) -> list[VerificationResult]:
        results: list[VerificationResult] = []
        chunk_tokens = [(_content_tokens(c.content), c) for c in source_chunks]

        for claim in self._split_claims(section.content):
            claim_tokens = _content_tokens(claim)
            if len(claim_tokens) < self.min_claim_tokens:
                # Too short to judge; treat as non-substantive and skip.
                continue

            best_score = 0.0
            best_source: Optional[str] = None
            for tokens, chunk in chunk_tokens:
                score = _overlap(claim_tokens, tokens)
                if score > best_score:
                    best_score, best_source = score, chunk.source

            supported = best_score >= _OVERLAP_THRESHOLD
            if not supported and self._embedder is not None and source_chunks:
                sem_score, sem_source = self._semantic_check(claim, source_chunks)
                if sem_score >= _SEMANTIC_THRESHOLD:
                    supported = True
                    best_score, best_source = sem_score, sem_source

            results.append(
                VerificationResult(
                    claim=claim,
                    is_supported=supported,
                    supporting_source=best_source if supported else None,
                    confidence=round(min(1.0, best_score), 3),
                )
            )
        return results

    def verify_report(self, report: GeneratedReport) -> GeneratedReport:
        """Verify every section and fold unsupported claims into warnings.

        Returns the same report object, mutated in place, so callers can treat
        verification as one more pipeline stage.
        """
        for section in report.sections:
            results = self.verify_section(section, section.supporting_chunks)
            unsupported = [r for r in results if not r.is_supported]
            for r in unsupported:
                report.warnings.append(
                    f"[{section.title}] unsupported claim: {_truncate(r.claim)}"
                )
            if results:
                supported_ratio = 1.0 - len(unsupported) / len(results)
                # Nudge section confidence toward how much we could ground.
                section.confidence_score = round(
                    (section.confidence_score + supported_ratio) / 2.0, 3
                )

        if report.sections:
            report.overall_confidence = round(
                sum(s.confidence_score for s in report.sections)
                / len(report.sections),
                3,
            )
        return report

    def _split_claims(self, text: str) -> list[str]:
        sentences = _SENTENCE_SPLIT.split(text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _semantic_check(
        self, claim: str, chunks: list[RetrievedChunk]
    ) -> tuple[float, Optional[str]]:
        vectors = self._embedder([claim] + [c.content for c in chunks])
        claim_vec = vectors[0]
        best_score, best_source = 0.0, None
        for vec, chunk in zip(vectors[1:], chunks):
            score = _cosine(claim_vec, vec)
            if score > best_score:
                best_score, best_source = score, chunk.source
        return best_score, best_source


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _truncate(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
