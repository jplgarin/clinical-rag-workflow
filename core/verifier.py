"""A lightweight hallucination guard.

This is deliberately simple and explainable rather than clever. We break each
section into sentence-level claims and check whether the retrieved evidence
plausibly supports them.

Grounding is judged primarily by *semantic* similarity: the generator
paraphrases the evidence rather than copying it verbatim, so exact token
overlap misses legitimately grounded prose and floods the report with false
"unsupported claim" warnings. When an embedder is supplied we embed each claim
and each evidence chunk and compare with cosine similarity; lexical overlap is
kept only as a cheap shortcut that can confirm support, never as the sole
reason to flag.

A claim is only flagged as unsupported when there is a concrete reason to doubt
it: it makes a specific numerical assertion absent from the evidence, or its
similarity to every chunk is low. Borderline claims are left unflagged so the
warnings section stays meaningful.
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
_CITATION = re.compile(r"\[\d+\]")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_STOPWORDS = frozenset(
    "the a an and or of to in is are was were be this that these those with for "
    "on at by from as it its their his her patient findings report section may "
    "be can could should would which who whom whose than then there here".split()
)

# A claim needs at least this fraction of its content tokens to appear in some
# single evidence chunk before overlap alone confirms support.
_OVERLAP_THRESHOLD = 0.4
# Cosine similarity at or above this confirms a claim is grounded.
_SEMANTIC_SUPPORT_THRESHOLD = 0.35
# Below this the claim is unrelated to every chunk and gets flagged.
_SEMANTIC_UNSUPPORTED_THRESHOLD = 0.25


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


def _numbers(text: str) -> set[float]:
    """Extract numeric values, ignoring inline citation markers like ``[2]``."""
    cleaned = _CITATION.sub(" ", text)
    out: set[float] = set()
    for token in _NUMBER.findall(cleaned):
        try:
            out.add(float(token))
        except ValueError:
            continue
    return out


def _has_ungrounded_number(claim: str, chunks: list[RetrievedChunk]) -> bool:
    """True when the claim asserts a number that appears in no evidence chunk.

    A "specific numerical assertion not present in the knowledge base" is the
    clearest sign of fabrication, so it is treated as a reason to flag even
    when the surrounding wording looks plausible.
    """
    claim_nums = _numbers(claim)
    if not claim_nums:
        return False
    evidence_nums: set[float] = set()
    for chunk in chunks:
        evidence_nums |= _numbers(chunk.content)
    return any(
        all(abs(n - m) > 1e-6 for m in evidence_nums) for n in claim_nums
    )


def embedder_from_model(model):
    """Adapt a sentence-transformers-style model to the embedder interface.

    ``model`` only needs an ``encode(list[str]) -> sequence`` method, which the
    ``VectorStore`` model already provides. Reusing the store's loaded model
    keeps the verifier semantic without pulling a second copy into memory.
    """

    def embed(texts):
        return model.encode(list(texts), show_progress_bar=False)

    return embed


def _decide(
    best_overlap: float,
    overlap_source: Optional[str],
    sem_score: Optional[float],
    sem_source: Optional[str],
    numeric_bad: bool,
) -> tuple[bool, Optional[str], float]:
    """Decide whether a claim is supported and return its grounding source.

    A claim is flagged unsupported only when there is a concrete reason: a
    numerical assertion missing from the evidence, or low similarity to every
    chunk. Everything else is left supported so paraphrased grounding does not
    raise false warnings. High lexical overlap can confirm support; it never
    triggers a flag on its own.
    """
    # Report the stronger of the two grounding signals.
    if sem_score is not None and sem_score >= best_overlap:
        score, source = sem_score, sem_source
    else:
        score, source = best_overlap, overlap_source

    lexical_ok = best_overlap >= _OVERLAP_THRESHOLD

    if sem_score is not None:
        if numeric_bad and sem_score < _SEMANTIC_SUPPORT_THRESHOLD:
            supported = False
        elif lexical_ok or sem_score >= _SEMANTIC_SUPPORT_THRESHOLD:
            supported = True
        elif sem_score < _SEMANTIC_UNSUPPORTED_THRESHOLD:
            supported = False
        else:
            # Borderline (0.25–0.35) with no numeric problem: don't flag.
            supported = True
    else:
        # No embedder available: lexical-only fallback.
        supported = lexical_ok and not numeric_bad

    return supported, (source if supported else None), score


class ReportVerifier:
    """Check generated claims against the evidence used to produce them.

    Args:
        embedder: Optional callable mapping a list of strings to a list of
            vectors. When supplied, cosine similarity between each claim and the
            evidence is the primary grounding signal, so paraphrased-but-faithful
            prose is not flagged. When omitted, verification falls back to pure
            token overlap, which keeps tests and lightweight callers offline.
            Use :func:`embedder_from_model` to wrap the retriever's model.
    """

    def __init__(self, embedder=None, min_claim_tokens: int = 4):
        self._embedder = embedder
        self.min_claim_tokens = min_claim_tokens

    def verify_section(
        self, section: GeneratedSection, source_chunks: list[RetrievedChunk]
    ) -> list[VerificationResult]:
        results: list[VerificationResult] = []
        chunk_tokens = [(_content_tokens(c.content), c) for c in source_chunks]
        use_semantic = self._embedder is not None and bool(source_chunks)

        for claim in self._split_claims(section.content):
            claim_tokens = _content_tokens(claim)
            if len(claim_tokens) < self.min_claim_tokens:
                # Too short to judge; treat as non-substantive and skip.
                continue

            best_overlap = 0.0
            overlap_source: Optional[str] = None
            for tokens, chunk in chunk_tokens:
                score = _overlap(claim_tokens, tokens)
                if score > best_overlap:
                    best_overlap, overlap_source = score, chunk.source

            sem_score: Optional[float] = None
            sem_source: Optional[str] = None
            if use_semantic:
                sem_score, sem_source = self._semantic_check(claim, source_chunks)

            numeric_bad = _has_ungrounded_number(claim, source_chunks)
            supported, source, score = _decide(
                best_overlap, overlap_source, sem_score, sem_source, numeric_bad
            )

            results.append(
                VerificationResult(
                    claim=claim,
                    is_supported=supported,
                    supporting_source=source,
                    confidence=round(min(1.0, score), 3),
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
