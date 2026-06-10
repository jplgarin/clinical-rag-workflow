from core.schema import GeneratedReport, GeneratedSection, RetrievedChunk
from core.verifier import ReportVerifier, _content_tokens, _cosine


def chunk(content, source="src.txt"):
    return RetrievedChunk(
        content=content, source=source, relevance_score=0.8, chunk_id=f"{source}::0"
    )


def test_content_tokens_drops_stopwords():
    tokens = _content_tokens("The patient has elevated theta activity")
    assert "the" not in tokens
    assert "patient" not in tokens
    assert "elevated" in tokens and "theta" in tokens


def test_supported_claim_via_overlap():
    section = GeneratedSection(
        title="S",
        content="Elevated theta beta ratio is associated with attentional dysregulation.",
    )
    src = [chunk("Elevated theta beta ratio is associated with attentional dysregulation in children.")]
    results = ReportVerifier().verify_section(section, src)
    assert len(results) == 1
    assert results[0].is_supported
    assert results[0].supporting_source == "src.txt"


def test_unsupported_claim_flagged():
    section = GeneratedSection(
        title="S",
        content="The patient requires immediate surgical intervention tomorrow morning.",
    )
    src = [chunk("Theta beta ratio reflects cortical arousal patterns.")]
    results = ReportVerifier().verify_section(section, src)
    assert results[0].is_supported is False
    assert results[0].supporting_source is None


def test_short_claims_skipped():
    section = GeneratedSection(title="S", content="Yes. No.")
    results = ReportVerifier().verify_section(section, [chunk("anything here")])
    assert results == []


def test_verify_report_adds_warnings_and_adjusts_confidence():
    section = GeneratedSection(
        title="Summary",
        content="The patient needs an unrelated fabricated cardiac procedure scheduled.",
        supporting_chunks=[chunk("Theta beta ratio reflects cortical arousal.")],
        confidence_score=0.9,
    )
    report = GeneratedReport(sections=[section], overall_confidence=0.9)
    verified = ReportVerifier().verify_report(report)
    assert verified.warnings
    assert "Summary" in verified.warnings[0]
    # confidence pulled down because nothing was grounded
    assert verified.sections[0].confidence_score < 0.9


def test_semantic_fallback_supports_paraphrase():
    # Embedder maps exact strings to orthonormal-ish vectors; paraphrase shares one.
    space = {
        "claim": [1.0, 1.0, 0.0],
        "evidence": [1.0, 1.0, 0.0],
    }

    def embedder(texts):
        return [space.get("claim" if i == 0 else "evidence", [0.0, 0.0, 1.0]) for i, _ in enumerate(texts)]

    section = GeneratedSection(
        title="S",
        content="Completely different lexical wording with no shared content words zzz.",
    )
    src = [chunk("Totally distinct vocabulary appears in this evidence passage qqq.")]
    verifier = ReportVerifier(embedder=embedder)
    results = verifier.verify_section(section, src)
    assert results[0].is_supported


def test_cosine_edge_cases():
    assert _cosine([0, 0], [1, 1]) == 0.0
    assert _cosine([1, 0], [1, 0]) == 1.0
