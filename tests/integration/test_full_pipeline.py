"""Full pipeline against the real Neuraxis adapter with mocked LLM + embeddings.

This exercises the genuine wiring (adapter -> retriever -> generator ->
verifier) end to end. Only the two external dependencies are faked: the
embedding model and the LLM call. Everything else is the production code path.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from core.generator import ReportGenerator
from core.pipeline import ClinicalReportPipeline
from core.retriever import Retriever, VectorStore
from core.verifier import ReportVerifier
from examples.adhd_neuraxis.adapter import NeuraxisAdapter

SAMPLE = Path("examples/adhd_neuraxis/sample_findings.json")


class HashingModel:
    """Cheap deterministic embedder so we avoid downloading a real model."""

    DIM = 32

    def encode(self, texts, show_progress_bar=False):
        vectors = []
        for t in texts:
            v = np.zeros(self.DIM, dtype=np.float32)
            for token in t.lower().split():
                v[hash(token) % self.DIM] += 1.0
            vectors.append(v)
        return np.asarray(vectors, dtype=np.float32)


class CannedLLM:
    """Echoes a plausible, evidence-grounded body for any section."""

    def complete(self, system, user, max_tokens, temperature):
        return (
            "The theta beta ratio was elevated relative to the normative mean [1]. "
            "This pattern is reported in a subgroup of children with ADHD [1]."
        )


@pytest.fixture
def pipeline():
    adapter = NeuraxisAdapter()
    store = VectorStore(name="adhd")
    store._model = HashingModel()
    store.load_documents(adapter.get_knowledge_base_path())
    store.embed_documents()

    retriever = Retriever([store])
    generator = ReportGenerator(model="fake", client=CannedLLM())
    return ClinicalReportPipeline(retriever, generator, ReportVerifier(), adapter)


def test_knowledge_base_loaded(pipeline):
    store = pipeline.retriever.stores[0]
    assert len(store) > 0


def test_full_run_over_sample(pipeline):
    raw = json.loads(SAMPLE.read_text())
    report = pipeline.run(raw, {"age": raw["patient_age"], "sex": raw["patient_sex"]})

    expected = NeuraxisAdapter().get_report_sections()
    assert [s.title for s in report.sections] == expected
    assert all(s.content for s in report.sections)
    assert report.sources_used  # retrieval actually pulled evidence
    assert 0.0 <= report.overall_confidence <= 1.0
    assert report.generation_metadata["domain"] == "adhd_neuraxis"


def test_each_section_has_supporting_chunks(pipeline):
    raw = json.loads(SAMPLE.read_text())
    report = pipeline.run(raw, {"age": 12, "sex": "M"})
    assert any(s.supporting_chunks for s in report.sections)
