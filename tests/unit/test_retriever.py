import numpy as np

from core.retriever import Retriever, VectorStore, _split_into_chunks


class FakeModel:
    """Deterministic bag-of-words embedder over a fixed vocabulary."""

    VOCAB = ["theta", "beta", "ratio", "glucose", "insulin", "frontal"]

    def encode(self, texts, show_progress_bar=False):
        vectors = []
        for t in texts:
            low = t.lower()
            vectors.append([float(low.count(w)) for w in self.VOCAB])
        return np.asarray(vectors, dtype=np.float32)


def make_store(texts):
    from core.retriever import _Document

    store = VectorStore(name="test")
    store._model = FakeModel()
    for i, text in enumerate(texts):
        store._documents.append(
            _Document(f"doc::{i}", text, f"doc{i}.txt", {"citation": f"doc{i}"})
        )
    store.embed_documents()
    return store


def test_split_into_chunks_packs_paragraphs():
    text = "a" * 500 + "\n\n" + "b" * 500 + "\n\n" + "c" * 100
    chunks = _split_into_chunks(text, target_chars=800)
    assert len(chunks) == 2


def test_split_ignores_blank_paragraphs():
    assert _split_into_chunks("\n\n  \n\n") == []


def test_search_ranks_by_similarity():
    store = make_store(
        [
            "theta beta ratio frontal",
            "glucose insulin levels",
            "theta theta beta",
        ]
    )
    hits = store.search("theta beta ratio", top_k=2)
    assert len(hits) == 2
    assert hits[0].relevance_score >= hits[1].relevance_score
    assert "theta" in hits[0].content


def test_search_empty_store_returns_nothing():
    store = VectorStore(name="empty")
    store._model = FakeModel()
    assert store.search("anything") == []


def test_save_and_load_index(tmp_path):
    store = make_store(["theta beta ratio", "glucose insulin"])
    path = tmp_path / "idx.pkl"
    store.save_index(path)

    restored = VectorStore()
    restored.load_index(path)
    assert len(restored) == 2
    restored._model = FakeModel()
    assert restored.search("theta")[0].content == "theta beta ratio"


def test_retriever_dedup_keeps_higher_score():
    s1 = make_store(["theta beta ratio frontal"])
    s2 = make_store(["theta beta ratio frontal"])
    retriever = Retriever([s1, s2])
    # identical chunk_id across stores collapses to one entry
    hits = retriever.retrieve("theta beta", top_k=5)
    ids = [h.chunk_id for h in hits]
    assert len(ids) == len(set(ids))


def test_retrieve_for_sections_keys():
    store = make_store(["theta beta", "glucose"])
    retriever = Retriever([store])
    out = retriever.retrieve_for_sections(["A", "B"], top_k=1)
    assert set(out) == {"A", "B"}
    assert all(len(v) <= 1 for v in out.values())
