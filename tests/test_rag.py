"""aicore.rag over the real components (installed from their repositories)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aicore import Chunk, Document, HashEmbedder
from aicore.rag import (
    BPETokenizer,
    Chunker,
    DocumentParser,
    GraphExtractor,
    HybridRetriever,
    RAGPipeline,
    Reranker,
    VectorIndex,
)

DOCS = [
    Document("paris", "The Eiffel Tower is in Paris. It was finished in 1889 for the World's Fair."),
    Document("python", "Python is a programming language created by Guido van Rossum in 1991."),
    Document("everest", "Mount Everest is the highest mountain above sea level, in the Himalayas."),
    Document("bread", "Sourdough bread ferments naturally with wild yeast and needs patience."),
]


def test_tokenizer_roundtrip_and_count() -> None:
    tok = BPETokenizer().train("hola hola mundo mundo mundo de los tokens", vocab_size=270)
    ids = tok.encode("hola mundo")
    assert tok.decode(ids) == "hola mundo"
    assert tok.count("hola mundo") == len(ids)
    assert "".join(tok.pieces("hola mundo")) == "hola mundo"


def test_tokenizer_save_load(tmp_path: Path) -> None:
    tok = BPETokenizer().train("abc abc abc abd", vocab_size=260)
    tok.save(tmp_path / "bpe.json")
    assert BPETokenizer().load(tmp_path / "bpe.json").encode("abc") == tok.encode("abc")


def test_chunker_returns_sdk_chunks_with_ids() -> None:
    text = (
        "Cats are small felines. Cats purr when happy. Cats sleep a lot. "
        "The stock market fell today. Investors sold shares. Bonds rallied."
    )
    chunks = Chunker(max_tokens=50).split(Document("doc1", text, {"lang": "en"}))
    assert chunks and all(isinstance(c, Chunk) for c in chunks)
    assert [c.id for c in chunks] == [f"doc1#{c.index}" for c in chunks]
    assert all(c.metadata["lang"] == "en" and c.document_id == "doc1" for c in chunks)
    assert " ".join(c.text for c in chunks).count("Cats") == 3


def test_vector_index_scores_higher_is_better(tmp_path: Path) -> None:
    emb = HashEmbedder(64)
    index = VectorIndex(64, index_type="flat")
    for d in DOCS:
        index.add(d.id, emb.embed([d.text])[0], d.text, {"kind": "fact"})
    hits = index.search(emb.embed(["Python programming language"])[0], k=2)
    assert hits[0].id == "python" and hits[0].score >= hits[1].score
    assert hits[0].text.startswith("Python") and "dense_distance" in hits[0].signals
    index.save(tmp_path / "v.pkl")
    again = VectorIndex.load(tmp_path / "v.pkl", texts=index.texts)
    assert again.search(emb.embed(["Python programming language"])[0], k=1)[0].id == "python"


def test_hybrid_retriever_fuses_bm25_and_dense() -> None:
    retriever = HybridRetriever(HashEmbedder(128))
    retriever.add(Chunk(d.id, d.id, d.text, 0) for d in DOCS)
    hits = retriever.search("highest mountain Himalayas", k=2)
    assert hits[0].id == "everest"
    assert {"fused", "bm25"} <= set(hits[0].signals)
    assert retriever.sparse_search("sourdough", k=1)[0].id == "bread"
    assert retriever.dense_search("Eiffel Tower Paris", k=1)[0].id == "paris"


def test_empty_retriever_returns_nothing() -> None:
    assert HybridRetriever().search("anything") == []


def test_reranker_offline_keeps_hits_and_adds_signal() -> None:
    retriever = HybridRetriever(HashEmbedder(128))
    retriever.add(Chunk(d.id, d.id, d.text, 0) for d in DOCS)
    hits = retriever.search("who created python language", k=4)
    reranked = Reranker(offline=True).rerank("who created python language", hits, k=2)
    assert len(reranked) == 2 and reranked[0].id == "python"
    assert "rerank" in reranked[0].signals and "fused" in reranked[0].signals


def test_pipeline_extractive_answer_without_generator() -> None:
    rag = RAGPipeline(embedder=HashEmbedder(128))
    assert rag.ingest(DOCS, chunk=False) == len(DOCS)
    answer = rag.answer("Who created Python?")
    assert "Guido van Rossum" in answer.text and answer.model == "extractive"
    assert answer.hits[0].id.startswith("python")


def test_pipeline_uses_generator_and_prompt() -> None:
    seen: list[str] = []

    class Echo:
        model = "echo"

        def complete(self, prompt: str, *, max_tokens: int = 64, temperature: float = 0.0) -> str:
            seen.append(prompt)
            return " 1889 "

    rag = RAGPipeline(embedder=HashEmbedder(128), generator=Echo())
    rag.ingest(DOCS, chunk=False)
    answer = rag.answer("When was the Eiffel Tower finished?", k=2)
    assert answer.text == "1889" and answer.model == "echo"
    assert "Question: When was the Eiffel Tower finished?" in seen[0] and "[1]" in seen[0]


def test_document_parser_reads_text_files(tmp_path: Path) -> None:
    f = tmp_path / "notes.md"
    f.write_text("# Notes\nhello", encoding="utf-8")
    doc = DocumentParser().parse(f)
    assert doc.id == "notes.md" and "hello" in doc.text and doc.metadata["source"].endswith("notes.md")


def test_document_parser_reads_pdf(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Quarterly revenue grew 12 percent.")
    path = tmp_path / "r.pdf"
    pdf.save(str(path))
    assert "Quarterly revenue" in DocumentParser().parse(path).text


def test_graph_extractor_offline_returns_triples() -> None:
    triples = GraphExtractor().extract(
        "Marie Curie worked at the University of Paris. Pierre Curie married Marie Curie."
    )
    assert isinstance(triples, list)
    for t in triples:
        assert t.subject and t.relation and t.object


def test_empty_collaborators_passed_in_are_kept() -> None:
    """An empty retriever or index is falsy (it has __len__): it must still be the one used."""
    index = VectorIndex(64, index_type="flat")
    retriever = HybridRetriever(HashEmbedder(64), index=index)
    assert retriever.index is index
    rag = RAGPipeline(retriever=retriever)
    assert rag.retriever is retriever
    retriever.add([Chunk("a", "a", "late addition about volcanoes", 0)])
    assert rag.retrieve("volcanoes", k=1)[0].id == "a"
