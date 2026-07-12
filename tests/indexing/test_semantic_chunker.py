from src.indexing.semantic_chunker import SemanticChunker


def test_semantic_chunk_ids_are_stable() -> None:
    chunker = SemanticChunker(parent_tokens=30, child_tokens=12, overlap_tokens=4)
    text = "1 Customer Due Diligence\nInstitutions must identify customers. They must verify identity. Records must be retained."
    first = chunker.split_page(document_id="guide.pdf", page_number=1, text=text)
    second = chunker.split_page(document_id="guide.pdf", page_number=1, text=text)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert all(chunk.metadata["chunk_version"] == "semantic-v2" for chunk in first)


def test_child_chunks_have_parent_and_overlap() -> None:
    chunker = SemanticChunker(parent_tokens=100, child_tokens=8, overlap_tokens=3)
    text = "Alpha controls are mandatory. Beta controls are documented. Gamma reviews occur annually. Delta records are retained."
    chunks = chunker.split_page(document_id="guide.pdf", page_number=2, text=text)
    children = [chunk for chunk in chunks if chunk.chunk_type == "child"]
    assert len(children) >= 2
    assert all(child.parent_id for child in children)
    first_words = set(children[0].text.split())
    second_words = set(children[1].text.split())
    assert first_words & second_words
