from pathlib import Path

from src.core.rag.models import Chunk, Document, Embedding


def test_document_stores_path_content_mime():
    doc = Document(path=Path("file.txt"), content="hello", mime_type="text/plain")
    assert doc.path == Path("file.txt")
    assert doc.content == "hello"
    assert doc.mime_type == "text/plain"


def test_chunk_stores_document_path_index_content():
    chunk = Chunk(document_path=Path("file.txt"), index=0, content="hello world")
    assert chunk.document_path == Path("file.txt")
    assert chunk.index == 0
    assert chunk.content == "hello world"


def test_embedding_stores_chunk_vector_model():
    chunk = Chunk(document_path=Path("file.txt"), index=0, content="hello")
    emb = Embedding(chunk=chunk, vector=[0.1, 0.2, 0.3], model="nomic-embed-text")
    assert emb.vector == [0.1, 0.2, 0.3]
    assert emb.model == "nomic-embed-text"
    assert emb.chunk is chunk
