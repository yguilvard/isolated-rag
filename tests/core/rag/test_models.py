from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

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


def test_embedding_defaults_owner_id_none_visibility_private():
    chunk = Chunk(document_path=Path("f.txt"), index=0, content="hi")
    emb = Embedding(chunk=chunk, vector=[0.1], model="nomic-embed-text")
    assert emb.owner_id is None
    assert emb.visibility == "private"


def test_embedding_accepts_owner_id_and_public_visibility():
    uid = uuid4()
    chunk = Chunk(document_path=Path("f.txt"), index=0, content="hi")
    emb = Embedding(chunk=chunk, vector=[0.1], model="m", owner_id=uid, visibility="public")
    assert emb.owner_id == uid
    assert emb.visibility == "public"


def test_embedding_rejects_invalid_visibility():
    chunk = Chunk(document_path=Path("f.txt"), index=0, content="hi")
    with pytest.raises(ValidationError):
        Embedding(chunk=chunk, vector=[0.1], model="m", visibility="internal")
