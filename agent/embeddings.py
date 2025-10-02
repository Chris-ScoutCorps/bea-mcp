"""Embedding utilities to generate document and query vectors consistent with the
vector index definition in MongoDB Atlas (knnVector, 1536 dims, cosine).

Uses OpenAI text-embedding-3-small by default (1536 dimensions).
Set OPENAI_API_KEY in your environment (.env) before use.

If you later change model or text construction, you MUST re-embed documents.
"""
from __future__ import annotations

import os
from typing import List, Dict, Any

try:
    from openai import OpenAI  # openai>=1.x
except ImportError as e:  # pragma: no cover
    raise ImportError("openai package not installed; ensure it is in pyproject.toml") from e

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EXPECTED_DIM = 1536  # Keep aligned with create_vector_search_index()
_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        _client = OpenAI(api_key=api_key)
    return _client

def _do_embed(texts: List[str]) -> List[List[float]]:
    client = _get_client()
    # Strip and ensure non-empty (OpenAI rejects empty strings)
    cleaned = [t.strip() or "(empty)" for t in texts]
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=cleaned)
    vectors = [d.embedding for d in resp.data]
    for v in vectors:
        if len(v) != EXPECTED_DIM:
            raise ValueError(f"Embedding dimension {len(v)} != expected {EXPECTED_DIM}")
    return vectors

def embed_text(text: str) -> List[float]:
    return _do_embed([text])[0]

def build_document_embedding_text(doc: Dict[str, Any]) -> str:
    """Construct the canonical text used for embedding a data_lookup document.

    Order matters—keep stable so re-embedding yields consistent semantics.
    Fields missing in some docs are skipped gracefully.
    """
    parts: List[str] = []
    for field in ("dataset_name", "table_name", "dataset_description", "table_description"):
        val = doc.get(field)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    # Flatten other_parameters (array of {parameter_name, parameter_description})
    other_params = doc.get("other_parameters") or []
    if isinstance(other_params, list):
        flattened = []
        for p in other_params:
            if not isinstance(p, dict):
                continue
            pn = p.get("parameter_name", "")
            pd = p.get("parameter_description", "")
            seg = f"{pn} {pd}".strip()
            if seg:
                flattened.append(seg)
        if flattened:
            parts.append("; ".join(flattened))
    return "\n".join(parts)

def embed_document_in_place(doc: Dict[str, Any]) -> Dict[str, Any]:
    text = build_document_embedding_text(doc)
    doc["embedding"] = embed_text(text)
    return doc

def embed_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    texts = [build_document_embedding_text(d) for d in docs]
    vectors = _do_embed(texts)
    for d, v in zip(docs, vectors):
        d["embedding"] = v
    return docs

def embed_query(query: str) -> List[float]:
    """Embed a user query string using the same model and preprocessing philosophy.

    We do NOT augment the query with document fields—just raw user text. You can
    experiment with light prompt engineering (e.g., prefixing domain hints) if needed.
    """
    return embed_text(query)
