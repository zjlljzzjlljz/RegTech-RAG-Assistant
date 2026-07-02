from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz
import numpy as np
from pymilvus import AnnSearchRequest, Collection, CollectionSchema, DataType, FieldSchema, MilvusClient, connections, utility

from config.settings import Settings, get_settings
from src.indexing.sparse_tokenizer import tokenize_and_weight

logger = logging.getLogger(__name__)


class DependencyUnavailableError(RuntimeError):
    """Raised when Milvus or embedding infrastructure is unavailable."""


@dataclass
class IndexedChunk:
    chunk_id: str
    parent_id: str | None
    text: str
    source_file: str
    page_number: int | None
    chunk_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BGEEmbeddingResult:
    dense_vector: list[float]
    sparse_vector: dict[int, float]


class BGEM3EmbeddingClient:
    """Local BGEM3 client with a simplified sparse representation fallback."""

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer as _ST

        settings = get_settings()
        self.model_name = model_name or settings.inference.embedding_model_name
        self._ST = _ST
        self.model = _ST(self.model_name, trust_remote_code=True)

    def encode(self, text: str, prompt: str | None = None) -> BGEEmbeddingResult:
        kwargs: dict[str, Any] = {"normalize_embeddings": True}
        if prompt:
            kwargs["prompt"] = prompt
        dense_raw = self.model.encode(text, **kwargs)
        if dense_raw.ndim == 2:
            dense_vec = dense_raw[0].tolist()
        else:
            dense_vec = dense_raw.tolist()
        sparse_vec = self._build_sparse_vector_fallback(text)
        return BGEEmbeddingResult(dense_vector=dense_vec, sparse_vector=sparse_vec)

    def encode_many(self, texts: list[str], prompt: str | None = None) -> list[BGEEmbeddingResult]:
        if not texts:
            return []
        kwargs: dict[str, Any] = {"normalize_embeddings": True}
        if prompt:
            kwargs["prompt"] = prompt
        dense_arrays = self.model.encode(texts, **kwargs)
        output: list[BGEEmbeddingResult] = []
        for i, text in enumerate(texts):
            dense_vec = np.asarray(dense_arrays[i], dtype=np.float32).tolist()
            sparse_vec = self._build_sparse_vector_fallback(text)
            output.append(BGEEmbeddingResult(dense_vector=dense_vec, sparse_vector=sparse_vec))
        return output

    def _build_sparse_vector_from_result(self, result: dict[str, Any]) -> dict[int, float]:
        """Extract sparse vector from BGE-M3 encode result, with fallback."""
        try:
            sparse = result.get("sparse")
            if sparse is None:
                raise ValueError("No sparse key in result")

            # scipy sparse matrix: convert to {idx: weight} dict
            from scipy.sparse import issparse

            if issparse(sparse):
                coo = sparse.tocoo()
                return {int(idx): float(val) for idx, val in zip(coo.col, coo.data)}
            # dict format: {"indices": [...], "values": [...]} or {str: float}
            if isinstance(sparse, dict):
                indices = sparse.get("indices") or []
                values = sparse.get("values") or []
                if indices and values:
                    return {int(idx): float(val) for idx, val in zip(indices, values)}
        except Exception:
            pass
        # Fallback: use shared deterministic tokenizer
        return self._build_sparse_vector_fallback("")

    def _build_sparse_vector(self, text: str) -> dict[int, float]:
        """Build sparse vector for *text* using the shared deterministic tokenizer."""
        return tokenize_and_weight(text)

    def _build_sparse_vector_fallback(self, text: str) -> dict[int, float]:
        """Build sparse vector using the shared deterministic tokenizer.

        This is the single source of truth for all sparse vectors at both
        ingest time and query time.  Uses blake2b (not Python hash()) so
        results are identical across Python processes.
        """
        return tokenize_and_weight(text)


class MilvusHybridStore:
    """Milvus storage wrapper with dual dense+sparse schema support."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.collection_name = self.settings.milvus.collection_name
        self._client: MilvusClient | None = None
        self._collection: Collection | None = None

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            self.connect()
        assert self._client is not None
        return self._client

    @property
    def collection(self) -> Collection:
        if self._collection is None:
            self.connect()
        assert self._collection is not None
        return self._collection

    def connect(self) -> None:
        try:
            connections.connect(
                alias=self.settings.milvus.alias,
                host=self.settings.milvus.host,
                port=self.settings.milvus.port,
                user=self.settings.milvus.user or None,
                password=self.settings.milvus.password or None,
                db_name=self.settings.milvus.database,
            )
            self._client = MilvusClient(uri=self.settings.milvus.uri)
            if utility.has_collection(self.collection_name, using=self.settings.milvus.alias):
                self._collection = Collection(self.collection_name, using=self.settings.milvus.alias)
            logger.info("Connected to Milvus at %s", self.settings.milvus.uri)
        except Exception as exc:  # noqa: BLE001
            raise DependencyUnavailableError(f"Failed to connect to Milvus: {exc}") from exc

    def build_schema(self, dense_dimension: int) -> CollectionSchema:
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=128),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=128, is_nullable=True),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="page_number", dtype=DataType.INT64, is_nullable=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dimension),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=65535),
        ]
        return CollectionSchema(fields=fields, description="RegTech compliance chunks")

    def ensure_collection(self, dense_dimension: int) -> None:
        self.connect()
        if utility.has_collection(self.collection_name, using=self.settings.milvus.alias):
            self._collection = Collection(self.collection_name, using=self.settings.milvus.alias)
            self._collection.load()
            return

        schema = self.build_schema(dense_dimension)
        collection = Collection(
            name=self.collection_name,
            schema=schema,
            using=self.settings.milvus.alias,
            consistency_level=self.settings.milvus.consistency_level,
        )
        collection.create_index(
            field_name="dense_vector",
            index_params={
                "index_type": self.settings.milvus.dense_index_type,
                "metric_type": self.settings.milvus.dense_metric_type,
                "params": {"M": 16, "efConstruction": 200},
            },
        )
        collection.create_index(
            field_name="sparse_vector",
            index_params={
                "index_type": self.settings.milvus.sparse_index_type,
                "metric_type": self.settings.milvus.sparse_metric_type,
                "params": {"drop_ratio_build": 0.1},
            },
        )
        collection.load()
        self._collection = collection

    def insert_chunks(self, chunks: list[IndexedChunk], embeddings: list[BGEEmbeddingResult]) -> int:
        if not chunks:
            return 0
        self.ensure_collection(len(embeddings[0].dense_vector))
        payload = [
            [chunk.chunk_id for chunk in chunks],
            [chunk.parent_id or "" for chunk in chunks],
            [chunk.chunk_type for chunk in chunks],
            [chunk.source_file for chunk in chunks],
            [chunk.page_number if chunk.page_number is not None else -1 for chunk in chunks],
            [chunk.text for chunk in chunks],
            [embedding.dense_vector for embedding in embeddings],
            [embedding.sparse_vector for embedding in embeddings],
            [json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True) for chunk in chunks],
        ]
        result = self.collection.insert(payload)
        self.collection.flush()
        return int(result.insert_count)

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        top_k: int,
        output_fields: list[str],
        filters: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self.collection.load()
        dense_hits = self.collection.search(
            data=[dense_vector],
            anns_field="dense_vector",
            param={"metric_type": self.settings.milvus.dense_metric_type, "params": {"ef": self.settings.milvus.search_probe}},
            limit=top_k,
            expr=filters,
            output_fields=output_fields,
        )[0]
        sparse_hits = self.collection.search(
            data=[sparse_vector],
            anns_field="sparse_vector",
            param={"metric_type": self.settings.milvus.sparse_metric_type, "params": {"drop_ratio_search": 0.0}},
            limit=top_k,
            expr=filters,
            output_fields=output_fields,
        )[0]
        return self._normalize_hits(dense_hits), self._normalize_hits(sparse_hits)

    def sparse_only_search(
        self,
        sparse_vector: dict[int, float],
        top_k: int,
        output_fields: list[str],
        filters: str | None = None,
    ) -> list[dict[str, Any]]:
        """Pure sparse-vector search — dense vector is bypassed.

        Used by eval_retrieval.py --sparse-only for diagnostic purposes only.
        """
        self.collection.load()
        sparse_hits = self.collection.search(
            data=[sparse_vector],
            anns_field="sparse_vector",
            param={"metric_type": self.settings.milvus.sparse_metric_type, "params": {"drop_ratio_search": 0.0}},
            limit=top_k,
            expr=filters,
            output_fields=output_fields,
        )[0]
        return self._normalize_hits(sparse_hits)

    def dense_only_search(
        self,
        dense_vector: list[float],
        top_k: int,
        output_fields: list[str],
        filters: str | None = None,
    ) -> list[dict[str, Any]]:
        """Pure dense-vector search — sparse vector is bypassed.

        Used by eval_retrieval.py --fusion dense-only for diagnostic purposes.
        """
        self.collection.load()
        dense_hits = self.collection.search(
            data=[dense_vector],
            anns_field="dense_vector",
            param={"metric_type": self.settings.milvus.dense_metric_type, "params": {"ef": self.settings.milvus.search_probe}},
            limit=top_k,
            expr=filters,
            output_fields=output_fields,
        )[0]
        return self._normalize_hits(dense_hits)

    def _normalize_hits(self, hits: list[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for hit in hits:
            entity = hit.entity
            normalized.append(
                {
                    "chunk_id": entity.get("chunk_id"),
                    "parent_id": entity.get("parent_id") or None,
                    "chunk_type": entity.get("chunk_type"),
                    "source_file": entity.get("source_file"),
                    "page_number": None if entity.get("page_number") == -1 else entity.get("page_number"),
                    "text": entity.get("text"),
                    "metadata": json.loads(entity.get("metadata_json") or "{}"),
                    "score": float(hit.score),
                }
            )
        return normalized


class MilvusIndexer:
    """PDF parsing, parent-child chunking, and Milvus ingest orchestration."""

    def __init__(
        self,
        store: MilvusHybridStore | None = None,
        embedding_client: BGEM3EmbeddingClient | None = None,
    ) -> None:
        self.settings = get_settings()
        self.store = store or MilvusHybridStore(self.settings)
        self.embedding_client = embedding_client or BGEM3EmbeddingClient()

    def parse_pdf(self, pdf_path: str | Path) -> list[dict[str, Any]]:
        document = fitz.open(pdf_path)
        pages: list[dict[str, Any]] = []
        for page_index, page in enumerate(document):
            text = page.get_text("text").strip()
            if not text:
                continue
            pages.append(
                {
                    "page_number": page_index + 1,
                    "text": text,
                }
            )
        return pages

    def build_parent_child_chunks(
        self,
        pages: list[dict[str, Any]],
        parent_size: int = 1500,
        child_size: int = 400,
    ) -> list[IndexedChunk]:
        chunks: list[IndexedChunk] = []
        for page in pages:
            words = page["text"].split()
            parent_index = 0
            for start in range(0, len(words), parent_size):
                parent_words = words[start : start + parent_size]
                if not parent_words:
                    continue
                parent_id = f"page-{page['page_number']}-parent-{parent_index}"
                parent_text = " ".join(parent_words)
                chunks.append(
                    IndexedChunk(
                        chunk_id=parent_id,
                        parent_id=None,
                        text=parent_text,
                        source_file="",
                        page_number=page["page_number"],
                        chunk_type="parent",
                        metadata={"page": page["page_number"]},
                    )
                )
                child_index = 0
                for child_start in range(0, len(parent_words), child_size):
                    child_words = parent_words[child_start : child_start + child_size]
                    if not child_words:
                        continue
                    child_id = f"{parent_id}-child-{child_index}"
                    chunks.append(
                        IndexedChunk(
                            chunk_id=child_id,
                            parent_id=parent_id,
                            text=" ".join(child_words),
                            source_file="",
                            page_number=page["page_number"],
                            chunk_type="child",
                            metadata={"page": page["page_number"], "parent_id": parent_id},
                        )
                    )
                    child_index += 1
                parent_index += 1
        return chunks

    def ingest_pdf(self, pdf_path: str | Path) -> int:
        pdf_path = Path(pdf_path)
        pages = self.parse_pdf(pdf_path)
        chunks = self.build_parent_child_chunks(pages)
        for chunk in chunks:
            chunk.source_file = pdf_path.name
            chunk.metadata["source_file"] = pdf_path.name
        embeddings = self.embedding_client.encode_many([chunk.text for chunk in chunks])
        inserted = self.store.insert_chunks(chunks, embeddings)
        logger.info("Indexed %s chunks from %s", inserted, pdf_path)
        return inserted


__all__ = [
    "BGEEmbeddingResult",
    "BGEM3EmbeddingClient",
    "DependencyUnavailableError",
    "IndexedChunk",
    "MilvusHybridStore",
    "MilvusIndexer",
]
