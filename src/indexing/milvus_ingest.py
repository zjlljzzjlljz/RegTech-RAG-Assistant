from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz
import httpx
import numpy as np
from pymilvus import AnnSearchRequest, Collection, CollectionSchema, DataType, FieldSchema, MilvusClient, connections, utility

from config.settings import Settings, get_settings
from src.indexing.sparse_tokenizer import tokenize_and_weight
from src.indexing.semantic_chunker import SemanticChunker

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
    """BGE-M3 hybrid encoder with native lexical weights on both write and query paths."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.inference.embedding_model_name
        self.service_url = settings.inference.embedding_service_url
        self.timeout = settings.inference.request_timeout_seconds
        self.batch_size = settings.inference.embedding_batch_size
        self.max_length = settings.inference.embedding_max_length
        self.allow_legacy_fallback = settings.inference.prefer_local_fallback
        self.model: Any = None
        self._native_sparse = False
        if not self.service_url:
            try:
                from FlagEmbedding import BGEM3FlagModel

                self.model = BGEM3FlagModel(self.model_name, use_fp16=True)
                self._native_sparse = True
            except Exception as exc:
                if not self.allow_legacy_fallback:
                    raise DependencyUnavailableError(
                        "BGE-M3 native sparse encoder is unavailable; install FlagEmbedding "
                        "or configure EMBEDDING_SERVICE_URL"
                    ) from exc
                from sentence_transformers import SentenceTransformer

                logger.warning("Using legacy deterministic sparse fallback: %s", exc)
                self.model = SentenceTransformer(self.model_name, trust_remote_code=True)

    def encode(self, text: str, prompt: str | None = None) -> BGEEmbeddingResult:
        return self.encode_many([text], prompt=prompt)[0]

    def encode_many(self, texts: list[str], prompt: str | None = None) -> list[BGEEmbeddingResult]:
        if not texts:
            return []
        if self.service_url:
            output: list[BGEEmbeddingResult] = []
            for start in range(0, len(texts), self.batch_size):
                response = httpx.post(
                    f"{self.service_url.rstrip('/')}/encode",
                    json={
                        "texts": texts[start : start + self.batch_size],
                        "batch_size": self.batch_size,
                        "max_length": self.max_length,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("model") != self.model_name:
                    raise DependencyUnavailableError(
                        f"Embedding service model mismatch: expected {self.model_name}, got {payload.get('model')}"
                    )
                output.extend(
                    BGEEmbeddingResult(
                        dense_vector=[float(value) for value in item["dense_vector"]],
                        sparse_vector={int(key): float(value) for key, value in item["sparse_vector"].items()},
                    )
                    for item in payload["embeddings"]
                )
            return output

        if self._native_sparse:
            result = self.model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            dense_arrays = result["dense_vecs"]
            sparse_arrays = result["lexical_weights"]
        else:
            dense_arrays = self.model.encode(texts, normalize_embeddings=True)
            sparse_arrays = [self._build_sparse_vector_fallback(text) for text in texts]
        output: list[BGEEmbeddingResult] = []
        for i, _text in enumerate(texts):
            dense_vec = np.asarray(dense_arrays[i], dtype=np.float32).tolist()
            raw_sparse = sparse_arrays[i]
            sparse_vec = {int(key): float(value) for key, value in raw_sparse.items()}
            output.append(BGEEmbeddingResult(dense_vector=dense_vec, sparse_vector=sparse_vec))
        return output

    def _build_sparse_vector(self, text: str) -> dict[int, float]:
        return self.encode(text).sparse_vector

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

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        escaped = [chunk_id.replace('"', '\\"') for chunk_id in chunk_ids]
        expr = "chunk_id in [" + ",".join(f'"{chunk_id}"' for chunk_id in escaped) + "]"
        rows = self.collection.query(
            expr=expr,
            output_fields=[
                "chunk_id",
                "parent_id",
                "chunk_type",
                "source_file",
                "page_number",
                "text",
                "metadata_json",
            ],
        )
        return [
            {
                **row,
                "page_number": None if row.get("page_number") == -1 else row.get("page_number"),
                "metadata": json.loads(row.get("metadata_json") or "{}"),
            }
            for row in rows
        ]

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
        self._embedding_client = embedding_client

    @property
    def embedding_client(self) -> BGEM3EmbeddingClient:
        if self._embedding_client is None:
            self._embedding_client = BGEM3EmbeddingClient()
        return self._embedding_client

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
        parent_size: int | None = None,
        child_size: int | None = None,
        overlap_size: int | None = None,
        document_id: str = "unknown-document",
    ) -> list[IndexedChunk]:
        chunker = SemanticChunker(
            parent_tokens=parent_size or self.settings.chunking.parent_tokens,
            child_tokens=child_size or self.settings.chunking.child_tokens,
            overlap_tokens=(
                self.settings.chunking.overlap_tokens if overlap_size is None else overlap_size
            ),
            version=self.settings.chunking.version,
        )
        chunks: list[IndexedChunk] = []
        for page in pages:
            semantic_chunks = chunker.split_page(
                document_id=document_id,
                page_number=page["page_number"],
                text=page["text"],
            )
            for semantic_chunk in semantic_chunks:
                chunks.append(
                    IndexedChunk(
                        chunk_id=semantic_chunk.chunk_id,
                        parent_id=semantic_chunk.parent_id,
                        text=semantic_chunk.text,
                        source_file="",
                        page_number=page["page_number"],
                        chunk_type=semantic_chunk.chunk_type,
                        metadata=dict(semantic_chunk.metadata),
                    )
                )
        return chunks

    def ingest_pdf(self, pdf_path: str | Path) -> int:
        pdf_path = Path(pdf_path)
        pages = self.parse_pdf(pdf_path)
        chunks = self.build_parent_child_chunks(pages, document_id=pdf_path.name)
        for chunk in chunks:
            chunk.source_file = pdf_path.name
            chunk.metadata["source_file"] = pdf_path.name
            chunk.metadata["embedding_model"] = self.embedding_client.model_name
        child_positions = [index for index, chunk in enumerate(chunks) if chunk.chunk_type == "child"]
        child_embeddings = self.embedding_client.encode_many([chunks[index].text for index in child_positions])
        if not child_embeddings:
            raise DependencyUnavailableError(f"No child chunks were produced for {pdf_path.name}")
        dimension = len(child_embeddings[0].dense_vector)
        embeddings = [BGEEmbeddingResult([0.0] * dimension, {}) for _ in chunks]
        for position, embedding in zip(child_positions, child_embeddings):
            embeddings[position] = embedding
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
