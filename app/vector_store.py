from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import List

from chromadb.config import Settings
from chromadb.telemetry.product import ProductTelemetryClient
from overrides import override

import chromadb

from .models import SourceDocument
from .splitter import ChunkRecord


class NoOpTelemetry(ProductTelemetryClient):
    @override
    def capture(self, event):
        return None


telemetry_module = types.ModuleType("rag_system_chroma_telemetry")
telemetry_module.NoOpTelemetry = NoOpTelemetry
sys.modules["rag_system_chroma_telemetry"] = telemetry_module


class VectorIndex:
    def __init__(self, db_path: Path, embedding_service):
        self.embedding_service = embedding_service
        settings = Settings(
            anonymized_telemetry=False,
            chroma_product_telemetry_impl="rag_system_chroma_telemetry.NoOpTelemetry",
            chroma_telemetry_impl="rag_system_chroma_telemetry.NoOpTelemetry",
        )
        self.client = chromadb.PersistentClient(path=str(db_path), settings=settings)

    @staticmethod
    def collection_name(session_id: str) -> str:
        return f"rag_session_{session_id.replace('-', '_')}"

    def _get_or_create_collection(self, session_id: str):
        return self.client.get_or_create_collection(
            name=self.collection_name(session_id),
            metadata={"hnsw:space": "cosine"},
        )

    def clear_session(self, session_id: str) -> None:
        try:
            self.client.delete_collection(self.collection_name(session_id))
        except Exception:
            return None

    def add_chunks(self, session_id: str, chunks: List[ChunkRecord]) -> None:
        if not chunks:
            return

        collection = self._get_or_create_collection(session_id)
        embeddings = self.embedding_service.embed_documents([chunk.content for chunk in chunks])
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "source_name": chunk.source_name,
                    "source_path": chunk.source_path,
                    "source_type": chunk.source_type,
                    "chunk_index": chunk.chunk_index,
                    "segment_label": chunk.segment_label,
                    "start_index": chunk.start_index,
                }
                for chunk in chunks
            ],
            embeddings=embeddings,
        )

    def search(self, session_id: str, query: str, top_k: int) -> List[SourceDocument]:
        collection = self._get_or_create_collection(session_id)
        if collection.count() == 0:
            return []

        query_embedding = [self.embedding_service.embed_query(query)]
        results = collection.query(query_embeddings=query_embedding, n_results=top_k)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        source_documents: List[SourceDocument] = []

        for metadata, content, distance in zip(metadatas, documents, distances):
            score = max(0.0, min(1.0, 1 - float(distance)))
            source_documents.append(
                SourceDocument(
                    source_id=f"{metadata['source_name']}::{metadata['segment_label']}",
                    document_id=str(metadata["document_id"]),
                    source_name=str(metadata["source_name"]),
                    source_path=str(metadata["source_path"]),
                    source_type=str(metadata["source_type"]),
                    chunk_index=int(metadata["chunk_index"]),
                    segment_label=str(metadata["segment_label"]),
                    content=content,
                    score=score,
                )
            )
        return source_documents
