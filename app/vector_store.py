from __future__ import annotations

import logging
import re
import sys
import types
from pathlib import Path
from typing import Dict, List

from chromadb.config import Settings
from chromadb.telemetry.product import ProductTelemetryClient
from overrides import override
from rank_bm25 import BM25Okapi

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

logger = logging.getLogger(__name__)


class VectorIndex:
    DENSE_WEIGHT = 0.65
    KEYWORD_WEIGHT = 0.35
    TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")

    def __init__(
        self,
        db_path: Path,
        embedding_service,
        reranker=None,
        candidate_top_k: int = 12,
    ):
        self.embedding_service = embedding_service
        self.reranker = reranker
        self.candidate_top_k = max(candidate_top_k, 1)
        self.keyword_chunks: Dict[str, List[ChunkRecord]] = {}
        self.keyword_indices: Dict[str, BM25Okapi] = {}
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
        self.keyword_chunks.pop(session_id, None)
        self.keyword_indices.pop(session_id, None)
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
        self._add_keyword_chunks(session_id, chunks)

    def search(self, session_id: str, query: str, top_k: int) -> List[SourceDocument]:
        candidate_k = max(top_k, self.candidate_top_k)
        dense_results = self._dense_search(session_id, query, candidate_k)
        keyword_results = self._keyword_search(session_id, query, candidate_k)
        merged_results = self._merge_results(dense_results, keyword_results, candidate_k)
        return self._rerank_results(query, merged_results, top_k)

    def _dense_search(self, session_id: str, query: str, top_k: int) -> List[SourceDocument]:
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

    def _keyword_search(self, session_id: str, query: str, top_k: int) -> List[SourceDocument]:
        bm25 = self.keyword_indices.get(session_id)
        chunks = self.keyword_chunks.get(session_id, [])
        if not bm25 or not chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        raw_scores = bm25.get_scores(query_tokens)
        if len(raw_scores) == 0:
            return []

        max_score = max(float(score) for score in raw_scores)
        if max_score <= 0:
            return []

        ranked_indices = sorted(
            range(len(raw_scores)),
            key=lambda index: raw_scores[index],
            reverse=True,
        )

        keyword_results: List[SourceDocument] = []
        for index in ranked_indices:
            raw_score = float(raw_scores[index])
            if raw_score <= 0:
                continue
            chunk = chunks[index]
            normalized_score = raw_score / max_score
            keyword_results.append(self._chunk_to_source_document(chunk, normalized_score))
            if len(keyword_results) >= top_k:
                break
        return keyword_results

    def _add_keyword_chunks(self, session_id: str, chunks: List[ChunkRecord]) -> None:
        session_chunks = self.keyword_chunks.setdefault(session_id, [])
        session_chunks.extend(chunks)

        tokenized_corpus = []
        for chunk in session_chunks:
            tokens = self._tokenize(chunk.content)
            tokenized_corpus.append(tokens or ["__empty__"])
        self.keyword_indices[session_id] = BM25Okapi(tokenized_corpus)

    def _merge_results(
        self,
        dense_results: List[SourceDocument],
        keyword_results: List[SourceDocument],
        top_k: int,
    ) -> List[SourceDocument]:
        merged: Dict[str, Dict[str, object]] = {}

        for document in dense_results:
            merged[document.source_id] = {
                "document": document,
                "dense_score": document.score,
                "keyword_score": 0.0,
            }

        for document in keyword_results:
            if document.source_id not in merged:
                merged[document.source_id] = {
                    "document": document,
                    "dense_score": 0.0,
                    "keyword_score": document.score,
                }
                continue
            merged[document.source_id]["keyword_score"] = max(
                float(merged[document.source_id]["keyword_score"]),
                document.score,
            )

        reranked: List[SourceDocument] = []
        for item in merged.values():
            dense_score = float(item["dense_score"])
            keyword_score = float(item["keyword_score"])
            fused_score = (dense_score * self.DENSE_WEIGHT) + (
                keyword_score * self.KEYWORD_WEIGHT
            )
            document = item["document"]
            reranked.append(
                SourceDocument(
                    source_id=document.source_id,
                    document_id=document.document_id,
                    source_name=document.source_name,
                    source_path=document.source_path,
                    source_type=document.source_type,
                    chunk_index=document.chunk_index,
                    segment_label=document.segment_label,
                    content=document.content,
                    score=fused_score,
                )
            )

        reranked.sort(key=lambda document: document.score, reverse=True)
        return reranked[:top_k]

    def _rerank_results(
        self,
        query: str,
        candidates: List[SourceDocument],
        top_k: int,
    ) -> List[SourceDocument]:
        if not candidates:
            return []
        if self.reranker is None:
            return candidates[:top_k]

        try:
            return self.reranker.rerank(query, candidates, top_k)
        except RuntimeError as exc:
            logger.warning("reranker disabled after load failure: %s", exc)
            self.reranker = None
            return candidates[:top_k]

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        tokens: List[str] = []
        for token in cls.TOKEN_PATTERN.findall(text.lower()):
            if re.fullmatch(r"[\u4e00-\u9fff]+", token):
                tokens.extend(list(token))
                if len(token) <= 6:
                    tokens.append(token)
                if len(token) >= 2:
                    tokens.extend(
                        token[index : index + 2] for index in range(len(token) - 1)
                    )
                continue
            tokens.append(token)
        return tokens

    @staticmethod
    def _chunk_to_source_document(chunk: ChunkRecord, score: float) -> SourceDocument:
        return SourceDocument(
            source_id=f"{chunk.source_name}::{chunk.segment_label}",
            document_id=chunk.document_id,
            source_name=chunk.source_name,
            source_path=chunk.source_path,
            source_type=chunk.source_type,
            chunk_index=chunk.chunk_index,
            segment_label=chunk.segment_label,
            content=chunk.content,
            score=score,
        )
