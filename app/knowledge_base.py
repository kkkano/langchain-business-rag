from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
from uuid import uuid4

from .config import Settings
from .document_loader import LoadedDocument, load_document, save_uploaded_file
from .models import IndexedDocument
from .session_manager import SessionManager
from .splitter import split_document
from .vector_store import VectorIndex


class KnowledgeBaseService:
    def __init__(
        self,
        settings: Settings,
        session_manager: SessionManager,
        vector_index: VectorIndex,
    ):
        self.settings = settings
        self.session_manager = session_manager
        self.vector_index = vector_index

    def ingest_samples(self, session_id: str) -> List[IndexedDocument]:
        sample_paths = sorted(
            path for path in self.settings.sample_docs_dir.glob("*") if path.is_file()
        )
        if not sample_paths:
            raise RuntimeError("样例知识库目录为空，请先准备 sample_docs。")
        return self.ingest_paths(session_id, [str(path) for path in sample_paths])

    def ingest_paths(self, session_id: str, paths: Iterable[str]) -> List[IndexedDocument]:
        loaded_documents = [load_document(self._resolve_path(raw_path)) for raw_path in paths]
        return self._index_documents(session_id, loaded_documents)

    def ingest_upload(self, session_id: str, filename: str, content: bytes) -> IndexedDocument:
        saved_path = save_uploaded_file(self.settings.upload_dir, filename, content)
        loaded = load_document(saved_path)
        documents = self._index_documents(session_id, [loaded])
        return documents[0]

    def reset_session_documents(self, session_id: str) -> None:
        self.vector_index.clear_session(session_id)
        self.session_manager.clear_documents(session_id)

    def _index_documents(
        self,
        session_id: str,
        loaded_documents: List[LoadedDocument],
    ) -> List[IndexedDocument]:
        indexed_documents: List[IndexedDocument] = []
        for loaded_document in loaded_documents:
            document_id = uuid4().hex
            chunks = split_document(
                loaded_document=loaded_document,
                document_id=document_id,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            self.vector_index.add_chunks(session_id, chunks)
            document = IndexedDocument(
                document_id=document_id,
                source_name=loaded_document.source_name,
                source_type=loaded_document.source_type,
                source_path=loaded_document.source_path,
                chunk_count=len(chunks),
            )
            self.session_manager.add_document(session_id, document)
            indexed_documents.append(document)
        return indexed_documents

    def _resolve_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            return candidate

        search_roots = [
            Path.cwd(),
            self.settings.sample_docs_dir.parent,
            self.settings.sample_docs_dir.parent.parent,
            self.settings.sample_docs_dir.parent.parent.parent,
        ]
        for root in search_roots:
            resolved = (root / candidate).resolve()
            if resolved.exists():
                return resolved
        return (Path.cwd() / candidate).resolve()
