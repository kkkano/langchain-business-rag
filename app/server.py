from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import get_settings
from .embeddings import EmbeddingService
from .knowledge_base import KnowledgeBaseService
from .models import (
    ChatRequest,
    ChatResponse,
    CreateSessionResponse,
    DocumentsResponse,
    IngestResponse,
    PathIngestRequest,
    ResetRequest,
    SampleIngestRequest,
)
from .rag_chain import RAGService
from .session_manager import SessionManager
from .vector_store import VectorIndex


settings = get_settings()
embedding_service = EmbeddingService(settings.embedding_model_name)
vector_index = VectorIndex(settings.vector_db_path, embedding_service)
session_manager = SessionManager()
knowledge_base = KnowledgeBaseService(settings, session_manager, vector_index)
rag_service = RAGService(settings, vector_index)


BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name)
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "api_key_configured": bool(settings.llm_api_key),
                "default_model": settings.llm_model,
                "provider_label": settings.provider_label,
                "primary_api_key_env": settings.primary_api_key_env,
            },
        )

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "base_url": settings.llm_base_url,
            "embedding_model": settings.embedding_model_name,
            "api_key_configured": bool(settings.llm_api_key),
        }

    @app.post("/api/session", response_model=CreateSessionResponse)
    def create_session():
        session = session_manager.create_session()
        return CreateSessionResponse(session_id=session.session_id)

    @app.get("/api/sessions/{session_id}/documents", response_model=DocumentsResponse)
    def list_documents(session_id: str):
        session = session_manager.get_or_create(session_id)
        return DocumentsResponse(
            session_id=session.session_id,
            documents=session_manager.list_documents(session.session_id),
        )

    @app.post("/api/documents/sample", response_model=IngestResponse)
    def ingest_sample_documents(payload: SampleIngestRequest):
        session = session_manager.get_or_create(payload.session_id)
        try:
            documents = knowledge_base.ingest_samples(session.session_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return IngestResponse(session_id=session.session_id, documents=documents)

    @app.post("/api/documents/path", response_model=IngestResponse)
    def ingest_documents_by_path(payload: PathIngestRequest):
        session = session_manager.get_or_create(payload.session_id)
        paths = [path.strip() for path in payload.paths if path.strip()]
        if not paths:
            raise HTTPException(status_code=400, detail="请至少提供一个有效路径。")
        try:
            documents = knowledge_base.ingest_paths(session.session_id, paths)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return IngestResponse(session_id=session.session_id, documents=documents)

    @app.post("/api/documents/upload", response_model=IngestResponse)
    async def ingest_documents_by_upload(
        session_id: str = Form(...),
        files: List[UploadFile] = File(...),
    ):
        session = session_manager.get_or_create(session_id)
        documents = []
        try:
            for upload in files:
                content = await upload.read()
                documents.append(
                    knowledge_base.ingest_upload(
                        session.session_id,
                        upload.filename or "uploaded.txt",
                        content,
                    )
                )
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return IngestResponse(session_id=session.session_id, documents=documents)

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest):
        session = session_manager.get_or_create(payload.session_id)
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空。")
        try:
            return rag_service.ask(session, question)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/session/reset", response_model=DocumentsResponse)
    def reset_session(payload: ResetRequest):
        session = session_manager.get_or_create(payload.session_id)
        session_manager.reset_history(session.session_id)
        if payload.clear_documents:
            knowledge_base.reset_session_documents(session.session_id)
        return DocumentsResponse(
            session_id=session.session_id,
            documents=session_manager.list_documents(session.session_id),
        )

    return app


app = create_app()
