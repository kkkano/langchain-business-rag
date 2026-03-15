from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class IndexedDocument(BaseModel):
    document_id: str
    source_name: str
    source_type: str
    source_path: str
    chunk_count: int


class SourceDocument(BaseModel):
    source_id: str
    document_id: str
    source_name: str
    source_path: str
    source_type: str
    chunk_index: int
    segment_label: str
    content: str
    score: float = Field(description="1 越接近 1 越相关")


class Citation(BaseModel):
    source_id: str = Field(description="必须来自上下文中的 source_id")
    source_name: str
    segment_label: str
    supporting_text: str = Field(description="引用的原文摘录")


class StructuredAnswer(BaseModel):
    answer: str = Field(
        description="严格基于上下文作答；若证据不足则返回“我不知道”。如果给出结论，请在相关句子后附上 [source_id] 标记。"
    )
    grounded: bool = Field(description="答案是否被上下文直接支持")
    citations: List[Citation] = Field(default_factory=list)


class CreateSessionResponse(BaseModel):
    session_id: str


class DocumentsResponse(BaseModel):
    session_id: str
    documents: List[IndexedDocument]


class IngestResponse(BaseModel):
    session_id: str
    documents: List[IndexedDocument]


class PathIngestRequest(BaseModel):
    session_id: str
    paths: List[str]


class SampleIngestRequest(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    grounded: bool
    rewritten_question: str
    citations: List[Citation]
    source_documents: List[SourceDocument]


class ResetRequest(BaseModel):
    session_id: str
    clear_documents: bool = False
