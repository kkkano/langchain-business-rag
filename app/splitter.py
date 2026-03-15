from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .document_loader import LoadedDocument


@dataclass
class ChunkRecord:
    chunk_id: str
    document_id: str
    source_name: str
    source_path: str
    source_type: str
    chunk_index: int
    segment_label: str
    content: str
    start_index: int


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    if chunk_size <= 0:
        raise RuntimeError("chunk_size 必须大于 0")
    if chunk_overlap < 0:
        raise RuntimeError("chunk_overlap 不能小于 0")
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size - 1

    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "; ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        keep_separator="end",
        is_separator_regex=False,
    )


def split_document(
    loaded_document: LoadedDocument,
    document_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[ChunkRecord]:
    splitter = build_splitter(chunk_size, chunk_overlap)
    documents = splitter.create_documents(
        texts=[loaded_document.text],
        metadatas=[
            {
                "document_id": document_id,
                "source_name": loaded_document.source_name,
                "source_path": loaded_document.source_path,
                "source_type": loaded_document.source_type,
            }
        ],
    )

    chunks: List[ChunkRecord] = []
    for index, document in enumerate(documents, start=1):
        source_name = document.metadata["source_name"]
        chunk_id = f"{document_id}-chunk-{index:03d}"
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                document_id=document_id,
                source_name=source_name,
                source_path=document.metadata["source_path"],
                source_type=document.metadata["source_type"],
                chunk_index=index,
                segment_label=f"第{index}段",
                content=document.page_content,
                start_index=int(document.metadata.get("start_index", 0)),
            )
        )
    return chunks
