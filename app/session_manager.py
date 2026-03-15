from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from uuid import uuid4

from langchain.memory import ConversationBufferMemory

from .models import IndexedDocument


@dataclass
class SessionState:
    session_id: str
    memory: ConversationBufferMemory
    documents: Dict[str, IndexedDocument] = field(default_factory=dict)


class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}

    def create_session(self) -> SessionState:
        session_id = uuid4().hex
        state = SessionState(
            session_id=session_id,
            memory=ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history",
                input_key="question",
                output_key="answer",
            ),
        )
        self.sessions[session_id] = state
        return state

    def get_or_create(self, session_id: Optional[str]) -> SessionState:
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        return self.create_session()

    def add_document(self, session_id: str, document: IndexedDocument) -> None:
        state = self.get_or_create(session_id)
        state.documents[document.document_id] = document

    def list_documents(self, session_id: str) -> List[IndexedDocument]:
        state = self.get_or_create(session_id)
        return list(state.documents.values())

    def reset_history(self, session_id: str) -> SessionState:
        state = self.get_or_create(session_id)
        state.memory.clear()
        return state

    def clear_documents(self, session_id: str) -> SessionState:
        state = self.get_or_create(session_id)
        state.documents.clear()
        return state

    def delete_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
