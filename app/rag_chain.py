from __future__ import annotations

from typing import List

from langchain_openai import ChatOpenAI

from .config import Settings
from .models import ChatResponse, Citation, SourceDocument, StructuredAnswer
from .prompts import build_answer_prompt, build_question_rewrite_prompt
from .session_manager import SessionState
from .vector_store import VectorIndex


def format_context(source_documents: List[SourceDocument]) -> str:
    if not source_documents:
        return "没有可用上下文。"

    blocks = []
    for document in source_documents:
        blocks.append(
            f"[source_id={document.source_id} | source_name={document.source_name} | "
            f"segment={document.segment_label} | score={document.score:.4f}]\n{document.content}"
        )
    return "\n\n".join(blocks)


class RAGService:
    def __init__(self, settings: Settings, vector_index: VectorIndex):
        self.settings = settings
        self.vector_index = vector_index
        self._rewrite_prompt = build_question_rewrite_prompt()
        self._answer_prompt = build_answer_prompt()

    def _build_llm(self):
        if not self.settings.llm_api_key:
            raise RuntimeError(
                "未检测到模型 API Key，请先配置 DEEPSEEK_API_KEY，"
                "或在 OpenAI 兼容模式下配置 OPENAI_API_KEY。"
            )

        kwargs = {
            "model": self.settings.llm_model,
            "api_key": self.settings.llm_api_key,
            "temperature": 0,
        }
        if self.settings.llm_base_url:
            kwargs["base_url"] = self.settings.llm_base_url
        return ChatOpenAI(**kwargs)

    @staticmethod
    def _message_to_text(message) -> str:
        content = message.content
        if isinstance(content, str):
            return content.strip()

        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts).strip()

    def _rewrite_question(self, llm: ChatOpenAI, chat_history, question: str) -> str:
        prompt_value = self._rewrite_prompt.invoke(
            {
                "chat_history": chat_history,
                "question": question,
            }
        )
        response_message = llm.invoke(prompt_value)
        return self._message_to_text(response_message) or question

    def _generate_structured_answer(
        self,
        structured_llm,
        chat_history,
        question: str,
        context: str,
    ) -> StructuredAnswer:
        prompt_value = self._answer_prompt.invoke(
            {
                "question": question,
                "chat_history": chat_history,
                "context": context,
            }
        )
        return structured_llm.invoke(prompt_value)

    def ask(self, session: SessionState, question: str) -> ChatResponse:
        if not session.documents:
            raise RuntimeError("当前会话还没有知识库文档，请先上传文档或加载样例知识库。")

        llm = self._build_llm()
        structured_llm = llm.with_structured_output(StructuredAnswer)
        chat_history = session.memory.load_memory_variables({}).get("chat_history", [])

        # Step 1: rewrite the follow-up question into a standalone query.
        standalone_question = self._rewrite_question(llm, chat_history, question)

        # Step 2: retrieve source documents from the local vector store.
        source_documents = self.vector_index.search(
            session_id=session.session_id,
            query=standalone_question,
            top_k=self.settings.top_k,
        )
        context = format_context(source_documents)

        # Step 3: ask the model to answer strictly from retrieved context.
        structured_answer = self._generate_structured_answer(
            structured_llm=structured_llm,
            chat_history=chat_history,
            question=question,
            context=context,
        )
        citations = self._sanitize_citations(structured_answer.citations, source_documents)
        answer_text = structured_answer.answer.strip() or "我不知道"
        grounded = bool(structured_answer.grounded and citations)
        if not grounded and answer_text != "我不知道":
            answer_text = "我不知道"
            citations = []

        response = ChatResponse(
            session_id=session.session_id,
            answer=answer_text,
            grounded=grounded,
            rewritten_question=standalone_question,
            citations=citations,
            source_documents=source_documents,
        )
        session.memory.save_context({"question": question}, {"answer": response.answer})
        return response

    @staticmethod
    def _sanitize_citations(
        citations: List[Citation],
        source_documents: List[SourceDocument],
    ) -> List[Citation]:
        valid_sources = {document.source_id: document for document in source_documents}
        sanitized: List[Citation] = []
        for citation in citations:
            if citation.source_id not in valid_sources:
                continue
            source_document = valid_sources[citation.source_id]
            sanitized.append(
                Citation(
                    source_id=citation.source_id,
                    source_name=source_document.source_name,
                    segment_label=source_document.segment_label,
                    supporting_text=citation.supporting_text.strip(),
                )
            )
        return sanitized
