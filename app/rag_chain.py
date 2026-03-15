from __future__ import annotations

from operator import itemgetter
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
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
        if not self.settings.openai_api_key:
            raise RuntimeError("未检测到 OPENAI_API_KEY，请先配置后再发起对话。")

        kwargs = {
            "model": self.settings.openai_model,
            "api_key": self.settings.openai_api_key,
            "temperature": 0,
        }
        if self.settings.openai_base_url:
            kwargs["base_url"] = self.settings.openai_base_url
        return ChatOpenAI(**kwargs)

    def ask(self, session: SessionState, question: str) -> ChatResponse:
        if not session.documents:
            raise RuntimeError("当前会话还没有知识库文档，请先上传文档或加载样例知识库。")

        llm = self._build_llm()
        structured_llm = llm.with_structured_output(StructuredAnswer)
        chat_history = session.memory.load_memory_variables({}).get("chat_history", [])

        rewrite_chain = self._rewrite_prompt | llm | StrOutputParser()
        answer_chain = (
            {
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
                "context": itemgetter("context"),
            }
            | self._answer_prompt
            | structured_llm
        )

        retrieval_chain = (
            RunnablePassthrough.assign(chat_history=lambda _: chat_history)
            .assign(standalone_question=rewrite_chain)
            .assign(
                source_documents=RunnableLambda(
                    lambda data: self.vector_index.search(
                        session_id=session.session_id,
                        query=data["standalone_question"],
                        top_k=self.settings.top_k,
                    )
                )
            )
            .assign(context=RunnableLambda(lambda data: format_context(data["source_documents"])))
            .assign(structured_answer=answer_chain)
        )

        result = retrieval_chain.invoke({"question": question})
        structured_answer: StructuredAnswer = result["structured_answer"]
        source_documents: List[SourceDocument] = result["source_documents"]
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
            rewritten_question=result["standalone_question"].strip() or question,
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
