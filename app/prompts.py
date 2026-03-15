from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def build_question_rewrite_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是检索前的问题改写助手。请结合对话历史，把当前用户问题改写成一个独立、清晰、可检索的问题。"
                "只补全代词指代和省略信息，不要回答问题，不要添加对话中未出现的事实。",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "当前问题：{question}"),
        ]
    )


def build_answer_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是企业知识库问答助手。你必须严格基于提供的上下文回答，不能使用上下文之外的常识、训练知识或猜测。"
                "如果上下文不足以支持答案，直接回答“我不知道”。"
                "如果回答中包含结论，请在相关句子后附加 [source_id] 形式的引用标记。"
                "每个结论都必须绑定到上下文中的 source_id；若回答“我不知道”，citations 返回空数组。",
            ),
            MessagesPlaceholder("chat_history"),
            (
                "human",
                "用户问题：{question}\n\n"
                "可用上下文：\n{context}\n\n"
                "请输出结构化结果，且 citations 里的 source_id 必须原样引用上下文中的 source_id。"
                "answer 字段中的引用标记也必须使用原样 source_id。",
            ),
        ]
    )
