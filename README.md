# LangChain Business RAG QA System

这是一个放在 `RAG_SYSTEM` 目录下的完整 RAG 问答系统，面向中文业务知识库场景设计，支持文档导入、多轮对话、代词指代理解、严格基于上下文回答、结构化输出以及引用溯源。

## 项目亮点

- 使用 LangChain LCEL 组织完整链路：问题改写、检索、生成全部以 Runnable 方式串联。
- 使用 `ChatPromptTemplate.from_messages` 设计严格约束 Prompt，要求模型只能依据上下文回答，不知道就明确回答“我不知道”。
- 使用 Pydantic 定义结构化输出，模型返回 `answer`、`grounded`、`citations`。
- 使用 `ConversationBufferMemory` 管理多轮对话历史，支持追问中的代词指代消解。
- 返回 `source_documents`，并在界面中展示“来自哪个文档哪一段”的引用结果。
- 提供 FastAPI Web 界面，支持上传文档、按路径导入文档、加载内置业务样例知识库。
- 模块化拆分了加载、切块、向量化、检索、生成、会话管理和 Web 层。

## 目录结构

```text
RAG_SYSTEM/
├── app/
│   ├── config.py
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── knowledge_base.py
│   ├── models.py
│   ├── prompts.py
│   ├── rag_chain.py
│   ├── server.py
│   ├── session_manager.py
│   ├── splitter.py
│   └── vector_store.py
├── data/
│   ├── sample_docs/
│   └── uploads/
├── static/
├── templates/
├── main.py
└── requirements.txt
```

## 技术选型说明

- `FastAPI`
  用来承载一个轻量 Web 界面和 JSON API，足够简单，也方便后续继续扩展成企业内部服务。
- `LangChain LCEL`
  用于把“问题改写 -> 向量检索 -> 严格问答 -> 结构化解析”组合成清晰可维护的链。
- `ConversationBufferMemory`
  保存多轮历史消息，让追问如“那它需要谁确认”能够先被改写为独立问题，再去检索。
- `ChatOpenAI`
  负责对问题进行历史改写和基于上下文生成答案，模型名称与 `OPENAI_MODEL` 通过环境变量配置。
- `SentenceTransformers + ChromaDB`
  使用本地向量模型 `paraphrase-multilingual-MiniLM-L12-v2` 进行中文向量化，检索结果落在本地 Chroma 持久化目录中。
- `Pydantic`
  约束模型结构化输出，便于前端稳定渲染引用、回答和检索结果。

## 运行方式

### 1. 安装依赖

```bash
cd /Users/wilson.zhang/Desktop/agent_engineering_lessons/RAG_SYSTEM
python3 -m pip install -r requirements.txt
```

### 2. 配置环境变量

至少需要设置：

```bash
export OPENAI_API_KEY="你的 API Key"
```

可选项：

```bash
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_BASE_URL=""
export RAG_TOP_K="4"
export RAG_CHUNK_SIZE="320"
export RAG_CHUNK_OVERLAP="60"
```

如果你使用的是 OpenAI 兼容接口，也可以同时设置 `OPENAI_BASE_URL`。

### 3. 启动服务

```bash
cd /Users/wilson.zhang/Desktop/agent_engineering_lessons/RAG_SYSTEM
python3 main.py
```

浏览器打开：

```text
http://127.0.0.1:8000
```

### 4. 推荐体验路径

1. 点击“加载内置样例知识库”。
2. 先问“退款金额高于 200 元怎么办？”
3. 再追问“那它需要谁二次确认？”
4. 再问“夜间无人值守时系统如何处理会话？”

你会看到：

- 系统会返回检索改写后的问题。
- 回答会附带引用来源。
- `source_documents` 会展示实际命中的文档片段及相似度。

## API 说明

- `POST /api/session`
  创建一个会话。
- `POST /api/documents/sample`
  把内置样例文档写入当前会话知识库。
- `POST /api/documents/path`
  通过文件路径导入文档。
- `POST /api/documents/upload`
  通过 Web 上传文档。
- `POST /api/chat`
  发起问答，返回结构化答案和 `source_documents`。
- `POST /api/session/reset`
  清空会话历史，或者连同知识库一起重置。

## 关键实现说明

### LCEL 链路

系统在 [`app/rag_chain.py`](./app/rag_chain.py) 中实现了完整 LCEL 流程：

1. 用 `ConversationBufferMemory` 读取 `chat_history`
2. 用 `ChatPromptTemplate.from_messages` 改写追问
3. 用本地向量检索召回 `source_documents`
4. 把上下文格式化为带 `source_id` 的证据块
5. 再用严格问答 Prompt + Pydantic structured output 返回结果

### 严格回答策略

问答 Prompt 中明确约束：

- 只能基于上下文回答
- 禁止补充上下文外知识
- 如果证据不足，直接回答“我不知道”
- citations 的 `source_id` 必须来自检索上下文

### 引用溯源

每个检索片段都会生成统一 `source_id`，格式为：

```text
文档名::第N段
```

返回结果中包含：

- `citations`: 结构化引用信息
- `source_documents`: 原始命中文档片段、文档路径、段落号、相似度

## 已支持的文档类型

- `.txt`
- `.md`
- `.pdf`
- `.docx`

## GitHub 推送

我已经把 `RAG_SYSTEM` 初始化成独立 Git 仓库，并提交到本地 `main` 分支。

- 当前本地提交：`31b7a91`
- 当前状态：工作区干净，可直接挂远程后推送

如果你已经有 GitHub 仓库地址：

```bash
cd /Users/wilson.zhang/Desktop/agent_engineering_lessons/RAG_SYSTEM
git remote add origin <你的仓库地址>
git push -u origin main
```

这台机器当前没有安装 `gh`，所以我还不能直接替你创建 GitHub 仓库。只要你给我一个可推送的远程仓库地址，或者安装并登录 `gh`，我就可以继续帮你完成最后一步 push。
