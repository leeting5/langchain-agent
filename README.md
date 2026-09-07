# LangChain Agent 知识库问答系统

基于 **LangChain + FastAPI** 构建的本地知识库问答 Web 应用，支持中文提问、检索增强生成（RAG）、多文档知识库、重排序与 query 改写优化。

## 项目简介

围绕 LangChain Agent 开发官方文档构建本地知识库，提供一套完整 RAG 问答链路：文档清洗 -> 向量化入库 -> 语义检索 -> 重排序 -> 大模型生成回答，并封装为可交互的 Web 应用（FastAPI 后端 + 原生 HTML 前端），支持 Docker 容器化部署。

## 核心能力

- **多文档知识库**：遍历清洗后的 4 篇 LangChain 官方文档（Agents / Context-engineering / Models / Multi-agent），按篇切分入库，共 475 个语义块，每条带来源元数据便于溯源。
- **中文友好问答**：内置中文->英文 query 改写，缓解中英文检索鸿沟，中文提问也能准确命中英文文档。
- **本地语义检索 + 重排**：BGE 中文向量模型召回 + reranker 重排取 Top5，提高相关性。
- **智能保存**：LLM 判断用户保存意图，命中时把回答保存为本地文件并提醒。
- **Web 交互界面**：FastAPI 接口 + 原生单页前端，浏览器直接提问。
- **容器化部署**：Dockerfile 打包一键运行。

## 技术栈

| 模块 | 技术 |
|------|------|
| 大模型 | DeepSeek（deepseek-chat） |
| Embedding | BAAI/bge-small-zh-v1.5 |
| 重排 | BAAI/bge-reranker-base |
| RAG 框架 | LangChain |
| 向量存储 | Chroma |
| Web 框架 | FastAPI + Uvicorn |
| 前端 | 原生 HTML + JavaScript |
| 部署 | Docker |

## 目录结构

```
aiAgent/
├── data/cleaned/          # 清洗后文档（入库源）
├── src/
│   ├── clean_docs.py      # 文档清洗
│   ├── build_vector_db.py # 多文档切分入库
│   ├── app.py             # FastAPI 后端 + query改写 + 智能保存
│   ├── rag_query.py       # 命令行 RAG 问答
├── static/index.html      # Web 前端
├── answers/               # 回答保存目录（运行时生成）
├── vector_db/             # 向量库（运行时生成）
├── Dockerfile
└── requirements.txt
```

## 快速开始

```bash
pip install -r requirements.txt
```

在根目录创建 `.env` 并填入：
```
DEEPSEEK_API_KEY=你的_key
```

启动 Web 应用：
```bash
uvicorn src.app:app --reload
```
浏览器访问 `http://localhost:8000` 即可提问。

重建知识库（新增文档后执行）：
```bash
python src/build_vector_db.py
```

## Docker 部署

```bash
docker build -t agent-kb .
docker run --rm -v "%CD%\vector_db:/app/vector_db" --env-file .env agent-kb
```

## 数据来源

知识库内容来自 LangChain 官方文档，仅用于本地学习与演示。