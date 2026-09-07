# -*- coding: utf-8 -*-
"""
build_vector_db.py (多文件版)
读取 data/cleaned/ 下所有 LangChain 文档 -> 切块 -> 向量化 -> 存入 Chroma 向量库

改造点（对比阶段2的单文档版）：
1. 遍历 data/cleaned/ 下的所有 LangChain_*.txt，不再只处理单个文件
2. 重建前自动清空旧向量库，避免旧 F1 向量残留与新文档混在一起
3. 每块带 source 元数据（来源文档名），便于 Web 端"答案出自哪篇"溯源

用法：在项目根目录执行  python src\build_vector_db.py
输出：vector_db/vectorstore
"""
import os
import shutil
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"   # 避免模型加载时的多线程警告

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# BGE 中文向量模型：把文字变成数字向量
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
)

# 数据目录 = 清洗后的文档位置；持久化目录沿用项目统一路径
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "cleaned"
PERSIST_DIR = "vector_db/vectorstore"

# 1) 收集所有清洗后的文档
files = sorted(DATA_DIR.glob("LangChain_*.txt"))
print(f"找到 {len(files)} 篇文档：{[f.name for f in files]}")

# 2) 读取每篇并切块（沿用阶段2参数：chunk_size=500, overlap=50）
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_chunks = []
for f in files:
    loader = TextLoader(str(f), encoding="utf-8")
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = f.name          # 记录来源文档，便于溯源
    chunks = splitter.split_documents(docs)
    all_chunks.extend(chunks)
    print(f"[{f.name}] 切分 {len(chunks)} 块")
print(f"全部切分块数：{len(all_chunks)}")

# 3) 重建前清空旧向量库（旧 F1 向量不能残留，否则检索会串味）
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)
    print("已清空旧向量库（含之前的 F1 数据）")

# 4) 向量化并存入 Chroma
vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embedding_model,
    persist_directory=PERSIST_DIR,
)
print("知识库构建完成啦！向量已存储")