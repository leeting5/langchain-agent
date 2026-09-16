# -*- coding: utf-8 -*-
"""
src/rag/kb.py -- 知识库构建（参数化）

将「文档读取 -> 切分 -> 向量化入库」流程参数化，支持：
- 自定义数据目录（默认 data/cleaned）
- 自定义切分参数（chunk_size / overlap）
- 重建前可选清空旧向量库
方便切换不同主题的知识库，而无需改动流程代码。

用法：
    python -m src.rag.kb --data data/raw --rebuild
"""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse

from config import settings
from .embeddings import get_embedding_model


def build(data_dir: str = None, rebuild: bool = True,
          chunk_size: int = None, chunk_overlap: int = None) -> int:
    """读取 data_dir 下所有 .txt，切分并写入向量库。返回切分块数。

    Args:
        data_dir: 文档目录，默认 settings.cleaned_data_dir
        rebuild:  是否重建前清空旧向量库。增量入库时传 False。
        chunk_size / chunk_overlap: 覆盖 settings 的切分参数。
    """
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma

    data_dir = data_dir or settings.cleaned_data_dir
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"文档目录不存在: {data_dir}。请先放置 .txt 文档。"
        )

    files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith(".txt") and not f.startswith(".")
    )
    if not files:
        raise FileNotFoundError(f"目录 {data_dir} 下没有 .txt 文件")

    print(f"找到 {len(files)} 篇文档：{files}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    all_chunks = []
    for name in files:
        loader = TextLoader(
            os.path.join(data_dir, name), encoding="utf-8"
        )
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = name  # 记录来源文档，便于溯源
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
        print(f"[{name}] 切分 {len(chunks)} 块")

    # 重建前清空旧向量库（避免旧内容残留串味）
    if rebuild and os.path.exists(settings.vectorstore_dir):
        import shutil
        shutil.rmtree(settings.vectorstore_dir)
        print("已清空旧向量库")

    _ = Chroma.from_documents(
        documents=all_chunks,
        embedding=get_embedding_model(),
        persist_directory=settings.vectorstore_dir,
    )
    print(f"知识库构建完成，共 {len(all_chunks)} 个语义块。")
    return len(all_chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 RAG 知识库")
    parser.add_argument("--data", default=None, help="文档目录")
    parser.add_argument("--no-rebuild", action="store_true",
                        help="不清空旧向量库（增量追加）")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    args = parser.parse_args()

    build(
        data_dir=args.data,
        rebuild=not args.no_rebuild,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )