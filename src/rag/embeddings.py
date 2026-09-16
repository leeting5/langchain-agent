# -*- coding: utf-8 -*-
"""
src/rag/embeddings.py -- Embedding 模型封装（单例）

使用线程安全的懒加载单例，保证同一进程内仅加载一次 embedding
模型，避免重复初始化的内存与时间开销，供各处复用。
"""
import os
import threading
from functools import lru_cache

from config import settings


# 生成 embedding 前关闭 tokenizer 的并行警告
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_lock = threading.Lock()
_embedding = None


def get_embedding_model():
    """返回全局唯一的 embedding 实例（线程安全懒加载单例）。"""
    global _embedding
    if _embedding is not None:
        return _embedding
    with _lock:
        if _embedding is None:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            _embedding = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": settings.embedding_device},
            )
    return _embedding