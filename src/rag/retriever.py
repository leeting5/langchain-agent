# -*- coding: utf-8 -*-
"""
src/rag/retriever.py -- 检索 + 重排序封装

将「向量召回候选 -> CrossEncoder 重排取 Top-N」收敛为一个 Retriever 类，
向调用方提供统一检索接口。检索策略（向量库、混合检索等）的调整集中
在此处，对外 API 保持不变。
"""
import os
import threading

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from config import settings
from .embeddings import get_embedding_model

_reranker = None
_reranker_lock = threading.Lock()


def _get_reranker():
    """全局唯一的 CrossEncoder 重排模型（懒加载单例）。"""
    global _reranker
    if _reranker is not None:
        return _reranker
    with _reranker_lock:
        if _reranker is None:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def _get_vectorstore():
    """从持久化目录加载向量库（单例）。"""
    from langchain_chroma import Chroma

    return Chroma(
        persist_directory=settings.vectorstore_dir,
        embedding_function=get_embedding_model(),
    )


class Retriever:
    """知识库检索器。

    用法：
        retriever = Retriever()
        docs = retriever.retrieve("2026赛季迈凯伦的1号车手是谁？")
        # docs[i].page_content / docs[i].metadata["source"]
    """

    def __init__(self):
        self._store = _get_vectorstore()

    def retrieve(self, query: str):
        """向量召回 + 重排，返回前 Top-N 文档。"""
        if not query.strip():
            return []
        # ① 向量召回更多候选（多捞）
        candidates = self._store.similarity_search(
            query, k=settings.retrieve_k
        )
        if not candidates:
            return []
        # ② 重排序打分
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = _get_reranker().predict(pairs)
        # ③ 按分数降序取前 Top-N
        top = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )[: settings.retrieve_top]
        return [doc for doc, _ in top]

    def retrieve_context(self, query: str) -> str:
        """便捷方法：直接返回拼接好的上下文文本（含来源标注）。"""
        docs = self.retrieve(query)
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源")
            parts.append(f"[来源: {source}]\n{doc.page_content}")
        return "\n\n".join(parts)


# 模块级便捷入口
def get_retriever():
    """返回全局唯一检索器（懒加载单例，复用同一向量库连接）。"""
    if not hasattr(get_retriever, "_inst"):
        get_retriever._inst = Retriever()
    return get_retriever._inst