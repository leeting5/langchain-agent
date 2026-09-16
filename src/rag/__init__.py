# -*- coding: utf-8 -*-
"""src.rag -- 检索增强生成（RAG）包。"""
from .embeddings import get_embedding_model
from .retriever import Retriever
from .generator import LLMClient, QueryRewriter, AnswerGenerator

__all__ = [
    "get_embedding_model",
    "Retriever",
    "LLMClient",
    "QueryRewriter",
    "AnswerGenerator",
]