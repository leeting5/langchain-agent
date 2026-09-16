# -*- coding: utf-8 -*-
"""
src/rag/generator.py -- LLM 统一封装

将 LLM 调用、查询改写、回答生成收敛为清晰的组件，并为所有大模型
调用统一提供请求超时、失败重试与输入长度校验。

组件：
- LLMClient      : 底层 LLM 客户端（带超时/重试），单例复用
- QueryRewriter  : 中文问题 -> 英文检索词
- AnswerGenerator: 依据检索上下文生成中文回答（含忠实性约束）
"""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from config import settings


class LLMClient:
    """统一 LLM 客户端：单例 + 超时 + 重试。"""

    _inst = None

    def __init__(self):
        settings.require_api_key()
        from langchain_openai import ChatOpenAI

        self.llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=settings.llm_temperature,
            timeout=settings.llm_request_timeout,
            max_retries=settings.llm_max_retries,
        )

    @classmethod
    def get(cls):
        """返回全局唯一 LLM 客户端。"""
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def chat(self, messages, **kwargs):
        """统一调用入口，带基础异常包装。messages 为 langchain Message 列表。"""
        try:
            resp = self.llm.invoke(messages, **kwargs)
            return resp.content
        except Exception as e:  # noqa: BLE001  — 统一转成可读错误
            raise RuntimeError(f"LLM 调用失败: {e}") from e


def _safe_input(text: str) -> str:
    """校验输入长度，防止超长输入造成异常消费。"""
    text = (text or "").strip()
    if len(text) > settings.max_input_chars:
        raise ValueError(
            f"输入过长（{len(text)} 字符），超过限制 {settings.max_input_chars}"
        )
    return text


class QueryRewriter:
    """中文问题 -> 适合检索英文文档的英文关键词/短句。"""

    SYSTEM = (
        "你是检索查询改写器。用户会用中文提问，请把它改写成适合"
        "在英文技术文档中搜索的英文关键词/短句，只输出英文，不要解释。"
    )

    def __init__(self):
        self._client = LLMClient.get()

    def rewrite(self, question: str) -> str:
        q = _safe_input(question)
        from langchain_core.prompts import ChatPromptTemplate

        tmpl = ChatPromptTemplate.from_messages(
            [("system", self.SYSTEM), ("human", "{q}")]
        )
        return self._client.chat(tmpl.invoke({"q": q}))


class AnswerGenerator:
    """依据检索上下文生成中文回答（忠实约束）。"""

    SYSTEM = (
        "你是知识库问答助手，只用中文、忠实依据提供的资料回答，"
        "资料里没有的就说暂时无法回答这个问题。回答时尽量注明信息来源文档。"
    )

    def __init__(self):
        self._client = LLMClient.get()

    def answer(self, context: str, question: str) -> str:
        q = _safe_input(question)
        from langchain_core.prompts import ChatPromptTemplate

        tmpl = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM),
                ("human", "资料：{context}\n\n问题：{question}"),
            ]
        )
        return self._client.chat(
            tmpl.invoke({"context": context, "question": q})
        )