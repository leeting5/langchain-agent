# -*- coding: utf-8 -*-
"""
src/api/app.py -- FastAPI 知识库问答服务

复用 src/rag 组件，提供统一 REST 接口，包含：
1. 统一异常处理：出错返回友好 JSON
2. 结构化日志埋点
3. 输入长度校验（防超长输入）
4. 智能保存：识别保存意图并落盘到 answers/
"""
import logging
import os
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi import Request
from pydantic import BaseModel

from config import settings
from src.rag.retriever import get_retriever
from src.rag.generator import QueryRewriter, AnswerGenerator, LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("f1-rag-api")

app = FastAPI(title="F1 知识库问答", version="1.0.0")

# 启动时预加载核心组件（常驻内存，单例）
_retriever = None
_rewriter = None
_generator = None


@app.on_event("startup")
async def startup():
    global _retriever, _rewriter, _generator
    logger.info("正在加载向量库与模型…")
    _retriever = get_retriever()
    _rewriter = QueryRewriter()
    _generator = AnswerGenerator()
    settings.require_api_key()
    logger.info("初始化完成，服务就绪。")


# ---------- 统一异常处理 ----------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("未处理异常: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": f"服务内部错误，请稍后重试。({exc})"},
    )


# ---------- 智能保存 ----------
def _save_answer(answer: str) -> str:
    os.makedirs(settings.answers_dir, exist_ok=True)
    filename = os.path.join(
        settings.answers_dir,
        f"qa_{time.strftime('%Y%m%d_%H%M%S')}.txt",
    )
    with open(filename, "w", encoding="utf-8") as f:
        f.write(answer)
    return filename


class SaveIntentDetector:
    """用 LLM 判断用户是否有「把回答保存到本地」的意图。"""

    PROMPT = (
        "你将收到一句用户的提问。请判断这句话里是否含有"
        "'把回答/答案/结果 保存到本地电脑/文件夹' 的明确意图。\n"
        "只回答 true 或 false，不要输出任何其他内容。\n\n"
        "用户提问：{q}"
    )

    def detect(self, user_input: str) -> bool:
        try:
            resp = LLMClient.get().chat(
                [{"role": "user",
                  "content": self.PROMPT.format(q=user_input)}],
            )
            return resp.strip().lower() == "true"
        except Exception as e:  # noqa: BLE001  判断失败视为无保存意图
            logger.warning("保存意图判断失败: %s", e)
            return False


_save_detector = SaveIntentDetector()


def _answer(question: str) -> str:
    """检索 - 改写 - 生成 的核心问答流程。"""
    # ① 中文 -> 英文检索词
    search_query = _rewriter.rewrite(question)
    logger.info("改写后的检索词: %s", search_query)
    # ② 检索 + 重排取上下文（带来源标注）
    context = _retriever.retrieve_context(search_query)
    # ③ 生成
    return _generator.answer(context, question)


# ---------- 接口 ----------
@app.get("/")
def index():
    return FileResponse(
        os.path.join(settings.static_dir, "index.html")
    )


class QueryIn(BaseModel):
    """POST /query 的请求体（JSON）：{ "question": "..." }。"""
    question: str


@app.post("/query")
def query(payload: QueryIn):
    q = (payload.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="请输入问题")
    if len(q) > settings.max_input_chars:
        raise HTTPException(
            status_code=400,
            detail=f"输入过长，请控制在 {settings.max_input_chars} 字符内",
        )
    try:
        answer = _answer(q)
        saved = False
        if _save_detector.detect(q):
            _save_answer(answer)
            saved = True
        return {"answer": answer, "saved": saved}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        logger.error("问答失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"问答失败: {e}")