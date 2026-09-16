# -*- coding: utf-8 -*-
"""
config.py -- 集中配置管理

用 python-dotenv 读取 .env 中的密钥，配合环境变量 / 默认值统一管理
模型名、检索参数、服务端口等配置。所有模块从这里取配置，避免散落硬编码。

用法：
    from config import settings
    settings.deepseek_api_key
"""
import os
from dotenv import load_dotenv

# 从项目根目录 .env 加载（模块被 import 时，__file__ 指向 src/../.env）
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


class Settings:
    """集中配置类。新增配置项只需在 __init__ 里加属性并给出默认值。

    优先级：已存在的环境变量 > .env 文件 > 代码默认值。
    """

    def __init__(self):
        # ---- LLM ----
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.deepseek_base_url = os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )
        self.llm_model = os.getenv("LLM_MODEL", "deepseek-chat")
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        self.llm_request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "60"))
        self.llm_max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))

        # ---- Embedding / 向量库 ----
        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
        )
        self.embedding_device = os.getenv("EMBEDDING_DEVICE", "cpu")
        # 向量库持久化目录（相对项目根）
        self.vectorstore_dir = os.getenv(
            "VECTORSTORE_DIR", "vector_db/vectorstore"
        )

        # ---- 重排序 ----
        self.reranker_model = os.getenv(
            "RERANKER_MODEL", "BAAI/bge-reranker-base"
        )

        # ---- 检索参数 ----
        self.retrieve_k = int(os.getenv("RETRIEVE_K", "10"))   # 初检候选数
        self.retrieve_top = int(os.getenv("RETRIEVE_TOP", "5"))  # 重排后取前 N
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))

        # ---- 数据目录 ----
        root = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = root
        self.data_dir = os.path.join(root, "data")
        self.raw_data_dir = os.path.join(root, "data", "raw")
        self.cleaned_data_dir = os.path.join(root, "data", "cleaned")
        self.answers_dir = os.path.join(root, "answers")
        self.static_dir = os.path.join(root, "static")

        # ---- 输入限制 ----
        self.max_input_chars = int(os.getenv("MAX_INPUT_CHARS", "2000"))

    def require_api_key(self):
        """缺少 API Key 时给出明确提示，避免运行期神秘报错。"""
        if not self.deepseek_api_key:
            raise RuntimeError(
                "未配置 DEEPSEEK_API_KEY。请在项目根目录 .env 中填写："
                "参考 .env.example"
            )


settings = Settings()