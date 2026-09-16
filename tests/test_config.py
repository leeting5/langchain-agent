# -*- coding: utf-8 -*-
"""
tests/test_config.py -- 配置加载单测

验证 config.py 能正常加载、取值、且有合理默认值（离线可跑）。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings


class TestSettings:
    def test_llm_defaults(self):
        assert settings.llm_model == "deepseek-chat"
        assert settings.deepseek_base_url.startswith("https://")

    def test_retrieve_params_defaults(self):
        assert settings.retrieve_k == 10
        assert settings.retrieve_top == 5

    def test_chunk_params(self):
        assert settings.chunk_size > 0
        assert settings.chunk_overlap >= 0

    def test_paths_exist(self):
        # 项目根目录、data、src 应真实存在
        assert os.path.isdir(settings.root_dir)
        assert os.path.isdir(os.path.join(settings.root_dir, "data"))
        assert os.path.isdir(os.path.join(settings.root_dir, "src"))

    def test_env_example_present(self):
        assert os.path.exists(
            os.path.join(settings.root_dir, ".env.example")
        )