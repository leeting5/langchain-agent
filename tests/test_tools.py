# -*- coding: utf-8 -*-
"""
tests/test_tools.py -- 安全计算器 & 时间/知识库工具的单测

重点验证安全加固：ast 求值器不再有 eval 注入风险。
这些测试不依赖 API Key / 向量库 / 网络，可离线运行。
"""
import sys
import os

# 确保能 import 项目根目录的 config.py 与 src 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.tools import safe_calc, get_current_time


class TestSafeCalc:
    def test_basic_arithmetic(self):
        assert safe_calc("2 * 3") == "6"
        assert safe_calc("23 * 47") == "1081"
        assert safe_calc("10 / 2") == "5"

    def test_float_result(self):
        assert safe_calc("7 / 2") == "3.5"

    def test_priority_and_paren(self):
        assert safe_calc("(1 + 2) * 3") == "9"
        assert safe_calc("2 + 3 * 4") == "14"

    def test_power(self):
        assert safe_calc("2 ** 10") == "1024"

    def test_unary_neg(self):
        assert safe_calc("-5 + 3") == "-2"

    def test_invalid_input_rejected(self):
        # 关键字 / 函数调用 / 属性访问应被拒绝，绝不允许执行代码
        assert safe_calc("__import__('os')") == "表达式包含不支持的语法"
        assert safe_calc("lambda: 1") == "表达式包含不支持的语法"
        assert safe_calc("[].__class__") == "表达式包含不支持的语法"

    def test_empty_and_whitespace(self):
        assert "请输入" in safe_calc("")
        assert "请输入" in safe_calc("   ")

    def test_overflow_limit(self):
        assert "过长" in safe_calc("1" * 300)


class TestTimeTool:
    def test_time_format(self):
        val = get_current_time()
        # 形如 2026-09-15 22:00:00
        assert len(val.split(":")) == 3
        assert "-" in val


class TestNoEvalInSource:
    def test_no_raw_eval_in_tools(self):
        """安全加固核查：src/agents/tools.py 不应再出现裸 eval(。"""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src", "agents", "tools.py",
        )
        with open(path, encoding="utf-8") as f:
            src = f.read()
        # allow pattern: eval_node (ast). 裸的 eval( 视为违规
        assert "= eval(" not in src