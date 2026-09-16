# -*- coding: utf-8 -*-
"""
src/agents/tools.py -- Agent 工具定义

safe_calc 使用 ast 白名单安全求值：仅允许数字与 + - * / % // ** ( ) 等
数学语法，从根源上杜绝动态执行用户输入带来的任意代码执行风险。
工具通过 build_* 工厂函数装配，供 Agent 编排层复用。
"""
import ast
import operator
from datetime import datetime

from src.rag.retriever import get_retriever


# ---------------- 安全计算器 ----------------
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}


def safe_calc(expr: str) -> str:
    """安全执行数学表达式（ast 白名单，不支持任意代码执行）。

    只允许：数字、+ - * / % // ** ( ) 四则/取模/整除/幂运算。
    任何不在白名单内的语法都会返回错误信息，而不是执行。
    """
    if not isinstance(expr, str):
        return "请输入有效的数学表达式"
    expr = expr.strip()
    if not expr:
        return "请输入有效的数学表达式"
    if len(expr) > 200:
        return "表达式过长，请简化后重试"
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return "无法解析的表达式，请检查语法"

    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant):
            # 仅允许数字常量
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("仅支持数字运算")
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_OPS:
                raise ValueError("不支持的运算符")
            return _ALLOWED_OPS[op_type](
                eval_node(node.left), eval_node(node.right)
            )
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_OPS:
                raise ValueError("不支持的运算符")
            return _ALLOWED_OPS[op_type](eval_node(node.operand))
        raise ValueError("表达式包含不支持的语法")

    try:
        result = eval_node(tree)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)
    except ValueError as e:
        # ValueError 的消息本身就是给用户看的干净文案（如"表达式包含不支持的语法"）
        return str(e)
    except (ZeroDivisionError, TypeError) as e:
        return f"计算出错: {e}"


# ---------------- 时间工具 ----------------
def get_current_time(_s: str = "") -> str:
    """查询当前日期与时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------- 工具工厂 ----------------
def build_knowledge_tool(name="知识库检索", desc=None):
    """知识库检索工具：问题与知识库内容相关时调用。"""
    from langchain.tools import Tool

    def _search(query: str) -> str:
        try:
            docs = get_retriever().retrieve(query)
            if not docs:
                return "知识库中没有检索到相关内容。"
            parts = [d.page_content for d in docs]
            return "\n\n".join(parts)
        except Exception as e:  # noqa: BLE001
            return f"知识库检索失败: {e}"

    return Tool(
        name=name,
        func=_search,
        description=desc or (
            "当用户问题与知识库内容（如F1、特定资料）相关时，调用这个工具"
            "检索相关资料；反之则不调用。"
        ),
    )


def build_time_tool(name="当前时间查询", desc=None):
    """当前时间查询工具。"""
    from langchain.tools import Tool

    return Tool(
        name=name,
        func=get_current_time,
        description=desc or (
            "当用户询问当前日期、时间、今天是几号时调用这个工具，反之则不调用。"
        ),
    )


def build_calc_tool(name="计算器", desc=None):
    """安全计算器工具（ast 白名单）。"""
    from langchain.tools import Tool

    return Tool(
        name=name,
        func=safe_calc,
        description=desc or (
            "用于执行四则运算/取模/整除/幂运算的数学计算，输入如'23*47'的"
            "数学表达式。遇到数学计算题时使用。"
        ),
    )