# -*- coding: utf-8 -*-
"""
src/cli_agent.py -- 命令行问答入口

统一命令行入口，通过 --mode 切换运行方式：

用法：
    # RAG 问答（检索-改写-生成）
    python -m src.cli_agent --mode rag --q "2026赛季迈凯伦的1号车手是谁？"
    # 单 Agent（知识库检索 + 时间工具，ReAct）
    python -m src.cli_agent --mode agent --q "今天日期是什么？"
    # 多 Agent 团队（调度 + 知识库/数学/时间专家）
    python -m src.cli_agent --mode multi --q "25的平方是多少？"
"""
import argparse
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from config import settings


def run_rag(question: str):
    from src.rag.retriever import get_retriever
    from src.rag.generator import QueryRewriter, AnswerGenerator

    settings.require_api_key()
    rewriter = QueryRewriter()
    retriever = get_retriever()
    generator = AnswerGenerator()

    search_query = rewriter.rewrite(question)
    print("改写后的检索词:", search_query)
    context = retriever.retrieve_context(search_query)
    answer = generator.answer(context, question)
    print("\nAI回答:\n", answer)


def run_agent(question: str):
    from src.agents.orchestrator import build_single_rag_agent

    agent = build_single_rag_agent(verbose=False)
    print(agent.invoke(question))


def run_multi(question: str):
    from src.agents.orchestrator import build_multi_agent_team

    dispatcher = build_multi_agent_team(verbose=False)
    print(dispatcher.invoke(question))


def main():
    parser = argparse.ArgumentParser(description="知识库问答 / Agent 命令行入口")
    parser.add_argument("--mode", choices=["rag", "agent", "multi"],
                        default="rag", help="运行模式")
    parser.add_argument("--q", required=True, help="用户问题")
    args = parser.parse_args()

    q = args.q.strip()
    if not q:
        print("请输入问题 (-q)")
        return

    if args.mode == "rag":
        run_rag(q)
    elif args.mode == "agent":
        run_agent(q)
    else:
        run_multi(q)


if __name__ == "__main__":
    main()