# -*- coding: utf-8 -*-
"""
src/agents/orchestrator.py -- Agent 编排入口

提供两类可复用入口：
- SingleAgentRAG : 单一 ReAct Agent，带知识库检索 + 时间工具
- MultiAgentTeam : 调度 Agent + 知识库/数学/时间三个专家

Agent 编排逻辑集中在此文件，与底层 rag/、tools 模块解耦，
便于独立扩展编排策略。
"""
import logging
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from langchain.agents import initialize_agent, AgentType

logger = logging.getLogger(__name__)


def _build_llm():
    from src.rag.generator import LLMClient
    return LLMClient.get().llm


def build_single_rag_agent(verbose: bool = False):
    """构建单 Agent：知识库检索 + 当前时间。"""
    from .tools import build_knowledge_tool, build_time_tool

    llm = _build_llm()
    tools = [
        build_knowledge_tool(),
        build_time_tool(),
    ]
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=verbose,
        handle_parsing_errors=True,
    )


def build_multi_agent_team(verbose: bool = False):
    """构建多 Agent 团队：调度 Agent + 三个专家子 Agent。

    子 Agent 作为「工具」挂到调度 Agent 上，由其判断任务归属。
    """
    from langchain.agents import initialize_agent, AgentType
    from langchain.tools import Tool
    from .tools import build_knowledge_tool, build_calc_tool, build_time_tool

    llm = _build_llm()

    # 子 Agent 1：知识库专家
    kb_agent = initialize_agent(
        tools=[build_knowledge_tool(name="知识库检索",
                desc="当问题涉及知识库内容(如F1)时使用，检索相关资料")],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        system_message=(
            "你是一个知识库问答专家，只回答与知识库资料相关的问题，"
            "回答须引用检索到的内容，资料没有的就说不知道。"
        ),
        verbose=verbose,
        handle_parsing_errors=True,
    )
    # 子 Agent 2：数学专家
    math_agent = initialize_agent(
        tools=[build_calc_tool(name="计算器",
                desc="执行四则运算，输入如'23*47'")],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        system_message=(
            "你是数学计算专家，只负责算术和数学问题，"
            "计算务必用计算器工具，不凭心算。"
        ),
        verbose=verbose,
        handle_parsing_errors=True,
    )
    # 子 Agent 3：时间专家
    time_agent = initialize_agent(
        tools=[build_time_tool(name="当前时间",
                desc="用户询问当前日期/时间时使用")],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        system_message="你只负责查询和回答当前日期时间类问题。",
        verbose=verbose,
        handle_parsing_errors=True,
    )

    # 调度 Agent：把三个专家作为工具挂载
    dispatcher = initialize_agent(
        tools=[
            Tool(name="知识库专家", func=kb_agent.run,
                 description=(
                     "当问题涉及具体资料内容(如F1车手、文档知识)时交给这个专家"
                 )),
            Tool(name="数学专家", func=math_agent.run,
                 description=(
                     "当问题涉及算术、数学计算、数值运算时交给这个专家"
                 )),
            Tool(name="时间专家", func=time_agent.run,
                 description="当问题询问当前日期、时间时交给这个专家"),
        ],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        system_message=(
            "你是任务调度主管。收到用户问题后，先判断属于哪类，"
            "选择最合适的专家处理；如属于闲聊无需专家，可直接回答。"
        ),
        verbose=verbose,
        handle_parsing_errors=True,
    )
    return dispatcher