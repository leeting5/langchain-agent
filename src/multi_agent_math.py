import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import Tool
load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)


def safe_calc(expr: str) -> str:
    """执行一个简单的数学表达式并返回结果"""
    # 只允许数字和四则运算符号，避免eval被执行危险代码
    try:
        allowed = set("0123456789 +-*/().")
        if not all(c in allowed for c in expr):
            return "包含不允许的字符，轻质输入数学表达式"
        return str(eval(expr))
    except Exception as e:
        return f"计算出错：{e}"


calc_tool = Tool(
    name="计算器",
    func=safe_calc,
    description="用于执行四则运算数学计算，输入如'23*47'这样的数学表达式。遇到数学计算题时使用。反之则不使用",
)


# 数学计算子agent：专门处理算术、逻辑计算，不使用知识库
math_agent = initialize_agent(
    llm=llm,
    tools=[calc_tool],
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, system_message="你是一个数学计算专家，只负责解答算术、数学逻辑、数值计算问题，其他类别的问题一律不回答。计算务必通过计算器工具，不要凭心算。",
    verbose=True,
    handle_parsing_errors=True,
)

print(math_agent.invoke("最小的质数是几？"))
print(math_agent.invoke("2*3等于几?"))
