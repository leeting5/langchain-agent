# multi_agent_full.py —— 完整多Agent团队：调度Agent + 3个子Agent
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

# ---------- 共用：知识库 ----------
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5", model_kwargs={"device": "cpu"})
vectorstore = Chroma(persist_directory="vector_db/vectorstore", embedding_function=embedding_model)
reranker = CrossEncoder("BAAI/bge-reranker-base")

# ---------- 子Agent 1：知识库问答 ----------
def search_knowledge(query: str) -> str:
    candidates = vectorstore.similarity_search(query, k=8)
    pairs = [(query, c.page_content) for c in candidates]
    scores = reranker.predict(pairs)
    top3 = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:3]
    return "\n\n".join(doc.page_content for doc, _ in top3)

kb_agent = initialize_agent(
    tools=[Tool(name="知识库检索", func=search_knowledge,
                description="当问题涉及知识库内容(如F1、特定资料)时使用，检索相关资料")],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    system_message="你是一个知识库问答专家，只回答与知识库资料相关的问题，回答须引用检索到的内容，资料没有的就说不知道。",
    verbose=False,
    handle_parsing_errors=True,
)

# ---------- 子Agent 2：数学计算 ----------
def safe_calc(expr: str) -> str:
    allowed = set("0123456789 +-*/().")
    if not all(c in allowed for c in expr):
        return "包含不允许的字符，请只输入数学表达式"
    try:
        return str(eval(expr))
    except Exception as e:
        return f"计算出错: {e}"

math_agent = initialize_agent(
    tools=[Tool(name="计算器", func=safe_calc, description="执行四则运算，输入如'23*47'")],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    system_message="你是数学计算专家，只负责算术和数学问题，计算务必用计算器工具，不凭心算。",
    verbose=False,
    handle_parsing_errors=True,
)

# ---------- 子Agent 3：时间查询 ----------
from datetime import datetime
def get_current_time(s: str) -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

time_agent = initialize_agent(
    tools=[Tool(name="当前时间", func=get_current_time, description="用户询问当前日期/时间时使用")],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    system_message="你只负责查询和回答当前日期时间类问题。",
    verbose=False,
    handle_parsing_errors=True,
)

# ---------- 调度Agent：手里挂着三个子Agent作为工具 ----------
Tools = [
    Tool(name="知识库专家", func=kb_agent.run,
         description="当问题涉及具体资料内容(如F1车手、文档知识)时交给这个专家"),
    Tool(name="数学专家", func=math_agent.run,
         description="当问题涉及算术、数学计算、数值运算时交给这个专家"),
    Tool(name="时间专家", func=time_agent.run,
         description="当问题询问当前日期、时间时交给这个专家"),
]

dispatcher = initialize_agent(
    tools=Tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    system_message="你是任务调度主管。收到用户问题后，先判断属于哪类，选择最合适的专家处理；如属于闲聊无需专家，可直接回答。",
    verbose=True,   # 打开，你能看到调度Agent怎么分配任务的
    handle_parsing_errors=True,
)

# ---------- 测试 ----------
print(dispatcher.invoke("2026赛季迈凯伦的1号车手是谁？"))
print(dispatcher.invoke("25的平方是多少？"))
print(dispatcher.invoke("今天多少号？"))