import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.tools import Tool
from langchain.agents import initialize_agent,AgentType
from sentence_transformers import CrossEncoder
from datetime import datetime

load_dotenv()

# 大模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

# （复用）加载知识库
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5",model_kwargs={"device": "cpu"})
vectorstore = Chroma(persist_directory="vector_db/vectorstore",embedding_function=embedding_model)
reranker = CrossEncoder("BAAI/bge-reranker-base")


# 定义一个“知识库检索”工具函数，agent可以自主决定调用它
def search_knowledge(query: str) -> str:
    """从知识库检索与query相关的文档内容并返回"""
    candidates = vectorstore.similarity_search(query, k=10)
    pairs = [(query, c.page_content) for c in candidates]
    scores = reranker.predict(pairs)
    top5 = sorted(zip(candidates,scores),
                  key=lambda x: x[1], reverse=True)[:5]
    return "\n\n".join(doc.page_content for doc, _ in top5)


def get_current_time(s: str) -> str:
    """查询当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 将函数包装成langchain工具，注册给agent
tools = [
    Tool(
        name="知识库检索",
        func=search_knowledge,
        description="当用户问题与知识库内容相关时，调用这个工具检索资料,反之则不调用",
    ),
    Tool(
        name="当前时间查询",
        func=get_current_time,
        description="当用户询问当前日期、时间、今天是几号时调用这个工具，反之则不调用"
    )
]


# 初始化agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#   verbose=True,  # 打印agent的思考过程
    verbose=False,
    handle_parsing_errors=True,
)


# 提问
print(agent.invoke("你好，介绍一下你自己"))
print(agent.invoke("2026赛季迈凯伦的1号车手是谁？"))
print(agent.invoke("当前日期是什么？"))

