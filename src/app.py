# 知识库问答接口
import os
import requests
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
load_dotenv()

# 启动时加载一次（模型常驻内存）
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5", model_kwargs={"device": "cpu"}
)
vectorstore = Chroma(persist_directory="vector_db/vectorstore",embedding_function=embedding_model)
reranker = CrossEncoder("BAAI/bge-reranker-base")
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)


# 查询函数：问题->回答
def answer(question: str) -> str:
    # ① 关键新增：中文问题 -> 英文检索词（拉近与英文文档的语义距离）
    rewrite_template = ChatPromptTemplate.from_messages([
        ("system", "你是检索查询改写器。用户会用中文提问，请把它改写成适合在英文技术文档中搜索的英文关键词/短句，只输出英文，不要解释。"),
        ("human", "{q}"),
    ])
    search_query = llm.invoke(rewrite_template.invoke({"q": question})).content
    print("改写后的检索词:", search_query)

    # ② 用英文检索词去向量库搜索
    candidates = vectorstore.similarity_search(search_query, k=10)
    # ③ 重排打分取前5
    pairs = [(search_query, c.page_content) for c in candidates]
    scores = reranker.predict(pairs)
    top5 = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:5]
    context = "\n\n".join(doc.page_content for doc, _ in top5)
    # ④ LLM 用中文、忠实依据资料回答
    template = ChatPromptTemplate.from_messages([
        ("system", "你是知识库问答助手，只用中文、忠实依据提供的资料回答，资料里没有的就说暂时无法回答这个问题"),
        ("human", "资料：{context}\n\n问题：{question}"),
    ])
    resp = llm.invoke(template.invoke({"context": context, "question": question}))
    return resp.content


# 接口：浏览器访问 /query?q=用户输入的问题
app = FastAPI()


# 在 query 函数里，拿到 answer 后调用它
def save_answer(answer: str):
    os.makedirs("answers", exist_ok=True)
    filename = f"answers/qa_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(answer)
    return filename


def wants_to_save(user_input: str) -> bool:
    """让 LLM 判断用户是否有保存到本地的意图"""
    prompt = (
        "你将收到一句用户的提问。请判断这句话里是否含有"
        "'把回答/答案/结果 保存到本地电脑/文件夹' 的明确意图。\n"
        "只回答 true 或 false，不要输出任何其他内容。\n\n"
        f"用户提问：{user_input}"
    )
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 5,
            "temperature": 0,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip().lower() == "true"




@app.get("/query")
def query(q: str):
    reply = answer(q)
    if wants_to_save(q):  # 用 LLM 判断
        filename = save_answer(reply)
        reply += f"\n\n✅ 已按要求将回答保存到本地：{filename}"
    return {"answer": reply}


@app.get("/")
def home():
    return FileResponse("static/index.html")


