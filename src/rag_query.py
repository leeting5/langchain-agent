from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

# 加载向量库+向量模型
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5",model_kwargs={"device": "cpu"})
vectorstore = Chroma(persist_directory="vector_db/vectorstore", embedding_function=embedding_model)
reranker = CrossEncoder("BAAI/bge-reranker-base")

# 使用deepseek大模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

# 检索+重排，得到最相关的3段文字
question = "2026赛季迈凯伦的1号车手是谁？"
candidates = vectorstore.similarity_search(question, k=10)
pairs = [(question, c.page_content) for c in candidates]
scores = reranker.predict(pairs)
top5 = sorted(zip(candidates, scores), key=lambda x: x[1],
reverse=True)[:5]

# 把所有命中的文字拼成“参考资料”
context = "\n\n".join(doc.page_content for doc, _ in top5)
print("已检索到最相关的5段文字：\n", context[:250], "\n...\n")

# 构造提示词+生成回答
template = ChatPromptTemplate.from_messages([
    ("system", "你是知识库问答助手，只依据提供的资料回答，资料里没有的就说不知道"),
    ("human", "资料：{context}\n\n问题：{question}"),
])

resp = llm.invoke(template.invoke({"context": context, "question": question}))
print('AI回答:',resp.content)

