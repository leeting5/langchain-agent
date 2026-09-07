import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# 1. 加载向量库
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5", model_kwargs={"device": "cpu"})
vectorstore = Chroma(persist_directory="vector_db/vectorstore", embedding_function=embedding_model)

# 2. 加载重排序模型(CrossEncoder:一次比对一段与问题是否相关)
reranker = CrossEncoder("BAAI/bge-reranker-base")

question = "2026赛季迈凯伦的1号车手是谁？"

# 3. 先向量检索"多捞"候选(k=8)
candidates = vectorstore.similarity_search(question, k=10)
print(f"初检候选 {len(candidates)} 条")

pairs = [(question, c.page_content) for c in candidates]

# 重排
scores = reranker.predict(pairs)          # 模型对每个候选打分
top5 = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:5]  # 按分数排，取前3

print("重排序后 Top3：")
for i, (doc, score) in enumerate(top5, 1):
    print(f"--- 第{i}条 (分数:{score:.5f}) ---")
    print(doc.page_content[:250])
    print()