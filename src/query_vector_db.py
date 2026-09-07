# 从已建好的向量库加载数据，并用向量相似度检索
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 加载embedding模型，和构建时必须是同一个（因为不同的模型坐标系不同，因此在哪里存的就在哪里查）
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
)

# 从已存储的目录加载向量库
vectorstore = Chroma(
    persist_directory="vector_db/vectorstore",
    embedding_function=embedding_model,
)

question = "2026赛季迈凯伦的1号车手是谁？"

# 找最相似的三段
results = vectorstore.similarity_search(question,k=5)

print(f"检索到{len(results)}条相关内容：")
print("-" * 50)

# 遍历每条结果，打印出它命中的内容前20字
for i, r in enumerate(results, 1):
    print(f"---第{i}条---")
    print(r.page_content[:50])

    print()