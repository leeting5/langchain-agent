from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = TextLoader("C:/Users/86177/projects/aiAgent/data/introduction_of_f1driver.txt", encoding="utf-8")
docs = loader.load()
print("原始文档长度：", len(docs[0].page_content), "字符")

splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=5)
chunks = splitter.split_documents(docs)
print("切分块数:", len(chunks))
print(chunks[0].page_content[:50])
