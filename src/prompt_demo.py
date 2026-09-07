from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

template = ChatPromptTemplate.from_messages([
    ("system","你是一名资深python工程师，20年经历，说话一针见血，且很擅长用简短的语言教会刚入行的晚辈"),
    ("human","{question}"),
])

resp = llm.invoke(template.invoke({"question": "怎么学python"}))

print(resp.content)
