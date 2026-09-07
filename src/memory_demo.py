from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

history = []


def chat(user_input):
    history.append({"role": "user", "content": user_input})

    resp = llm.invoke(history)

    history.append({"role": "assistant", "content": resp.content})
    return resp.content


print(chat("你好，我叫小李"))
print(chat("帮我记住我爱喝咖啡"))
print(chat("我叫什么名字？"))
print(chat("我喜欢喝什么？"))
