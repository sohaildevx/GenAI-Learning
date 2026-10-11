from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

llm = ChatOpenAI(model="gpt-4", streaming=True)

question = "What is Gen-AI"

res = llm.stream(question)

for chunk in res:
    print(chunk.content, end="")
print(res)