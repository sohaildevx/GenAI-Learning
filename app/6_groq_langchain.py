from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b")
openAiLLm = ChatOpenAI(model="gpt-5")

res = llm.invoke("Can u explain me Gen-Ai ?")
# res1 = openAiLLm.invoke("Can u explain me Gen-Ai ?")

print(res.content)
# print("res 1: ", res1.content)