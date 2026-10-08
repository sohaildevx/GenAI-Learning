# Langchain model - OpenAi.

from dotenv import load_dotenv

load_dotenv();

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o");

result = llm.invoke("Who is pm of india ?");

print(result.content)