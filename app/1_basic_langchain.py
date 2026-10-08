# Langchain model - OpenAi.

from dotenv import load_dotenv

load_dotenv();

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o");

# prompt = [
#     ("system","You are a senior python developer."),
#     ("user", "Write a pyhon code to sort the array with explanation")
# ]

# result = llm.invoke(prompt);

# print(result.content)


# Langchain model - Gemini

from langchain_google_genai import GoogleGenerativeAI

gemini_llm = GoogleGenerativeAI(model="gemini-3.7-flash")

prompt = [
    ("system","You are a senior Full stack developer."),
    ("user", "Write a pyhon code to sort the array with explanation")
]

result = gemini_llm.invoke(prompt);
print(result);