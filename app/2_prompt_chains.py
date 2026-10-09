from langchain_openai import ChatOpenAI
from dotenv import load_dotenv;
from langchain_core.output_parsers import StrOutputParser

load_dotenv();

# llm = ChatOpenAI(model="gpt-4o")

# prompts =[
#     {"role":"system", "content":"You are a Python Developer?"},
#     {"role":"user", "content":"Difference b/w map and filter"},
# ]

# result = llm.invoke(prompts)
# print(result.content)


# Prompt Templates  - Dynamic Prompts
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o")
out = StrOutputParser()

def transform_case(result:str):
    return result.upper()

prompts = ChatPromptTemplate.from_messages([
    {"role":"system", "content":"You are a translator and translate input {language}?"},
    {"role":"user", "content":"{query}"},
])

# final_prompts = prompts.format_messages(language="Hinglish", query="I love python, and javascript")

# res = llm.invoke(final_prompts);
# print(res.content);

# res = llm.invoke(prompts.invoke({"language":"Hindi","query":"I love python and javascript"}))

# print(res.content)

chains = prompts | llm | out | transform_case
res = chains.invoke({"language":"Hinglish","query":"I love python and javascript"})
print(res)