from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

# prompts =[
#     {"role":"user", "content":"Hello my name is sohail shaikh."},
#     {"role":"ai","content":"Hello, sohail shaikh! How can i help u?"},
#     {"role":"user","content":"What is my name?"}
# ]

# res = llm.invoke(prompts)

# print(res.content)

# Manage History

history = []

while True:
    query = input("User :")
    if query.lower() in ["exit","quit","bye"]:
        print("Good Bye")
        break
    
    history.append({"role":"user","content":query})
    print("User: ", query)

    res = llm.invoke(history)
    history.append({"role":"ai","content":res.content})

    print("AI: ", res.content, "\n")