from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent;
from langgraph.checkpoint.memory import MemorySaver

load_dotenv();

model = ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()

# res = search.run("Address of TecchSimplus Learning")

# print(res)

agent = create_agent(
    model=model,
    tools=[search.run],
    system_prompt="You are a agent and can search for any question on google",
    checkpointer=MemorySaver()
)

while True:
    question = input("User: ");
    if question.lower() in ["bye","exit","end"]:
        print("Good Bye!")
        break;

    response = agent.invoke({"messages":[{"role":"user","content":question}]},
     {"configurable":{"thread_id":"sohail"}}                       
    )

    print("AI: ",response["messages"][-1].content)

     
     