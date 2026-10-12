from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv();

llm = ChatOpenAI(model="gpt-4o")

@tool
def add_numbers(a:int, b:int):
    """
       It will return the sum of two numbers

       Args:
            a: Number One
            b: Number Two
    """

    return a + b;

@tool
def multiply_numbers(a:int, b:int):
    """
       It will return the multiplication of two numbers

       Args:
            a: Number One
            a: Number Two
    """

    return a * b;

res = add_numbers.invoke({"a":12,"b":3})
print(res)


agent =create_agent(
       model=llm,
       tools=[add_numbers, multiply_numbers],
       system_prompt="You are a math teacher, and always use tool for calculations"
)

response = agent.invoke({"messages":[{"role":"user", "content":"what is 2+3 * 5"}]})

# for res in response['messages']:
#     print(res)
#     print("\n")



print(response['messages'][-1].content)