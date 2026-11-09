import os
from urllib.parse import quote
from urllib.request import urlopen

from dotenv import  load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel, Field
from typing import Literal

# libs for agents
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool
load_dotenv()

class FlowState(BaseModel):
    question:str = Field(description="User Asked Question")
    category: Literal["coding","google_search","weather"] = Field(default="google_search")
    answer:str = Field(default="")

class QuestionCategory(BaseModel):
    category: Literal["coding","google_search","weather"] = Field(default="google_search",description="Question Category")

llm = ChatGroq(model="openai/gpt-oss-20b")

search = GoogleSerperAPIWrapper()
# memory = MemorySaver()

google_agent = create_agent(
    model=llm,
    tools=[search.run],
    system_prompt="You are a agent and can search for my questions on google"
)

@tool
def get_weather(city:str) -> str:
    """It provides real time weather details for any city"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENWEATHER_API_KEY is not configured")

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?units=metric&q={quote(city)}&appid={api_key}"
    )
    with urlopen(url) as response:
        return response.read().decode("utf-8")

weather_agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a agent and can provide real time weather details."
)

def check_question_catgeory(state:FlowState) -> FlowState:
    st_llm = llm.with_structured_output(QuestionCategory, method="json_mode")
    res = st_llm.invoke(
        f"Classify this question as exactly one of coding, google_search, or weather: "
        f"{state.question}. If unsure, use google_search. "
        'Return only valid JSON in this format: {"category":"google_search"}.'
    )
    print(res)
    state.category = res.category
    return state

# flow = FlowState(question="i want to write code in python")
# check_question_catgeory(flow)

def route(state:FlowState) -> Literal["coding","google_search","weather"]:
    return state.category

def coding_node(state:FlowState) -> FlowState:
    res = llm.invoke("You are a coding expert: {state.question}")
    state.answer = res.content
    return state

def weather_node(state:FlowState)->FlowState:
    print("Weather code")
    res = weather_agent.invoke({"messages":[
        {"role":"user","content": state.question}
    ]})
    state.answer = res["messages"][-1].content
    return state

def google_search_node(state:FlowState) -> FlowState:
    print("Google state")
    res = google_agent.invoke({"messages":[
        {"role":"user","content":state.question}
    ]})
    state.answer = res["messages"][-1].content
    return state

graph = StateGraph(FlowState)

graph.add_node("chechk_question_category",check_question_catgeory)
graph.add_node("coding",coding_node)
graph.add_node("google_search",google_search_node)
graph.add_node("weather",weather_node)

graph.add_edge(START, "chechk_question_category")
graph.add_conditional_edges("chechk_question_category",route)
graph.add_edge("coding",END)
graph.add_edge("google_search",END)
graph.add_edge("weather",END)

graph = graph.compile()

response = graph.invoke({"question":"what is the temperature in pune"})

print(response)