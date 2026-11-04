from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from IPython.display import Image

class GreetState(BaseModel):
    message:str = ""

graph = StateGraph(GreetState)

def Greet(state:GreetState):
    state.message = f"{state.message} ! All Good ! "
    return state

def upperCase(state:GreetState):
    state.message = state.message.upper()
    return state

graph.add_node("greet",Greet)
graph.add_node("upper_case",upperCase)

graph.add_edge(START, "greet")
graph.add_edge("greet",END)

# graph.add_edge(START, "upper_case")
graph.add_edge("upper_case",END)

finalGraph = graph.compile()

res = finalGraph.invoke({"message":"I love langGraph !"})

# imageResult = Image(finalGraph.get_graph().draw_mermaid_png())
print(res)