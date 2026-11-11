from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph
from langchain_groq import ChatGroq
from pydantic import BaseModel
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from typing import Literal

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b")
class MailState(BaseModel):
    query:str = ""
    draft:str = ""
    human_feedback:str = ""
    final_response:str = ""

# Defining the nodes

def DraftEmail(state:MailState) -> MailState:
    if state.human_feedback:
        draft = llm.invoke(f"""
                            You have generated this mail: {state.draft},
                            Human feedback to apply: {state.human_feedback},
                            Rewrite this draft email for {state.query}
                            """).content
    else:
        draft = llm.invoke(state.query).content

    state.draft = draft
    return state


def human_fb(state:MailState)->MailState:
    feedback = interrupt({
        "draft_email": state.draft,
        "question":"Do you want to continue or re-rewrite the mail"
    })

    fb = (feedback or "").strip().lower()
    if fb in ("approved","ok","yes"):
        state.human_feedback = ""
        return state
    else:
        state.human_feedback = feedback
        return state

def finalize_node(state:MailState) -> MailState:
    if state.human_feedback:
        state.final_response = state.human_feedback
        return state
    else:
        print("Email Sent Successfully...")
        state.final_response = "Email Sent Successfully"
        return state

def condition_routing(state:MailState) -> Literal["draft","final"]:
    if state.human_feedback:
        return "draft"
    return "final"



graph_builder = StateGraph(MailState)
graph_builder.add_node("draft",DraftEmail)
graph_builder.add_node("human_fb",human_fb)
graph_builder.add_node("final",finalize_node)

graph_builder.add_edge(START,"draft")
graph_builder.add_edge("draft","human_fb")
graph_builder.add_conditional_edges("human_fb",condition_routing)
graph_builder.add_edge("final",END)

graph = graph_builder.compile(checkpointer=InMemorySaver())

config = {"configurable":{"thread_id":"1"}}

res = graph.invoke({"query":"I want to send a email for medical leave request, dates will be 2 june to 5 june, so create a simple mail."}, config)

res_command = graph.invoke(Command(resume="yes"), config=config)
print(res["draft"])
# print(res_command)