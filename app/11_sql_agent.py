from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import streamlit as st
load_dotenv()



db = SQLDatabase.from_uri("sqlite:///my_task.db")

db.run(
    """
   CREATE TABLE IF NOT EXISTS task(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT CHECK (status IN ('pending','in_progress','completed')) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   )
"""
)

llm_model = ChatGroq(model="openai/gpt-oss-20b")
toolkit = SQLDatabaseToolkit(db=db, llm=llm_model)
tools = toolkit.get_tools()


system_prompt = """
You are a task management assistant that interacts with a SQL database containing a 'task'

TASK RULES:
1. Limit SELECT queries to 10 results max with ORDEr By created_at DESc
2. After CREATE/UPDATE/DELETE, confirm with SELECT query
3. If the user requests a list of tasks, present the output in a structured table format

CRUD OPERATING:
CREATE: INSERT INTO task(title, description, status)
READ: SELECT * FROM task WHERE ... LIMIT 10
UPDATE: UPDATE task SET status=? WHERE id=? OR title=?
DELETE: DELETE FROM task WHERE id=? OR title=?

Table schema: id, title, description, status(pending/in_progress/completed), created_at.
"""

@st.cache_resource
def get_agent():
    agent = create_agent(
    model=llm_model,
    tools=tools,
    checkpointer=InMemorySaver(),
    system_prompt=system_prompt
   )
    return agent

agent = get_agent()

st.subheader("📃TaskBot - Manage your tasks")

if "messages" not in st.session_state:
       st.session_state.messages = [];

for message in st.session_state.messages:
     st.chat_message(message["role"]).markdown(message["content"])
     

prompt = st.chat_input("Ask me to manage your task ")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant"):
        with st.spinner("Processing"):
            response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        {"configurable": {"thread_id": "1"}}
        )

    result = response["messages"][-1].content
    st.markdown(result)
    st.session_state.messages.append({"role":"assistant","content":result})