from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st;

load_dotenv();

llm = ChatOpenAI(model="gpt-4o")

# while True:
#     query = input("User: ")

#     if query.lower() in ["quit","exit","bye"]:
#         print("GoodBye ")
#         break
    
#     result = llm.invoke(query)
#     print("Agent: ",result.content)

st.title(" AskBuddy - AI QnA Bot")
st.markdown("My QnA Bot with LangChain and OpenAI !")

if "messages" not in st.session_state:
  st.session_state.messages = []

for message in st.session_state.messages:
   role = message["role"]
   content = message["content"]
   st.chat_message(role).markdown(content)


query = st.chat_input("Ask Anything")
if query:
    st.session_state.messages.append({"role":"user", "content":query})
    st.chat_message("user").markdown(query)
    res = llm.invoke(query)
    st.chat_message("ai").markdown(res.content)
    st.session_state.messages.append({"role":"ai", "content":res.content})

