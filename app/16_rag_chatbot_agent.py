from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import  OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st
load_dotenv()

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

def process_document(path):

    loader = PyPDFDirectoryLoader(path)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

    docs_splitter = splitter.split_documents(documents=docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = InMemoryVectorStore.from_documents(
        documents=docs_splitter,
        embedding=embeddings
    )

    llm_model = ChatGroq(model="openai/gpt-oss-20b")

    @tool
    def retrieve_context(query:str):
        """
        Retrieve documents relevant to a  query from the knowledge base.
        """

        context  = ""

        docs = vector_db.similarity_search(query=query, k=3)
        for doc in docs:
            context = doc.page_content + "\n\n"

        return context

    system_prompt = """You are a helpful question-answering assistant for the provided PDF knowledge base.

    Answer the user's question using only information retrieved from the PDF with the
    `retrieve_context` tool. Always call the tool before answering questions about the
    PDF. Do not rely on your general knowledge, invent facts, or fill in missing details.

    If the retrieved context does not contain enough information to answer the question,
    say: "I couldn't find that information in the provided PDF." Do not guess.

    Keep answers clear, concise, and directly relevant to the user's question. When
    useful, organize the response with short bullet points. If the answer comes from
    multiple parts of the PDF, combine them accurately without changing their meaning.
    """

    memory = InMemorySaver()

    agent = create_agent(
        model=llm_model,
        tools = [retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True

if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="Select PDF File", type={"pdf"}, accept_multiple_files=True)

    if uploaded:
        with st.spinner("Processing..."):
            path = "./docs_files/"
            for file in uploaded:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())

            process_document(path)
            st.rerun()


if st.session_state.document_uploaded and st.session_state.agent:

    for messages in st.session_state.messages:
        role = messages.get("role")
        content = messages.get("content")
        st.chat_message(role).markdown(content)

    query = st.chat_input("Ask Anything related to uploaded documents...")


    if query:
        st.session_state.messages.append({"role":"user","content":query})

        st.chat_message("user").markdown(query)
        response = st.session_state.agent.invoke(
            {"messages":[{"role":"user","content":query}]},
            {"configurable":{"thread_id":1}}
        )

        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role":"ai","content":answer})


            