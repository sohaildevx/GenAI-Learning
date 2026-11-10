from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START,END
from pydantic import BaseModel, Field

load_dotenv()

loader = PyPDFLoader("../data/flamingo.pdf")

docs = loader.load()

llm_model = ChatGroq(model="openai/gpt-oss-20b")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=200
)
docs = splitter.split_documents(docs)

embed = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore.from_documents(
    documents=docs,
    embedding=embed
)

class RagState(BaseModel):
    question:str = Field(description="User Question")
    documents: list = Field(default_factory=list)
    context:str = Field(description="Context data for user questions",default="")
    answer:str = Field(description="Final Answer..", default="")

def retrive_node(state:RagState) -> RagState:
    docs = vector_store.similarity_search(query=state.question)
    state.documents = docs
    return state

def create_context_node(state:RagState) -> RagState:
    conetext = ""
    for doc in state.documents:
        conetext += doc.page_content + "\n\n"
    state.context = conetext
    return state

def generate_node(state:RagState) -> RagState:
    prompt = f"""
        You are an assistant. Answer the user's question using only the provided context.
        If the answer is not in the context, say "I don't know."

        Context:
        {state.context}

        Question: {state.question}
    """
    res = llm_model.invoke(prompt)
    state.answer = res.content
    return state

graph = StateGraph(RagState)
graph.add_node("retrive_node", retrive_node)
graph.add_node("create_context_node",create_context_node)
graph.add_node("generate_node",generate_node)

graph.add_edge(START, "retrive_node")
graph.add_edge("retrive_node", "create_context_node")
graph.add_edge("create_context_node","generate_node")
graph.add_edge("generate_node",END)

graph = graph.compile()

res = graph.invoke({"question":"what is the pdf about?"})
print(res["answer"])