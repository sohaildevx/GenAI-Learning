from dotenv import load_dotenv;
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
load_dotenv()

loader = PyPDFLoader("../data/flamingo.pdf")
docs = loader.load()

print(len(docs))

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

splitted_docs = splitter.split_documents(docs)

# print(splitted_docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = InMemoryVectorStore.from_documents(
    documents=splitted_docs,
    embedding=embeddings
)

@tool
def retriever_tool(query: str):
    """
    Retrieve relevant passages from the NCERT Flamingo English textbook PDF.

    Use this tool whenever the user asks about the book, its chapters, poems,
    prose, authors, themes, or any other information that should come from
    the PDF.
    """

    print("Tool Called: ", query)
    docs = vector_store.similarity_search(query=query, k=4)  
    # k is responsible for search how many similarity search we want to search
    return "\n\n".join(doc.page_content for doc in docs)


# retriever_tool.invoke("Book Name")

llm = ChatOpenAI(model="gpt-5")

system_prompt = """
You answer questions about the NCERT Flamingo English textbook.

For every question about the book, its chapters, poems, prose, authors, themes,
or other document content, call `retriever_tool` first and answer only from
the passages it returns. Do not rely on outside knowledge or invent details.
If the retrieved passages do not contain the answer, say that the information
is not available in the PDF. Keep the answer clear and mention the relevant
chapter or poem when the context identifies it.
"""

agent = create_agent(
    model=llm,
    tools=[retriever_tool],
    system_prompt=system_prompt
)

query = "What is the name of this book, and what does it contain?"

response = agent.invoke({"messages": [{"role": "user", "content": query}]})
result = response["messages"][-1].content

print(result)