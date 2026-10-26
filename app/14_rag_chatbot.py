from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
load_dotenv()

data_file = Path(__file__).resolve().parent.parent / "data" / "flamingo.pdf"
loader = PyPDFLoader(str(data_file))
docs = loader.load()
print(len(docs))

if not any(doc.page_content.strip() for doc in docs):
    raise ValueError(
        f"No text could be extracted from {data_file.name}. "
        "This PDF appears to contain scanned images; run OCR on it or provide a text-based PDF."
    )

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

splitted_data = splitter.split_documents(docs)

print(len(splitted_data))

if not splitted_data:
    raise ValueError("The PDF produced no text chunks for embedding.")


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma.from_documents(
    documents=splitted_data,
    embedding=embeddings,
)

llm_model = ChatOpenAI(model="gpt-4o")

query = "what is the best chapter to read from it and why"

# data = vector_store.similarity_search(query=query)
# print(data[0])

# context = ""

# for doc in data:
#     context += doc.page_content + "\n"  
    # print(context)

# res = llm_model.invoke(f"Can u provide me the answer based on provided context for my question, context: {context} and question: {query}")

# print(res.content)

def get_context(query:str):
    data = vector_store.similarity_search(query=query)
    context = ""
    for doc in data:
      context += doc.page_content + "\n" 

    return{
        "context":context,
        "question":query
    }

prompt = PromptTemplate.from_template(
    """
    You are a helpful assistant and provide answers based on the context for the user's question.
    If you don't know the answer, say "I don't know."
    Context: {context}
    Question: {question}
    """
)

rag_chain = RunnableLambda(get_context) | prompt | llm_model

res = rag_chain.invoke("What is the first chapter name?")

print(res.content)