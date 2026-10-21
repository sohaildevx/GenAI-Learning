from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
load_dotenv()

document = [
    "Artificial intelligence is changing how people work.",
    "Machine learning finds patterns in data.",
    "Natural language processing helps computers understand text.",
    "Embeddings represent text as numerical vectors.",
    "Python is widely used for data science.",
    "Search systems can compare similar meanings.",
    "Clean data improves machine learning results.",
    "Practice is important when learning new technology.",
]

embeddigns = OpenAIEmbeddings(model="text-embedding-3-large")

vector = embeddigns.embed_documents(document)
print(len(vector[0]))


# Store Vector Embedding in vector db.

vector_store = Chroma.from_texts(
    texts=document,
    embedding=embeddigns,
    persist_directory="./vector_db"
)

query = "Machine Learnings"

result = vector_store.similarity_search(query=query, k=2)

print(result)