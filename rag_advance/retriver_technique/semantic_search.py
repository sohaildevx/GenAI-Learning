from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma 

load_dotenv()

documents = [
    "Python is widely used for machine learning.",
    "JavaScript is commonly used for frontend development.",
    "Machine learning helps computers identify patterns in data.",
    "RAG retrieves external information to improve LLM responses.",
    "Embeddings convert text into numerical vectors.",
    "Docker helps developers containerize applications.",
    "Natural language processing enables computers to understand human language.",
    "Vector databases store embeddings for fast similarity searches.",
    "A prompt gives a language model instructions and context.",
    "Supervised learning trains models with labeled examples.",
    "Unsupervised learning discovers patterns in unlabeled data.",
    "Cloud computing provides on-demand access to computing resources.",
    "APIs allow different software applications to communicate with each other.",
    "Git helps developers track changes to source code.",
    "Data preprocessing improves the quality of information used for training.",
    "Large language models can generate text based on learned patterns.",
]

docs = OpenAIEmbeddings(model="text-embedding-3-small")

# providing an id to chroma so it won't return duplicate values
ids = [f"doc_{i}" for i in range(len(documents))]
vector_db = Chroma.from_texts(
    texts=documents,
    embedding=docs,
    ids=ids,
    collection_name="semantic_search",
    persist_directory="./semantic_db",
)

query = "What is rag?"

result = vector_db.similarity_search(
    query=query,
    k = 5
)

print("Total documents:", len(documents))

for index, documents in enumerate(result, start=1):
    print(f"Result {index}")
    print(documents.page_content)