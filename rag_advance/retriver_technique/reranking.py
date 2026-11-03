
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from sentence_transformers import CrossEncoder

load_dotenv()

# 1. Create documents
documents = [
    Document(
        page_content="Python is widely used for machine learning."
    ),
    Document(
        page_content="RAG retrieves external information to improve LLM responses."
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors."
    ),
    Document(
        page_content="Docker helps developers containerize applications."
    ),
    Document(
        page_content="Large language models can generate text based on learned patterns."
    ),
    Document(
        page_content="Vector databases store embeddings for fast similarity searches."
    ),
]

# 2. Create embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# 3. Create vector store
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="reranking_demo"
)

# 4. Initial retrieval
query = "What is Retrieval-Augmented Generation?"

retrieved_docs = vector_store.similarity_search(
    query=query,
    k=5
)

print("\n--- Initial Retrieval ---")

for index, doc in enumerate(retrieved_docs, start=1):
    print(f"\nRank {index}: {doc.page_content}")

# 5. Load Cross-Encoder
reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

# 6. Create query-document pairs
pairs = [
    [query, doc.page_content]
    for doc in retrieved_docs
]

# 7. Calculate relevance scores
scores = reranker.predict(pairs)

# 8. Combine documents with scores
ranked_results = sorted(
    zip(retrieved_docs, scores),
    key=lambda item: item[1],
    reverse=True
)

# 9. Display reranked results
print("\n--- Reranked Results ---")

for index, (doc, score) in enumerate(ranked_results, start=1):
    print(f"\nRank: {index}")
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content}")

