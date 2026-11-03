
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
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
    "Large language models can generate text based on learned patterns."
]

# Create embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Create vector store
vector_store = Chroma.from_texts(
    texts=documents,
    embedding=embeddings,
    collection_name="top_k_demo"
)

query = "Explain machine learning"

# Experiment with different K values
for k in [1, 3, 5, 10]:

    results = vector_store.similarity_search_with_relevance_scores(
        query=query,
        k=k
    )

    for doc, score in results:
        if score >= 0.5:
            print(doc.page_content)

    # print(f"\n========== K = {k} ==========")

    # for index, doc in enumerate(results, start=1):
    #     print(f"\nResult {index}:")
    #     print(doc.page_content)
