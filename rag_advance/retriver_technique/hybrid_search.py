
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

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

# 1. Create embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# 2. Create vector store
vector_store = Chroma.from_texts(
    texts=documents,
    embedding=embeddings,
    collection_name="hybrid_search",
)

# 3. Create semantic retriever
semantic_retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# 4. Create BM25 retriever
keyword_retriever = BM25Retriever.from_texts(documents)
keyword_retriever.k = 3

# 5. Combine both retrievers
ensemble_retriever = EnsembleRetriever(
    retrievers=[
        semantic_retriever,
        keyword_retriever
    ],
    weights=[0.7, 0.5]
)

# 6. Perform hybrid search
query = "Which database does the application use?"

results = ensemble_retriever.invoke(query)

# 7. Display results
for index, doc in enumerate(results, start=1):
    print(f"\nResult {index}")
    print(doc.page_content)
