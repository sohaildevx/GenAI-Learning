
from langchain_community.retrievers import BM25Retriever

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


retriever = BM25Retriever.from_texts(documents)


retriever.k = 3


query = "What is rag"


results = retriever.invoke(query)

for index, doc in enumerate(results, start=1):
    print(f"\nResult {index}")
    print(doc.page_content)

