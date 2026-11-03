from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

documents = [
     Document( page_content="RAG retrieves external information to improve LLM responses.", 
              metadata={"category": "GenAI", "source": "rag_notes.pdf"} ), 
     Document( page_content="Embeddings convert text into numerical vectors.", 
              metadata={"category": "GenAI", "source": "rag_notes.pdf"} ),
     Document( page_content="Large language models can generate text based on learned patterns.", 
              metadata={"category": "GenAI", "source": "llm_notes.pdf"} ), 
    Document( page_content="Docker helps developers containerize applications.", 
             metadata={"category": "DevOps", "source": "docker_notes.pdf"} ), 
    Document( page_content="Git helps developers track changes to source code.", 
             metadata={"category": "DevOps", "source": "git_notes.pdf"} ), 
    Document( page_content="JavaScript is commonly used for frontend development.", 
             metadata={"category": "Web Development", "source": "javascript_notes.pdf"} ), 
]

embeddigns = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddigns,
    collection_name="metadata"
)

result = vector_store.similarity_search(
    query="Explain technology used in AI applications",
    k=3,
    filter={
        "category": {
        "$in": ["GenAI","DevOps"]
    }
    }
)

for index, docs in enumerate(result, start=1):
    print(f"result {index}")
    print(docs.page_content)
    print(docs.metadata)