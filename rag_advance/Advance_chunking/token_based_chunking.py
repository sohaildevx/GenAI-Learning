from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import tiktoken

tokenizer = tiktoken.encoding_for_model("gpt-4o")

text = """ Retrieval-Augmented Generation (RAG) combines information retrieval with language generation. RAG retrieves relevant information from external sources and provides that information to an LLM. The LLM uses the retrieved context to generate more relevant and informative responses. Embeddings convert text into numerical vectors. Vector databases store these vectors for efficient similarity searches. """

documents = Document( page_content=text, metadata={"source": "rag_notes.txt"} )

def token_length(text):
    return len(tokenizer.encode(text))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=65, 
    chunk_overlap=10, 
    length_function=token_length
)

chunk = splitter.split_documents([documents])

for index, chunk in enumerate(chunk, start=1): 
    print(f"\nChunk {index}") 
    print("Content:", chunk.page_content) 
    print("Token count:", token_length(chunk.page_content)) 
    print("Metadata:", chunk.metadata)