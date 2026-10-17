from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
load_dotenv()

data_dir = Path(__file__).resolve().parent.parent / "data"

text = "RAG Data Loading - Beginner Notes, RAG means Retrieval-Augmented Generation. It lets an application retrieve" 

splitter = RecursiveCharacterTextSplitter(chunk_size = 50, chunk_overlap= 20)

chunks = splitter.split_text(text)

# print(chunks)

# Document Text Loader

loader = TextLoader(str(data_dir / "rag_data_loading.txt"), encoding="utf-8")
texts = loader.load()

pdfloader = PyPDFLoader(str(data_dir / "rag_data_loading.pdf"))
pdfs = pdfloader.load()
print(len(pdfs))

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

text_chunks = splitter.split_documents(texts)

# print(text_chunks)

pdf_chunks = splitter.split_documents(pdfs)
print(len(pdf_chunks))