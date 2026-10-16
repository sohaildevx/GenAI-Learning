from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader,CSVLoader

data_file = Path(__file__).resolve().parent.parent / "data" / "rag_data_loading.txt"
loader = TextLoader(str(data_file), encoding="utf-8")

docs = loader.load()

oneDoc = docs[0]
print(oneDoc.page_content)
print(docs)


# PDF document loader

pdf_file = Path(__file__).resolve().parent.parent / "data" / "rag_data_loading.pdf"

pdf_loader = PyPDFLoader(str(pdf_file))

pdf_docs = pdf_loader.load()

print(pdf_docs)


# CSV Loader

loader = CSVLoader("")
docs = loader.load()
docs[:5]
