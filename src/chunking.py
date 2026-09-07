from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Load PDF
pdf_path = Path("Data/YEAB_Technologies_Employee_Handbook.pdf")

loader = PyPDFLoader(str(pdf_path))
documents = loader.load()

print(f"Total pages: {len(documents)}")


# 2. Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)


# 3. Split documents into chunks
chunks = text_splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")

