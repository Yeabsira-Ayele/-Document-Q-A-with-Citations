from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from QuestiionandAnswer.src.chunking import chunks


# Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Store chunks in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,    persist_directory="./chroma_db",
)

print("Vector database created successfully!")
