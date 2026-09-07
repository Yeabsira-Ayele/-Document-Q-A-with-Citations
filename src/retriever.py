from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load existing Chroma database
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)


# Question
question = "How many days of PTO do employees receive?"


# Similarity search
results = vectorstore.similarity_search_with_score(
    question,
    k=5
)


print(f"\nFound {len(results)} results.\n")


for i, (document, score) in enumerate(results, start=1):

    print(f"========== RESULT {i} ==========")

    print(f"Similarity score: {score}")

    print(f"Page: {document.metadata.get('page')}")

    print("\nContent:")
    print(document.page_content)

    print()