from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load our existing vector database
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)


# Create the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0,
)


# Ask the user a question
question = input("\nAsk a question about the employee handbook: ")


# Retrieve relevant chunks
results = vectorstore.similarity_search(
    question,
    k=5
)


# Combine the retrieved chunks into context
context = "\n\n".join(
    document.page_content
    for document in results
)


# Create the prompt
prompt = f"""
You are an assistant that answers questions about the YEAB Technologies
Employee Handbook.

Answer the question ONLY using the information provided in the context.

If the answer cannot be found in the context, say:
"I couldn't find that information in the employee handbook."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""


# Ask Gemini
response = llm.invoke(prompt)


# Display the answer
print("\n================ ANSWER ================\n")
print(response.text)