import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from rag_chain import build_rag_chain
from memory_helper import add_memory
from utils import extract_text

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env")

FAISS_DIR = "/data/ashish_backup/langchainn/faiss_index"

# ----------------------------
# Embeddings (LOCAL)
# ----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------------
# Load FAISS Vector Store
# ----------------------------
vectorstore = FAISS.load_local(
    FAISS_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ----------------------------
# Gemini LLM
# ----------------------------
llm = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest",
    temperature=0.2
)

# ----------------------------
# Build RAG + Memory Chain
# ----------------------------
base_chain = build_rag_chain(llm, retriever)
rag_chain = add_memory(base_chain)

SESSION_ID = "default-session"

# ----------------------------
# Run Chat Loop
# ----------------------------
if __name__ == "__main__":
    while True:
        q = input("\nAsk (or 'exit'): ")
        if q.lower() == "exit":
            break

        result = rag_chain.invoke(
            {"question": q},
            config={"configurable": {"session_id": SESSION_ID}}
        )

        answer = extract_text(result)
        print("\nAnswer:\n", answer)
