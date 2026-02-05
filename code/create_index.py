import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

PDF_PATH = "/data/ashish_backup/langchainn/data/rag_doc.pdf"
FAISS_DIR = "/data/ashish_backup/langchainn/faiss_index"

# Load PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

# Local embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS index
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save index
vectorstore.save_local(FAISS_DIR)

print("✅ FAISS index created and saved")
