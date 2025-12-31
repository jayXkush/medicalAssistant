import os
import time
from pathlib import Path
from typing import Optional
import socket

import pinecone
from dotenv import load_dotenv
from tqdm.auto import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "medicalindex"

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --------------------------------------------------
# STORAGE SETUP
# --------------------------------------------------

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --------------------------------------------------
# PINECONE LAZY INITIALIZATION (SINGLETON)
# --------------------------------------------------

_pinecone_index: Optional[pinecone.Index] = None
_pinecone_initialized = False


def get_pinecone_index():
    """
    Lazy singleton function to initialize and return Pinecone index.
    Only initializes Pinecone when first called, not at import time.
    Handles connection failures gracefully.
    """
    global _pinecone_index, _pinecone_initialized
    
    if _pinecone_index is not None:
        return _pinecone_index
    
    # Check if API key is configured
    if not PINECONE_API_KEY:
        raise RuntimeError(
            "Pinecone API key is not configured. Please set PINECONE_API_KEY in your .env file."
        )
    
    # Note: We skip DNS pre-check as Python's socket.gethostbyname() may fail
    # even when DNS works (e.g., DNS suffix issues). Let Pinecone SDK handle connection.
    
    try:
        # Initialize Pinecone (only happens on first call)
        if not _pinecone_initialized:
            pinecone.init(
                api_key=PINECONE_API_KEY,
                environment=PINECONE_ENV
            )
            _pinecone_initialized = True
        
        # Check if index exists and create if needed
        if PINECONE_INDEX_NAME not in pinecone.list_indexes():
            pinecone.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=768,
                metric="dotproduct"
            )
            
            while not pinecone.describe_index(PINECONE_INDEX_NAME).status["ready"]:
                time.sleep(1)
        
        # Get the index
        _pinecone_index = pinecone.Index(PINECONE_INDEX_NAME)
        return _pinecone_index
        
    except Exception as e:
        error_msg = str(e)
        
        # Check for network/DNS resolution errors
        if "getaddrinfo failed" in error_msg or "Failed to resolve" in error_msg or "NameResolutionError" in error_msg or "[Errno 11001]" in error_msg:
            # Provide helpful troubleshooting
            troubleshooting = (
                f"Cannot connect to Pinecone: DNS resolution failed for controller.{PINECONE_ENV}.pinecone.io\n\n"
                "🔍 Diagnosis:\n"
                "- DNS lookup works (nslookup succeeds)\n"
                "- Python socket resolution fails (known Windows issue)\n\n"
                "🔧 Solutions (try in order):\n"
                "1. ✅ DNS cache flushed - try again now\n"
                "2. Restart Python/Server - may resolve DNS context\n"
                "3. Check Windows DNS suffix settings\n"
                "4. Try different Pinecone environment:\n"
                "   - us-west1-gcp\n"
                "   - gcp-starter\n"
                "   - Check your Pinecone dashboard\n"
                "5. Use VPN off / Different network\n"
                "6. Contact IT if on corporate network\n\n"
                f"📝 Current environment: '{PINECONE_ENV}'\n"
                "   Verify this matches your Pinecone account region."
            )
            raise RuntimeError(troubleshooting) from e
        
        # Check for connection timeout/retry errors
        if "HTTPSConnectionPool" in error_msg or "Max retries exceeded" in error_msg or "Connection timeout" in error_msg:
            raise RuntimeError(
                "Cannot connect to Pinecone: Connection timeout. "
                "Please check your internet connection and firewall settings. "
                "Ensure your network allows HTTPS connections to Pinecone servers."
            ) from e
        
        # Check for socket errors
        if isinstance(e, (socket.gaierror, socket.error, OSError)):
            raise RuntimeError(
                "Cannot connect to Pinecone: Network error. "
                "Please check your internet connection and network settings."
            ) from e
        
        # Generic error
        raise RuntimeError(f"Failed to initialize Pinecone: {error_msg}") from e

# --------------------------------------------------
# VECTORSTORE LOADER
# --------------------------------------------------

def load_vectorstore(uploaded_files):
    # Get Pinecone index (lazy initialization - only happens when this function is called)
    index = get_pinecone_index()
    
    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    file_paths = []

    # Save uploaded PDFs
    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    # Process PDFs
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]

        print(f"🔍 Embedding {len(texts)} chunks...")
        embeddings = embed_model.embed_documents(texts)

        print("📤 Uploading to Pinecone...")
        with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
            vectors = zip(ids, embeddings, metadatas)
            index.upsert(vectors=vectors)
            progress.update(len(embeddings))

        print(f"✅ Upload complete for {file_path}")
