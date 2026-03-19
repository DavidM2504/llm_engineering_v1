"""
This script builds a vector database from a local knowledge base of Markdown files.

Steps performed:
1. Load all Markdown documents from subfolders inside the knowledge base directory.
2. Attach metadata (document type based on folder name) to each document.
3. Split documents into smaller chunks for better embedding performance.
4. Generate embeddings for each chunk using a HuggingFace model.
5. Store the embeddings in a Chroma vector database for later retrieval (e.g., in RAG systems).

NOTES FOR EXTENDING THIS SCRIPT:

- Adding other file formats:
  You can extend the DirectoryLoader by:
    * Changing the glob pattern (e.g., "*.txt", "*.pdf")
    * Using different loader classes (e.g., PyPDFLoader, CSVLoader, UnstructuredFileLoader)
  Example:
    loader = DirectoryLoader(folder, glob="**/*.pdf", loader_cls=PyPDFLoader)

- Using logging instead of print:
  Replace print() with Python's logging module:
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info("message")
  This allows better control over log levels and output destinations.

- Converting into a reusable class/module:
  Wrap the functions into a class (e.g., VectorDBBuilder) with methods like:
    - load_documents()
    - split_documents()
    - build_vectorstore()
  Then initialize with config (paths, model, chunk size) for reuse across projects.
"""

import os
import glob
from pathlib import Path

# LangChain document loaders for reading files from directories
from langchain_community.document_loaders import DirectoryLoader, TextLoader
# To support other formats, you could import additional loaders like:
# from langchain_community.document_loaders import PyPDFLoader, CSVLoader, UnstructuredFileLoader

# Text splitters to break documents into manageable chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter

# Vector database (Chroma) for storing embeddings
from langchain_chroma import Chroma

# Embedding models
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# Load environment variables (e.g., API keys)
from dotenv import load_dotenv

# Optional: For production use, replace print() with logging:
# import logging
# logging.basicConfig(level=logging.INFO)

MODEL = "gpt-4.1-nano"

# Path where the vector database will be stored
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

# Path to the folder containing knowledge base documents
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

# Initialize embedding model (local HuggingFace model)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load environment variables from .env file
load_dotenv(override=True)

# Optional: Use OpenAI embeddings instead of HuggingFace (currently commented out)
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def fetch_documents():
    """
    Loads all Markdown documents from subfolders inside the knowledge base directory.

    HOW TO EXTEND FOR OTHER FILE TYPES:
    - Change glob="**/*.md" to include other formats (e.g., "*.txt", "*.pdf")
    - Use appropriate loader_cls depending on file type:
        *.txt  -> TextLoader
        *.pdf  -> PyPDFLoader
        *.csv  -> CSVLoader
        mixed  -> UnstructuredFileLoader (handles multiple formats)

    Example for multiple formats:
        glob="**/*.*" and choose loader dynamically per file type.
    """
    # Get all subfolders inside the knowledge base directory
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))

    documents = []

    # Loop through each folder (each represents a document category/type)
    for folder in folders:
        # Extract the folder name (used as document type metadata)
        doc_type = os.path.basename(folder)

        # Create a loader that reads all .md files recursively in the folder
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",  # 🔹 Change this to support other file types
            loader_cls=TextLoader,  # 🔹 Replace with appropriate loader if needed
            loader_kwargs={"encoding": "utf-8"}
        )

        # Load documents from the folder
        folder_docs = loader.load()

        # Add metadata and collect documents
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)

    return documents


def create_chunks(documents):
    """
    Splits documents into smaller overlapping chunks.

    NOTE:
    - chunk_size controls how large each chunk is
    - chunk_overlap ensures context is preserved between chunks
    - You can tune these values depending on your embedding model and use case
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # Alternative: Markdown-aware splitting (preserves headings/structure)
    # text_splitter = MarkdownTextSplitter()

    chunks = text_splitter.split_documents(documents)

    return chunks


def create_embeddings(chunks):
    """
    Generates embeddings for each chunk and stores them in a Chroma vector database.

    LOGGING NOTE:
    - Replace print() with logging for better production control:
        logging.info(f"...")
        logging.warning(f"...")
        logging.error(f"...")
    - You can also log to files instead of console.
    """
    # If a previous vector database exists, delete its collection
    if os.path.exists(DB_NAME):
        Chroma(
            persist_directory=DB_NAME,
            embedding_function=embeddings
        ).delete_collection()

    # Create a new vector store from document chunks
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(
        limit=1,
        include=["embeddings"]
    )["embeddings"][0]

    dimensions = len(sample_embedding)

    # Replace this with logging.info(...) in production
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")

    return vectorstore


# REUSABLE MODULE / CLASS NOTE:
# --------------------------------
# To make this reusable:
# 1. Wrap logic into a class, e.g.:
#
#    class VectorDBBuilder:
#        def __init__(self, db_path, kb_path, embedding_model):
#            ...
#
#        def load_documents(self):
#            ...
#
#        def split_documents(self):
#            ...
#
#        def build(self):
#            ...
#
# 2. Move config (paths, chunk size, model) into __init__
# 3. Allow dependency injection (pass embeddings, splitter, etc.)
# 4. Import and reuse in other scripts instead of running as __main__
#
# This makes the pipeline reusable for APIs, batch jobs, or different datasets.


if __name__ == "__main__":
    # Step 1: Load documents from knowledge base
    documents = fetch_documents()

    # Step 2: Split documents into chunks
    chunks = create_chunks(documents)

    # Step 3: Generate embeddings and store them in vector DB
    create_embeddings(chunks)

    # Replace with logging.info("Ingestion complete") if using logging
    print("Ingestion complete")