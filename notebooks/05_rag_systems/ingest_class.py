"""
VectorDBBuilder: A reusable pipeline for building a vector database from a knowledge base.

Features:
- Supports multiple file formats (Markdown, TXT, PDF, etc.)
- Configurable chunking and embedding
- Uses logging instead of print statements
- Designed for reuse in scripts, APIs, or batch jobs
"""

import os
import glob
import logging
from pathlib import Path
from typing import List, Optional, Type

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    # Optional loaders for other formats:
    # PyPDFLoader,
    # CSVLoader,
    # UnstructuredFileLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class VectorDBBuilder:
    def __init__(
        self,
        knowledge_base_path: str,
        db_path: str,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        glob_pattern: str = "**/*.md",
        loader_cls: Type = TextLoader,
        log_level: int = logging.INFO,
    ):
        """
        Initialize the vector database builder.

        Args:
            knowledge_base_path: Path to folder containing documents
            db_path: Path where vector DB will be stored
            embedding_model_name: HuggingFace embedding model
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            glob_pattern: File pattern (e.g., "*.md", "*.txt", "*.pdf")
            loader_cls: Loader class depending on file type
            log_level: Logging level
        """
        self.knowledge_base_path = knowledge_base_path
        self.db_path = db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.glob_pattern = glob_pattern
        self.loader_cls = loader_cls

        # Setup logging
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # Load environment variables
        load_dotenv(override=True)

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name
        )

    # ------------------------------------------------------------------
    # DOCUMENT LOADING
    # ------------------------------------------------------------------
    def load_documents(self) -> List:
        """
        Load documents from the knowledge base.

        To support other formats:
        - Change glob_pattern (e.g., "*.pdf", "*.txt")
        - Change loader_cls (e.g., PyPDFLoader)

        For mixed formats:
        - You can implement dynamic loader selection per file type
        """
        self.logger.info("Loading documents...")

        folders = glob.glob(str(Path(self.knowledge_base_path) / "*"))
        documents = []

        for folder in folders:
            doc_type = os.path.basename(folder)

            loader = DirectoryLoader(
                folder,
                glob=self.glob_pattern,
                loader_cls=self.loader_cls,
                loader_kwargs={"encoding": "utf-8"},
            )

            folder_docs = loader.load()

            for doc in folder_docs:
                doc.metadata["doc_type"] = doc_type
                documents.append(doc)

        self.logger.info(f"Loaded {len(documents)} documents")
        return documents

    # ------------------------------------------------------------------
    # CHUNKING
    # ------------------------------------------------------------------
    def split_documents(self, documents: List) -> List:
        """
        Split documents into chunks for embedding.
        """
        self.logger.info("Splitting documents into chunks...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks = splitter.split_documents(documents)

        self.logger.info(f"Created {len(chunks)} chunks")
        return chunks

    # ------------------------------------------------------------------
    # VECTOR STORE CREATION
    # ------------------------------------------------------------------
    def build_vectorstore(self, chunks: List) -> Chroma:
        """
        Create and persist vector database.
        """
        self.logger.info("Building vector store...")

        # Delete existing DB if it exists
        if os.path.exists(self.db_path):
            self.logger.warning("Existing DB found. Deleting collection...")
            Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings,
            ).delete_collection()

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_path,
        )

        # Diagnostics
        collection = vectorstore._collection
        count = collection.count()

        sample_embedding = collection.get(
            limit=1, include=["embeddings"]
        )["embeddings"][0]

        dimensions = len(sample_embedding)

        self.logger.info(
            f"Vector store created with {count:,} vectors ({dimensions} dimensions)"
        )

        return vectorstore

    # ------------------------------------------------------------------
    # FULL PIPELINE
    # ------------------------------------------------------------------
    def build(self) -> Chroma:
        """
        Run full pipeline:
        1. Load documents
        2. Split into chunks
        3. Build vector store
        """
        self.logger.info("Starting full ingestion pipeline...")

        documents = self.load_documents()
        chunks = self.split_documents(documents)
        vectorstore = self.build_vectorstore(chunks)

        self.logger.info("Ingestion complete")
        return vectorstore


# ----------------------------------------------------------------------
# EXAMPLE USAGE
# ----------------------------------------------------------------------
if __name__ == "__main__":
    base_path = Path(__file__).parent.parent

    builder = VectorDBBuilder(
        knowledge_base_path=str(base_path / "knowledge-base"),
        db_path=str(base_path / "vector_db"),

        # 🔹 Change these to support other formats:
        glob_pattern="**/*.md",        # e.g. "**/*.pdf"
        loader_cls=TextLoader,        # e.g. PyPDFLoader

        # 🔹 Tune chunking:
        chunk_size=1000,
        chunk_overlap=200,
    )

    builder.build()