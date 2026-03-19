"""
Document Chunking and Embedding Pipeline for Insurellm Knowledge Base

This module processes company documents into embeddings for RAG systems:

1. Fetch documents from a local knowledge base
2. Split documents into overlapping chunks with headlines and summaries
3. Generate embeddings for chunks and store them in a persistent Chroma DB
4. Supports parallel processing for efficiency

Production Notes:
- Uses tenacity retry for robust LLM calls
- Uses Pydantic models for structured output
- WORKERS can be adjusted to avoid API rate limits
- Overlapping chunks improve retrieval performance in downstream RAG tasks
"""

from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from multiprocessing import Pool
from tenacity import retry, wait_exponential

# ------------------------------------------------------------------
# ENVIRONMENT SETUP
# ------------------------------------------------------------------
load_dotenv(override=True)  # Load .env variables

# LLM model for chunking
MODEL = "openai/gpt-4.1-nano"

# Paths and constants
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
collection_name = "docs"
embedding_model = "text-embedding-3-large"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"

AVERAGE_CHUNK_SIZE = 100  # Approx. number of characters per chunk for prompt guidance
WORKERS = 3               # Number of parallel processes for chunk creation
wait = wait_exponential(multiplier=1, min=10, max=240)  # Retry backoff

# Initialize OpenAI client
openai = OpenAI()

# ------------------------------------------------------------------
# DATA MODELS
# ------------------------------------------------------------------
class Result(BaseModel):
    """Represents a processed document chunk ready for embedding."""
    page_content: str
    metadata: dict


class Chunk(BaseModel):
    """Represents a single chunk generated from a document."""
    headline: str = Field(
        description="Brief heading summarizing the chunk"
    )
    summary: str = Field(
        description="Short summary capturing key points of the chunk"
    )
    original_text: str = Field(
        description="Exact original text from the document"
    )

    def as_result(self, document):
        """
        Convert Chunk into Result format for embedding and vectorstore ingestion.
        Adds metadata for source and document type.
        """
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
        )


class Chunks(BaseModel):
    """Wrapper for a list of Chunk objects returned by the LLM."""
    chunks: list[Chunk]

# ------------------------------------------------------------------
# DOCUMENT FETCHING
# ------------------------------------------------------------------
def fetch_documents():
    """
    Fetch all Markdown documents from the knowledge base.
    Mimics LangChain DirectoryLoader.
    
    Returns:
        List of dicts with 'type', 'source', and 'text'
    """
    documents = []

    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                documents.append({
                    "type": doc_type,
                    "source": file.as_posix(),
                    "text": f.read()
                })

    print(f"Loaded {len(documents)} documents")
    return documents

# ------------------------------------------------------------------
# PROMPT CONSTRUCTION
# ------------------------------------------------------------------
def make_prompt(document):
    """
    Construct a prompt for LLM to chunk a document.
    
    Includes:
    - Document type and source
    - Recommended number of chunks
    - Instructions for overlap (~25%)
    - Request for headline, summary, original text for each chunk
    """
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

Document type: {document["type"]}
Source: {document["source"]}

Split into at least {how_many} chunks with overlap (~25% or ~50 words).

For each chunk, provide:
- Headline
- Summary
- Original text

Document content:
{document["text"]}

Respond with the chunks.
"""


def make_messages(document):
    """Convert a document into messages suitable for LLM completion."""
    return [{"role": "user", "content": make_prompt(document)}]

# ------------------------------------------------------------------
# DOCUMENT PROCESSING
# ------------------------------------------------------------------
@retry(wait=wait)
def process_document(document):
    """
    Process a single document with LLM to generate chunks.

    Returns:
        List[Result]: chunked and metadata-enriched results
    """
    messages = make_messages(document)
    response = completion(model=MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]

# ------------------------------------------------------------------
# CHUNK CREATION (PARALLEL)
# ------------------------------------------------------------------
def create_chunks(documents):
    """
    Create chunks for all documents using multiple workers.

    Notes:
    - Adjust WORKERS to 1 to avoid rate-limiting
    - Uses tqdm for progress bar
    """
    chunks = []
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            chunks.extend(result)
    return chunks

# ------------------------------------------------------------------
# EMBEDDING CREATION
# ------------------------------------------------------------------
def create_embeddings(chunks):
    """
    Generate embeddings for each chunk and store them in Chroma.

    Steps:
    1. Delete existing collection if it exists
    2. Generate embeddings for all chunk texts
    3. Add documents, embeddings, and metadata to collection
    """
    chroma = PersistentClient(path=DB_NAME)

    # Delete old collection if present
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    # Generate embeddings
    texts = [chunk.page_content for chunk in chunks]
    emb = openai.embeddings.create(model=embedding_model, input=texts).data
    vectors = [e.embedding for e in emb]

    # Create new collection
    collection = chroma.get_or_create_collection(collection_name)

    # Assign IDs and metadata
    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]

    # Add to vectorstore
    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")

# ------------------------------------------------------------------
# MAIN SCRIPT
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: Load documents
    documents = fetch_documents()

    # Step 2: Generate chunks using LLM
    chunks = create_chunks(documents)

    # Step 3: Generate embeddings and store in Chroma
    create_embeddings(chunks)
    print("Ingestion complete")