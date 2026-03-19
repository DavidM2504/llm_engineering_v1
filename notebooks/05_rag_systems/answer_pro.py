"""
RAG (Retrieval-Augmented Generation) Pipeline with Re-ranking and Query Rewriting

This module provides a full RAG pipeline for answering questions using:
1. Embedding-based retrieval from a Chroma persistent database
2. Query rewriting to improve relevance
3. Re-ranking of retrieved chunks with an LLM
4. Generating answers from retrieved and re-ranked context

Key Components:
- fetch_context: Retrieve relevant documents from the KB
- rewrite_query: Convert user questions into more effective KB queries
- rerank: Re-order retrieved chunks using LLM relevance scoring
- answer_question: Generate final answer using context and conversation history
- make_rag_messages: Format system + user messages for LLM input

Production Notes:
- Uses tenacity retry logic for robustness
- Pydantic models enforce structured output
- Constants like RETRIEVAL_K, FINAL_K control retrieval and answer quality
- SYSTEM_PROMPT ensures consistent assistant behavior
"""

import json
from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field
from pathlib import Path
from tenacity import retry, wait_exponential

# ------------------------------------------------------------------
# ENVIRONMENT SETUP
# ------------------------------------------------------------------
load_dotenv(override=True)

# LLM model to use for both query rewriting and answer generation
MODEL = "openai/gpt-4.1-nano"
# Alternative: MODEL = "groq/openai/gpt-oss-120b"

# Paths
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
SUMMARIES_PATH = Path(__file__).parent.parent / "summaries"

# Chroma collection & embedding setup
collection_name = "docs"
embedding_model = "text-embedding-3-large"

# Retry settings for robust API calls
wait = wait_exponential(multiplier=1, min=10, max=240)

# Initialize clients
openai = OpenAI()
chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(collection_name)

# Retrieval settings
RETRIEVAL_K = 20  # Number of top results to fetch initially
FINAL_K = 10      # Number of chunks to return after re-ranking

# System prompt for the assistant
SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""

# ------------------------------------------------------------------
# DATA MODELS
# ------------------------------------------------------------------
class Result(BaseModel):
    """Represents a retrieved document chunk."""
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    """Structured output for chunk re-ranking."""
    order: list[int] = Field(
        description="Order of chunk ids from most relevant to least relevant"
    )


# ------------------------------------------------------------------
# RERANKING
# ------------------------------------------------------------------
@retry(wait=wait)
def rerank(question, chunks):
    """
    Re-order retrieved chunks using an LLM based on relevance to the question.

    Steps:
    1. Provide the LLM with all chunks (with IDs) and the question
    2. Ask LLM to return only a list of ranked chunk IDs
    3. Map IDs back to chunks

    Returns:
        List[Result]: Chunks ordered by relevance
    """
    system_prompt = """
You are a document re-ranker.
Rank the provided chunks by relevance to the question. Reply only with the list of chunk IDs.
"""
    user_prompt = f"The user asked:\n{question}\n\nChunks:\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = completion(model=MODEL, messages=messages, response_format=RankOrder)
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    return [chunks[i - 1] for i in order]


# ------------------------------------------------------------------
# MESSAGE FORMATTING
# ------------------------------------------------------------------
def make_rag_messages(question, history, chunks):
    """
    Construct messages for LLM input.

    Combines:
    - System prompt with context
    - Conversation history
    - User's current question
    """
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


# ------------------------------------------------------------------
# QUERY REWRITING
# ------------------------------------------------------------------
@retry(wait=wait)
def rewrite_query(question, history=[]):
    """
    Rewrite user's question into a more specific Knowledge Base query.

    Ensures:
    - More precise retrieval
    - Better relevance ranking
    """
    message = f"""
You are in a conversation with a user about Insurellm.
History: {history}
Current question: {question}
Respond ONLY with a short, precise query for the KB.
"""
    response = completion(model=MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content


# ------------------------------------------------------------------
# MERGING AND FETCHING CONTEXT
# ------------------------------------------------------------------
def merge_chunks(chunks, reranked):
    """
    Merge original and reranked chunks, avoiding duplicates.

    Ensures that new relevant chunks are added while keeping existing ones.
    """
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged


def fetch_context_unranked(question):
    """
    Fetch top-K chunks based on embedding similarity.

    Returns:
        List[Result]: Chunks in unranked order
    """
    query = openai.embeddings.create(model=embedding_model, input=[question]).data[0].embedding
    results = collection.query(query_embeddings=[query], n_results=RETRIEVAL_K)
    chunks = []
    for result in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=result[0], metadata=result[1]))
    return chunks


def fetch_context(original_question):
    """
    Full context retrieval pipeline:
    1. Rewrite question
    2. Fetch unranked results for original & rewritten queries
    3. Merge and rerank
    4. Return top FINAL_K chunks
    """
    rewritten_question = rewrite_query(original_question)
    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)
    chunks = merge_chunks(chunks1, chunks2)
    reranked = rerank(original_question, chunks)
    return reranked[:FINAL_K]


# ------------------------------------------------------------------
# RAG ANSWERING
# ------------------------------------------------------------------
@retry(wait=wait)
def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
    """
    Generate an answer using RAG.

    Steps:
    1. Fetch context chunks
    2. Format system + user messages
    3. Call LLM completion
    4. Return answer + retrieved chunks
    """
    chunks = fetch_context(question)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks