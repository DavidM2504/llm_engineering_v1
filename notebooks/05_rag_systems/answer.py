"""
This script implements a Retrieval-Augmented Generation (RAG) pipeline.

Steps performed:
1. Load a persisted Chroma vector database.
2. Convert it into a retriever to fetch relevant documents.
3. Combine user question with conversation history.
4. Retrieve top-K relevant documents as context.
5. Inject context into a system prompt.
6. Send the prompt + conversation to an LLM (ChatOpenAI).
7. Return the generated answer along with source documents.

NOTES FOR EXTENDING THIS SCRIPT:

- Adding other embedding models:
  Replace HuggingFaceEmbeddings with:
    OpenAIEmbeddings(model="text-embedding-3-large")
  Or plug in any LangChain-compatible embedding model.

- Using logging instead of print/debugging:
  Add:
    import logging
    logging.basicConfig(level=logging.INFO)
  Then replace debug/print statements with:
    logging.info("message")

- Converting into a reusable class/module:
  Wrap logic into a class (e.g., RAGPipeline) with methods like:
    - retrieve_context()
    - build_prompt()
    - generate_answer()
  Pass dependencies (retriever, llm, config) via __init__ for flexibility.

- Improving retrieval:
  You can:
    * Adjust RETRIEVAL_K
    * Use similarity_score_threshold
    * Add re-ranking (e.g., cross-encoders)
"""

from pathlib import Path

# LLM and embedding providers
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Vector database
from langchain_chroma import Chroma

# Message and document handling
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

# Environment variable loader (for API keys, etc.)
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv(override=True)

# Model configuration
MODEL = "gpt-4.1-nano"

# Path to persisted vector database
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

# Initialize embedding model (must match the one used during ingestion)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Optional: Use OpenAI embeddings instead
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Number of documents to retrieve for context
RETRIEVAL_K = 3

# System prompt template (context will be injected dynamically)
SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

# Initialize vector store from persisted database
vectorstore = Chroma(
    persist_directory=DB_NAME,
    embedding_function=embeddings
)

# Convert vector store into a retriever interface
retriever = vectorstore.as_retriever()

# Initialize LLM (temperature=0 for deterministic responses)
llm = ChatOpenAI(
    temperature=0,
    model_name=MODEL
)


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.

    NOTES:
    - Uses vector similarity search via Chroma retriever
    - RETRIEVAL_K controls how many documents are returned
    - You can extend this with:
        * score thresholds
        * metadata filtering
        * hybrid search (keyword + vector)
    """
    return retriever.invoke(question, k=RETRIEVAL_K)


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all previous user messages into a single query string.

    WHY:
    - Improves retrieval quality by adding conversational context
    - Helps resolve follow-up questions like "What about pricing?"

    NOTE:
    - Only user messages are included (not assistant responses)
    - You could improve this by:
        * limiting history length
        * summarizing long conversations
    """
    prior = "\n".join(
        m["content"] for m in history if m["role"] == "user"
    )
    return prior + "\n" + question


def answer_question(
    question: str,
    history: list[dict] = []
) -> tuple[str, list[Document]]:
    """
    Answer a question using Retrieval-Augmented Generation (RAG).

    Steps:
    1. Combine question with conversation history
    2. Retrieve relevant documents
    3. Build context string from documents
    4. Inject context into system prompt
    5. Send messages to LLM
    6. Return answer + source documents

    LOGGING NOTE:
    - Replace intermediate debugging with logging:
        logging.info(f"Retrieved {len(docs)} docs")
        logging.debug(context)

    RETURNS:
    - answer (str): Generated response from LLM
    - docs (list[Document]): Source documents used for context
    """
    # Combine current question with prior conversation
    combined = combined_question(question, history)

    # Retrieve relevant documents from vector store
    docs = fetch_context(combined)

    # Build context string from retrieved documents
    context = "\n\n".join(doc.page_content for doc in docs)

    # Inject context into system prompt
    system_prompt = SYSTEM_PROMPT.format(context=context)

    # Create message list starting with system instruction
    messages = [SystemMessage(content=system_prompt)]

    # Add previous conversation history (converted to LangChain format)
    messages.extend(convert_to_messages(history))

    # Add current user question
    messages.append(HumanMessage(content=question))

    # Call LLM to generate response
    response = llm.invoke(messages)

    return response.content, docs


# REUSABLE MODULE / CLASS NOTE:
# --------------------------------
# To convert this into a reusable class:
#
# class RAGPipeline:
#     def __init__(self, retriever, llm, system_prompt):
#         self.retriever = retriever
#         self.llm = llm
#         self.system_prompt = system_prompt
#
#     def retrieve_context(self, question):
#         ...
#
#     def generate_answer(self, question, history):
#         ...
#
# Benefits:
# - Easier to reuse in APIs (FastAPI, Flask)
# - Easier to test (mock retriever/LLM)
# - Cleaner dependency injection
#
# You can then import and use it like:
#   rag = RAGPipeline(retriever, llm, SYSTEM_PROMPT)
#   rag.generate_answer("question")