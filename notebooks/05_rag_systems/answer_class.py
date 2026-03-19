"""
RAGPipeline: Production-ready Retrieval-Augmented Generation pipeline.

Features:
- Uses Chroma vector DB for retrieval
- Supports conversation history
- Streaming + non-streaming responses
- Clean logging (no print statements)
- Fully reusable and configurable

NOTES:

- Streaming:
  Use stream_answer() to yield tokens in real-time (useful for chat UIs).

- Logging:
  Uses Python logging instead of print for better production control.

- Extensibility:
  You can swap:
    * retriever (e.g., Pinecone, Weaviate)
    * LLM (OpenAI, local models)
    * embedding models

- Improvements you can add:
    * re-ranking
    * metadata filtering
    * caching
"""

import logging
from pathlib import Path
from typing import List, Tuple, Generator, Optional

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    convert_to_messages,
)
from langchain_core.documents import Document


class RAGPipeline:
    def __init__(
        self,
        db_path: str,
        model_name: str = "gpt-4.1-nano",
        embedding_model: str = "all-MiniLM-L6-v2",
        retrieval_k: int = 3,
        temperature: float = 0.0,
        use_openai_embeddings: bool = False,
        log_level: int = logging.INFO,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            db_path: Path to persisted Chroma DB
            model_name: LLM model name
            embedding_model: HuggingFace embedding model
            retrieval_k: Number of documents to retrieve
            temperature: LLM temperature
            use_openai_embeddings: Switch embedding provider
            log_level: Logging level
        """

        # Setup logging
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # Load environment variables (API keys, etc.)
        load_dotenv(override=True)

        self.retrieval_k = retrieval_k

        # Initialize embeddings
        if use_openai_embeddings:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large"
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model
            )

        # Load vector store
        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings,
        )

        self.retriever = self.vectorstore.as_retriever()

        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=temperature,
            model_name=model_name,
        )

        # System prompt template
        self.system_prompt_template = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.

Context:
{context}
"""

    # ------------------------------------------------------------------
    # CONTEXT RETRIEVAL
    # ------------------------------------------------------------------
    def retrieve_context(self, question: str) -> List[Document]:
        """
        Retrieve top-K relevant documents.
        """
        self.logger.info("Retrieving context...")
        docs = self.retriever.invoke(question, k=self.retrieval_k)
        self.logger.info(f"Retrieved {len(docs)} documents")
        return docs

    # ------------------------------------------------------------------
    # QUESTION COMBINATION
    # ------------------------------------------------------------------
    def combine_question(
        self,
        question: str,
        history: List[dict]
    ) -> str:
        """
        Combine conversation history with the current question.
        """
        prior = "\n".join(
            m["content"] for m in history if m["role"] == "user"
        )
        return prior + "\n" + question

    # ------------------------------------------------------------------
    # PROMPT BUILDING
    # ------------------------------------------------------------------
    def build_messages(
        self,
        question: str,
        docs: List[Document],
        history: List[dict],
    ):
        """
        Construct message list for LLM.
        """
        context = "\n\n".join(doc.page_content for doc in docs)

        system_prompt = self.system_prompt_template.format(
            context=context
        )

        messages = [SystemMessage(content=system_prompt)]
        messages.extend(convert_to_messages(history))
        messages.append(HumanMessage(content=question))

        return messages

    # ------------------------------------------------------------------
    # NON-STREAMING RESPONSE
    # ------------------------------------------------------------------
    def answer(
        self,
        question: str,
        history: Optional[List[dict]] = None,
    ) -> Tuple[str, List[Document]]:
        """
        Generate a full response (non-streaming).
        """
        history = history or []

        combined = self.combine_question(question, history)
        docs = self.retrieve_context(combined)

        messages = self.build_messages(question, docs, history)

        self.logger.info("Generating response...")
        response = self.llm.invoke(messages)

        return response.content, docs

    # ------------------------------------------------------------------
    # STREAMING RESPONSE
    # ------------------------------------------------------------------
    def stream_answer(
        self,
        question: str,
        history: Optional[List[dict]] = None,
    ) -> Generator[str, None, Tuple[str, List[Document]]]:
        """
        Stream response token-by-token.

        Yields:
            tokens (str): Partial response chunks

        Returns (at end):
            full_response (str), docs (List[Document])
        """
        history = history or []

        combined = self.combine_question(question, history)
        docs = self.retrieve_context(combined)

        messages = self.build_messages(question, docs, history)

        self.logger.info("Streaming response...")

        full_response = ""

        # Stream tokens from LLM
        for chunk in self.llm.stream(messages):
            token = chunk.content or ""
            full_response += token
            yield token  # 🔥 real-time output

        # Final return (accessible via StopIteration.value if needed)
        return full_response, docs


# ----------------------------------------------------------------------
# EXAMPLE USAGE
# ----------------------------------------------------------------------
if __name__ == "__main__":
    base_path = Path(__file__).parent.parent

    rag = RAGPipeline(
        db_path=str(base_path / "vector_db"),
        retrieval_k=3,
    )

    question = "What does Insurellm do?"

    # ✅ Non-streaming
    answer, docs = rag.answer(question)
    print("\nFinal Answer:\n", answer)

    # ✅ Streaming (example)
    print("\nStreaming Answer:\n")
    for token in rag.stream_answer(question):
        print(token, end="", flush=True)