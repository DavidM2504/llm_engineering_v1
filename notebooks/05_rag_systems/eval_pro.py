"""
This script evaluates a RAG (Retrieval-Augmented Generation) system.

It supports two types of evaluation:
1. Retrieval Evaluation:
   - MRR (Mean Reciprocal Rank)
   - nDCG (Normalized Discounted Cumulative Gain)
   - Keyword Coverage

2. Answer Evaluation (LLM-as-a-judge):
   - Accuracy
   - Completeness
   - Relevance

It can be used:
- Programmatically (via functions)
- In a dashboard (Gradio app)
- From CLI (single test evaluation)

NOTES FOR EXTENDING:

- Logging:
  Replace print() with logging for production:
    import logging
    logging.basicConfig(level=logging.INFO)

- Adding metrics:
  Extend RetrievalEval or AnswerEval models and update calculation functions.

- Improving evaluation:
  * Add semantic similarity scoring
  * Use cross-encoders for re-ranking
  * Improve judge prompt or use multiple judges

- Performance:
  Current implementation is sequential.
  Can be optimized with async or batching.
"""

import sys
import math
from pydantic import BaseModel, Field
from litellm import completion
from dotenv import load_dotenv

# Test data and RAG pipeline integration
from evaluation.test import TestQuestion, load_tests
from pro_implementation.answer import answer_question, fetch_context


# Load environment variables (API keys, etc.)
load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
db_name = "vector_db"


# ------------------------------------------------------------------
# DATA MODELS (STRUCTURED OUTPUTS)
# ------------------------------------------------------------------
class RetrievalEval(BaseModel):
    """
    Structured output for retrieval evaluation metrics.

    Using Pydantic ensures:
    - Validation
    - Typed outputs
    - Easy serialization (JSON, etc.)
    """

    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")


class AnswerEval(BaseModel):
    """
    Structured output for LLM-based answer evaluation.

    LLM acts as a "judge" and returns:
    - qualitative feedback
    - quantitative scores (1–5)
    """

    feedback: str = Field(
        description="Concise feedback comparing generated answer with reference answer"
    )
    accuracy: float = Field(
        description="Factual correctness (1 = wrong, 5 = perfect)"
    )
    completeness: float = Field(
        description="Coverage of all required information (1–5)"
    )
    relevance: float = Field(
        description="How well the answer addresses the question (1–5)"
    )


# ------------------------------------------------------------------
# RETRIEVAL METRICS
# ------------------------------------------------------------------
def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    """
    Calculate Reciprocal Rank for a single keyword.

    Logic:
    - Find first document containing keyword
    - Score = 1 / rank
    - If not found → 0

    Example:
        keyword found at rank 1 → 1.0
        keyword found at rank 3 → 0.33
    """
    keyword_lower = keyword.lower()

    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank

    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain (DCG).

    Higher-ranked documents contribute more to the score.

    Formula:
        relevance / log2(rank + 1)
    """
    dcg = 0.0

    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)  # rank starts at 1

    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
    """
    Calculate normalized DCG (nDCG) for a keyword.

    Steps:
    1. Assign binary relevance (1 if keyword present, else 0)
    2. Compute DCG
    3. Compute ideal DCG (best possible ranking)
    4. Normalize: DCG / IDCG
    """
    keyword_lower = keyword.lower()

    # Binary relevance vector
    relevances = [
        1 if keyword_lower in doc.page_content.lower() else 0
        for doc in retrieved_docs[:k]
    ]

    dcg = calculate_dcg(relevances, k)

    # Ideal ranking (sorted relevances)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


# ------------------------------------------------------------------
# RETRIEVAL EVALUATION
# ------------------------------------------------------------------
def evaluate_retrieval(test: TestQuestion, k: int = 10) -> RetrievalEval:
    """
    Evaluate retrieval performance for a single test question.

    Steps:
    1. Retrieve documents using RAG retriever
    2. Compute MRR across keywords
    3. Compute nDCG across keywords
    4. Compute keyword coverage

    Returns:
        RetrievalEval object
    """
    # Retrieve documents from vector store
    retrieved_docs = fetch_context(test.question)

    # MRR (average across keywords)
    mrr_scores = [
        calculate_mrr(keyword, retrieved_docs)
        for keyword in test.keywords
    ]
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    # nDCG (average across keywords)
    ndcg_scores = [
        calculate_ndcg(keyword, retrieved_docs, k)
        for keyword in test.keywords
    ]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    # Keyword coverage
    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(test.keywords)

    keyword_coverage = (
        (keywords_found / total_keywords) * 100
        if total_keywords > 0
        else 0.0
    )

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )


# ------------------------------------------------------------------
# ANSWER EVALUATION (LLM AS JUDGE)
# ------------------------------------------------------------------
def evaluate_answer(test: TestQuestion) -> tuple[AnswerEval, str, list]:
    """
    Evaluate answer quality using an LLM as a judge.

    Steps:
    1. Generate answer using RAG pipeline
    2. Send generated + reference answer to LLM
    3. Ask LLM to score quality
    4. Parse structured output into AnswerEval

    Returns:
        (AnswerEval, generated_answer, retrieved_docs)
    """
    # Generate answer using RAG system
    generated_answer, retrieved_docs = answer_question(test.question)

    # Build evaluation prompt
    judge_messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator assessing answer quality.",
        },
        {
            "role": "user",
            "content": f"""Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Evaluate on:
1. Accuracy
2. Completeness
3. Relevance

Score 1–5. Only give 5 if perfect.""",
        },
    ]

    # Call LLM with structured output (Pydantic model)
    judge_response = completion(
        model=MODEL,
        messages=judge_messages,
        response_format=AnswerEval
    )

    # Parse response into structured object
    answer_eval = AnswerEval.model_validate_json(
        judge_response.choices[0].message.content
    )

    return answer_eval, generated_answer, retrieved_docs


# ------------------------------------------------------------------
# BATCH EVALUATION (GENERATORS)
# ------------------------------------------------------------------
def evaluate_all_retrieval():
    """
    Iterate over all retrieval tests.

    Yields:
        (test, result, progress)
    Useful for:
        - dashboards
        - progress tracking
    """
    tests = load_tests()
    total_tests = len(tests)

    for index, test in enumerate(tests):
        result = evaluate_retrieval(test)
        progress = (index + 1) / total_tests
        yield test, result, progress


def evaluate_all_answers():
    """
    Iterate over all answer evaluations.

    NOTE:
    Currently sequential — can be optimized with async batching.
    """
    tests = load_tests()
    total_tests = len(tests)

    for index, test in enumerate(tests):
        result = evaluate_answer(test)[0]
        progress = (index + 1) / total_tests
        yield test, result, progress


# ------------------------------------------------------------------
# CLI EVALUATION
# ------------------------------------------------------------------
def run_cli_evaluation(test_number: int):
    """
    Run evaluation for a single test via CLI.

    Displays:
    - Test details
    - Retrieval metrics
    - Generated answer
    - LLM evaluation scores
    """
    tests = load_tests("tests.jsonl")

    # Validate input
    if test_number < 0 or test_number >= len(tests):
        print(f"Error: test_row_number must be between 0 and {len(tests) - 1}")
        sys.exit(1)

    test = tests[test_number]

    # Print test info
    print(f"\n{'=' * 80}")
    print(f"Test #{test_number}")
    print(f"{'=' * 80}")
    print(f"Question: {test.question}")
    print(f"Keywords: {test.keywords}")
    print(f"Category: {test.category}")
    print(f"Reference Answer: {test.reference_answer}")

    # ---------------- RETRIEVAL ----------------
    print(f"\n{'=' * 80}")
    print("Retrieval Evaluation")
    print(f"{'=' * 80}")

    retrieval_result = evaluate_retrieval(test)

    print(f"MRR: {retrieval_result.mrr:.4f}")
    print(f"nDCG: {retrieval_result.ndcg:.4f}")
    print(f"Keywords Found: {retrieval_result.keywords_found}/{retrieval_result.total_keywords}")
    print(f"Keyword Coverage: {retrieval_result.keyword_coverage:.1f}%")

    # ---------------- ANSWER ----------------
    print(f"\n{'=' * 80}")
    print("Answer Evaluation")
    print(f"{'=' * 80}")

    answer_result, generated_answer, _ = evaluate_answer(test)

    print(f"\nGenerated Answer:\n{generated_answer}")
    print(f"\nFeedback:\n{answer_result.feedback}")
    print("\nScores:")
    print(f"  Accuracy: {answer_result.accuracy:.2f}/5")
    print(f"  Completeness: {answer_result.completeness:.2f}/5")
    print(f"  Relevance: {answer_result.relevance:.2f}/5")


# ------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------
def main():
    """
    CLI entry point.

    Usage:
        uv run eval.py <test_row_number>
    """
    if len(sys.argv) != 2:
        print("Usage: uv run eval.py <test_row_number>")
        sys.exit(1)

    try:
        test_number = int(sys.argv[1])
    except ValueError:
        print("Error: test_row_number must be an integer")
        sys.exit(1)

    run_cli_evaluation(test_number)


if __name__ == "__main__":
    main()