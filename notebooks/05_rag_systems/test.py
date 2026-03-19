"""
This module defines the structure and loading of evaluation test cases for a RAG system.

Each test case includes:
- A question to ask the system
- Expected keywords for retrieval evaluation
- A reference answer for answer quality evaluation
- A category label (for grouping/analysis)

Data is stored in a JSONL (JSON Lines) file, where:
- Each line is a separate JSON object
- This format is efficient for large datasets and streaming

NOTES FOR EXTENDING:

- Adding new fields:
  Extend the TestQuestion model (e.g., difficulty, source, tags)

- Validation:
  Pydantic ensures all fields are present and correctly typed

- Alternative formats:
  You could load from:
    * CSV (via pandas)
    * database (e.g., PostgreSQL)
    * API endpoint

- File location:
  TEST_FILE points to "tests.jsonl" in the same directory as this script
"""

import json
from pathlib import Path
from pydantic import BaseModel, Field


# Path to the JSONL file containing test cases
TEST_FILE = str(Path(__file__).parent / "tests.jsonl")


# ------------------------------------------------------------------
# DATA MODEL
# ------------------------------------------------------------------
class TestQuestion(BaseModel):
    """
    Represents a single evaluation test case.

    Fields:
    - question: Input to the RAG system
    - keywords: Expected terms that should appear in retrieved documents
    - reference_answer: Ground truth answer for evaluation
    - category: Used for grouping and analysis (e.g., direct_fact, reasoning)
    """

    # The question that will be sent to the RAG system
    question: str = Field(
        description="The question to ask the RAG system"
    )

    # Keywords expected to appear in retrieved context (for retrieval evaluation)
    keywords: list[str] = Field(
        description="Keywords that must appear in retrieved context"
    )

    # Reference answer used for evaluating generated answers
    reference_answer: str = Field(
        description="The reference answer for this question"
    )

    # Category for grouping tests (used in dashboards and analysis)
    category: str = Field(
        description="Question category (e.g., direct_fact, spanning, temporal)"
    )


# ------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------
def load_tests() -> list[TestQuestion]:
    """
    Load test questions from a JSONL file.

    Steps:
    1. Open the JSONL file
    2. Read each line (each line = one JSON object)
    3. Parse JSON into Python dict
    4. Convert dict into TestQuestion (Pydantic validation)
    5. Return list of test objects

    RETURNS:
        List[TestQuestion]

    NOTES:
    - JSONL format allows efficient streaming and large datasets
    - Invalid rows will raise validation errors via Pydantic
    - You may want to add error handling for robustness in production
    """
    tests = []

    # Open file in UTF-8 encoding to support all characters
    with open(TEST_FILE, "r", encoding="utf-8") as f:

        # Iterate over each line (each line is a JSON object)
        for line in f:
            # Remove whitespace and parse JSON
            data = json.loads(line.strip())

            # Convert dictionary into structured TestQuestion object
            # Pydantic validates fields and types here
            tests.append(TestQuestion(**data))

    return tests