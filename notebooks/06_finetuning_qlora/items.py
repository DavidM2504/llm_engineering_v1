# =============================================================================
# Program: items.py
#
# PURPOSE:
#   Defines the central Item data model used throughout the entire pipeline.
#   An Item represents one product with its title, category, price, and several
#   optional fields that are populated at different stages of the pipeline.
#   Also provides helpers for pushing/loading datasets from HuggingFace Hub.
#
# CONNECTIONS TO OTHER PROGRAMS:
#   - parsers.py creates Item objects from raw Amazon dataset datapoints
#   - loaders.py uses parsers.py to produce lists of Items
#   - batch.py and preprocessor.py populate item.summary with AI-generated text
#   - deep_neural_network.py reads item.summary (text) and item.price (label)
#     for training and inference
#   - evaluator.py reads item.title and item.price for evaluation reporting
#   - The push_to_hub / from_hub methods allow the processed dataset to be
#     shared and reloaded without re-running the expensive loaders pipeline
# =============================================================================

from pydantic import BaseModel
from datasets import Dataset, DatasetDict, load_dataset
from typing import Optional, Self

# Template strings used to build the fine-tuning prompt for LLM-based pricing
PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"


class Item(BaseModel):
    """
    Central data model for one product in the pricing pipeline.

    Fields and their lifecycle:
      title     : product title (from raw Amazon data via parsers.py)
      category  : Amazon product category (from loaders.py)
      price     : ground-truth price in USD (from parsers.py, validated in range)
      full      : cleaned raw product text (set by parsers.py / scrub()),
                  sent to the AI summariser in batch.py / preprocessor.py
      weight    : item weight in pounds (from parsers.py / get_weight()),
                  available as an optional feature for downstream models
      summary   : AI-generated structured summary (set by batch.py or
                  preprocessor.py), used as input text by deep_neural_network.py
      prompt    : formatted fine-tuning prompt (set by make_prompt()),
                  used when fine-tuning an LLM directly on the pricing task
      id        : integer index assigned during dataset preparation, used by
                  batch.py to map batch results back to the correct Item
    """

    title: str
    category: str
    price: float
    full: Optional[str] = None      # Raw product text (input to summariser)
    weight: Optional[float] = None  # Item weight in pounds
    summary: Optional[str] = None   # AI summary (output of batch.py / preprocessor.py)
    prompt: Optional[str] = None    # LLM fine-tuning prompt
    id: Optional[int] = None        # Index for batch result mapping (batch.py)

    # -------------------------------------------------------------------------
    # LLM fine-tuning prompt helpers
    # -------------------------------------------------------------------------

    def make_prompt(self, text: str):
        """
        Build a fine-tuning prompt in the format:
            "What does this cost to the nearest dollar?

            <text>

            Price is $<price>.00"

        The text parameter is typically item.summary (from batch.py) or
        item.full (raw text from parsers.py).
        The prompt is stored on item.prompt for serialisation to HuggingFace Hub.
        """
        self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}{round(self.price)}.00"

    def test_prompt(self) -> str:
        """
        Return the prompt up to (but not including) the answer portion.
        Used during inference/evaluation to feed only the question to the model,
        leaving the price for it to predict.

        Example:
            Full prompt : "What does this cost?\n\n...\n\nPrice is $25.00"
            Test prompt : "What does this cost?\n\n...\n\nPrice is $"
        """
        return self.prompt.split(PREFIX)[0] + PREFIX

    def __repr__(self) -> str:
        return f"<{self.title} = ${self.price}>"

    # -------------------------------------------------------------------------
    # HuggingFace Hub I/O
    # -------------------------------------------------------------------------

    @staticmethod
    def push_to_hub(dataset_name: str, train: list[Self], val: list[Self], test: list[Self]):
        """
        Serialise train / validation / test Item lists and push them to
        HuggingFace Hub as a DatasetDict. This allows the processed dataset
        (with summaries already generated) to be shared or reloaded later
        without re-running loaders.py and batch.py.

        Each Item is converted to a plain dict via Pydantic's model_dump().
        """
        DatasetDict(
            {
                "train": Dataset.from_list([item.model_dump() for item in train]),
                "validation": Dataset.from_list([item.model_dump() for item in val]),
                "test": Dataset.from_list([item.model_dump() for item in test]),
            }
        ).push_to_hub(dataset_name)

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        """
        Load a previously pushed DatasetDict from HuggingFace Hub and
        reconstruct the three Item lists (train, validation, test).

        Pydantic's model_validate() is used instead of plain __init__() so that
        validation and type coercion are applied during deserialization.

        Returns:
            (train_items, val_items, test_items)
        """
        ds = load_dataset(dataset_name)
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )