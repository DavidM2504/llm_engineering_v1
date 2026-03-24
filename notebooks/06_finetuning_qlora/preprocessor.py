# =============================================================================
# Program: preprocessor.py
#
# PURPOSE:
#   Provides a lightweight, single-item alternative to the bulk batch processing
#   in batch.py. Given one product's raw text, it calls an AI model (via LiteLLM)
#   to produce a concise structured summary, and tracks running token usage and
#   cost. Suitable for small datasets, interactive testing in notebooks, or
#   pre-processing items one at a time.
#
# CONNECTIONS TO OTHER PROGRAMS:
#   - Accepts item.full (raw product text built by parsers.py / scrub()) as input
#   - Produces the same structured summary format that batch.py writes to
#     item.summary — both outputs are consumed identically by
#     deep_neural_network.py and evaluator.py
#   - Uses an identical SYSTEM_PROMPT to batch.py, so model outputs are
#     consistent regardless of which preprocessing path was used
#   - LiteLLM acts as a unified gateway, so the model name can be swapped to
#     any provider (OpenAI, Anthropic, Groq, etc.) without changing the code
# =============================================================================

from litellm import completion

# Default model and reasoning settings; can be overridden at instantiation time.
# The "groq/" prefix tells LiteLLM to route this request through the Groq API.
# This matches the model used in batch.py for consistent output quality.
DEFAULT_MODEL_NAME = "groq/openai/gpt-oss-20b"
DEFAULT_REASONING_EFFORT = "low"  # Low effort = faster and cheaper inference

# The system prompt instructs the model to produce a compact, structured summary.
# This is identical to the SYSTEM_PROMPT in batch.py, ensuring outputs from both
# preprocessing paths are interchangeable as inputs to deep_neural_network.py.
SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


class Preprocessor:
    """
    Wraps a single-item AI summarisation call with token and cost tracking.

    Usage:
        preprocessor = Preprocessor()
        summary = preprocessor.preprocess(item.full)   # item.full from parsers.py
        item.summary = summary                          # stored on Item (items.py)

    For large-scale processing, use Batch.run() in batch.py instead — it
    submits thousands of items in a single API call and is far more efficient.
    """

    def __init__(self, model_name=DEFAULT_MODEL_NAME, reasoning_effort=DEFAULT_REASONING_EFFORT):
        """
        model_name       : LiteLLM model string; swap to change provider/model
        reasoning_effort : passed to the API to control reasoning depth vs cost
        """
        # Cumulative token and cost counters — useful for estimating total spend
        # when processing large numbers of items one at a time
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort

    def messages_for(self, text: str) -> list[dict]:
        """
        Build the message list for a single API call.
        The system message contains the output format instructions (SYSTEM_PROMPT).
        The user message contains the raw product text (item.full from parsers.py).
        This two-message structure mirrors what batch.py sends in each JSONL line.
        """
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]

    def preprocess(self, text: str) -> str:
        """
        Send one product's raw text to the AI model and return the structured summary.

        Steps:
          1. Build the message list via messages_for()
          2. Call the model via LiteLLM (supports any provider through unified API)
          3. Accumulate token usage and cost from the response metadata
          4. Return the model's text response, which follows the structured format
             defined in SYSTEM_PROMPT and can be stored as item.summary

        The returned summary is in the same format as batch.py produces, so it can
        be used interchangeably by deep_neural_network.py as training/inference text.
        """
        messages = self.messages_for(text)
        response = completion(
            messages=messages,
            model=self.model_name,
            reasoning_effort=self.reasoning_effort
        )
        # Track cumulative usage for cost monitoring
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens
        self.total_cost += response._hidden_params["response_cost"]
        return response.choices[0].message.content  # The structured product summary
