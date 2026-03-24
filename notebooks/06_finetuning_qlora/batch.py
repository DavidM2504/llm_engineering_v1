# =============================================================================
# Program: batch.py
#
# PURPOSE:
#   Handles large-scale batch processing of product data using the Groq API.
#   Instead of calling the AI model one item at a time (which would be slow and
#   expensive), this program groups items into batches of 1,000 and submits them
#   together for asynchronous processing. It then polls for completion and
#   retrieves the results.
#
# CONNECTIONS TO OTHER PROGRAMS:
#   - Uses Item objects from items.py (each item has a .full and .summary field)
#   - The AI-generated summaries written to item.summary are later consumed by
#     deep_neural_network.py (as training text) and evaluator.py (for inference)
#   - preprocessor.py does the same summarization job but one item at a time
#     (suitable for small runs); batch.py is the scalable alternative for the
#     full dataset
# =============================================================================

import os
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path
import json
import pickle
from tqdm.notebook import tqdm

# Load environment variables (e.g. GROQ_API_KEY) from a .env file
load_dotenv(override=True)
groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Model to use for summarization; reasoning_effort="low" keeps costs down
MODEL = "openai/gpt-oss-20b"

# Folder names for storing batch request files and their outputs
BATCHES_FOLDER = "batches"
OUTPUT_FOLDER = "output"

# File used to persist batch state between sessions (so work isn't lost if the
# notebook kernel restarts while waiting for Groq to finish processing)
state = Path("batches.pkl")

# The system prompt sent to the model with every item.
# It instructs the model to produce a compact, structured product summary.
# This same prompt is mirrored in preprocessor.py for single-item processing.
SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


class Batch:
    """
    Represents a single batch of up to BATCH_SIZE items.
    Manages the full lifecycle of one batch: writing the JSONL request file,
    uploading it to Groq, submitting the batch job, polling for completion,
    downloading results, and writing the AI summary back onto each Item object.
    """

    # Maximum number of items per batch (Groq's batch API limit)
    BATCH_SIZE = 1_000

    # Class-level list shared across all Batch instances — holds every batch
    # created during a run. Class methods (create, run, fetch, save, load)
    # operate on this list.
    batches = []

    def __init__(self, items, start, end, lite):
        """
        items : full list of Item objects (shared reference across all batches)
        start : index of the first item in this batch
        end   : index one past the last item in this batch
        lite  : if True, write files under a "lite/" subfolder (small test run);
                otherwise use "full/" (production run)
        """
        self.items = items
        self.start = start
        self.end = end
        self.filename = f"{start}_{end}.jsonl"  # Unique filename per batch slice

        # IDs assigned by Groq after each API call
        self.file_id = None        # ID of the uploaded JSONL file
        self.batch_id = None       # ID of the submitted batch job
        self.output_file_id = None # ID of the result file once the job completes

        self.done = False  # Flipped to True once output has been applied to items

        # Resolve output paths depending on whether this is a lite or full run
        folder = Path("lite") if lite else Path("full")
        self.batches = folder / BATCHES_FOLDER
        self.output = folder / OUTPUT_FOLDER
        self.batches.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Batch file construction
    # -------------------------------------------------------------------------

    def make_jsonl(self, item):
        """
        Serialize one Item into a single JSONL line in the format required by
        the Groq batch API. Each line contains:
          - custom_id : the item's database ID so we can match results later
          - method/url: the API endpoint to call for each request
          - body      : the model, messages, and reasoning_effort settings

        The item.full field (raw scraped product text from parser.py) is sent
        as the user message; the model returns a structured summary.
        """
        body = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item.full},  # Raw product text from items.py / parsers.py
            ],
            "reasoning_effort": "low",  # Low effort = faster and cheaper
        }
        line = {
            "custom_id": str(item.id),  # Used to map results back to the right Item
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        return json.dumps(line)

    def make_file(self):
        """Write the JSONL batch request file to disk (one line per item)."""
        batch_file = self.batches / self.filename
        with batch_file.open("w") as f:
            for item in self.items[self.start : self.end]:
                f.write(self.make_jsonl(item))
                f.write("\n")

    # -------------------------------------------------------------------------
    # Groq API interaction
    # -------------------------------------------------------------------------

    def send_file(self):
        """Upload the JSONL file to Groq and store the returned file_id."""
        batch_file = self.batches / self.filename
        with batch_file.open("rb") as f:
            response = groq.files.create(file=f, purpose="batch")
        self.file_id = response.id

    def submit_batch(self):
        """
        Tell Groq to process the uploaded file as a batch job.
        The 24h completion window means Groq can schedule it flexibly
        (and at lower cost than synchronous calls).
        """
        response = groq.batches.create(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=self.file_id,
        )
        self.batch_id = response.id

    def is_ready(self):
        """
        Poll Groq for the batch status. Returns True only when the job has
        completed. On completion, also captures the output_file_id needed to
        download results.
        """
        response = groq.batches.retrieve(self.batch_id)
        status = response.status
        if status == "completed":
            self.output_file_id = response.output_file_id
        return status == "completed"

    # -------------------------------------------------------------------------
    # Result retrieval and application
    # -------------------------------------------------------------------------

    def fetch_output(self):
        """Download the completed batch results file from Groq to local disk."""
        output_file = str(self.output / self.filename)
        response = groq.files.content(self.output_file_id)
        response.write_to_file(output_file)

    def apply_output(self):
        """
        Parse the downloaded JSONL results and write each AI-generated summary
        back onto the corresponding Item object (item.summary).

        The custom_id in each result line maps to the item's id, so we can
        update the right item even if results arrive out of order.

        After this step, item.summary is populated and ready to be used by
        deep_neural_network.py as training/inference input text.
        """
        output_file = str(self.output / self.filename)
        with open(output_file, "r") as f:
            for line in f:
                json_line = json.loads(line)
                id = int(json_line["custom_id"])
                summary = json_line["response"]["body"]["choices"][0]["message"]["content"]
                self.items[id].summary = summary  # Write summary back to Item (see items.py)
        self.done = True

    # -------------------------------------------------------------------------
    # Class-level orchestration methods
    # -------------------------------------------------------------------------

    @classmethod
    def create(cls, items, lite):
        """
        Slice the full item list into chunks of BATCH_SIZE and instantiate a
        Batch object for each chunk. Call this once before calling run().
        """
        for start in range(0, len(items), cls.BATCH_SIZE):
            end = min(start + cls.BATCH_SIZE, len(items))
            batch = Batch(items, start, end, lite)
            cls.batches.append(batch)
        print(f"Created {len(cls.batches)} batches")

    @classmethod
    def run(cls):
        """
        Submit all batches to Groq:
          1. Write each JSONL file to disk
          2. Upload it to Groq
          3. Submit the batch job
        After this returns, you typically wait (minutes to hours) and then
        call fetch() to collect results.
        """
        for batch in tqdm(cls.batches):
            batch.make_file()
            batch.send_file()
            batch.submit_batch()
        print(f"Submitted {len(cls.batches)} batches")

    @classmethod
    def fetch(cls):
        """
        Check every pending batch for completion. For each batch that is ready,
        download its results and apply them to the Item objects.
        Call this repeatedly (e.g. in a polling loop) until all batches are done.
        """
        for batch in tqdm(cls.batches):
            if not batch.done:
                if batch.is_ready():
                    batch.fetch_output()
                    batch.apply_output()
        finished = [batch for batch in cls.batches if batch.done]
        print(f"Finished {len(finished)} of {len(cls.batches)} batches")

    @classmethod
    def save(cls):
        """
        Persist the current batch state (file IDs, job IDs, done flags) to disk
        using pickle. The item list is temporarily detached before pickling to
        avoid serializing the entire dataset (which could be huge). This allows
        resuming after a kernel restart without re-submitting jobs.
        """
        items = cls.batches[0].items
        for batch in cls.batches:
            batch.items = None  # Detach items so they aren't pickled
        with state.open("wb") as f:
            pickle.dump(cls.batches, f)
        for batch in cls.batches:
            batch.items = items  # Re-attach after saving
        print(f"Saved {len(cls.batches)} batches")

    @classmethod
    def load(cls, items):
        """
        Restore previously saved batch state from disk and re-attach the
        provided item list. Use this after a kernel restart to resume fetching
        results without re-submitting jobs to Groq.
        """
        with state.open("rb") as f:
            cls.batches = pickle.load(f)
        for batch in cls.batches:
            batch.items = items  # Re-attach the live item list
        print(f"Loaded {len(cls.batches)} batches")
