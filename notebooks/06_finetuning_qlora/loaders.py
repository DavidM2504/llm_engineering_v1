# =============================================================================
# Program: loaders.py
#
# PURPOSE:
#   Loads raw product data from the McAuley-Lab Amazon Reviews 2023 dataset
#   on HuggingFace and converts it into lists of Item objects. Processing is
#   parallelised across CPU cores for speed since the full dataset contains
#   millions of rows.
#
# CONNECTIONS TO OTHER PROGRAMS:
#   - Calls parsers.py (parse()) to validate and convert each raw datapoint
#     into an Item object
#   - Produces lists of Item objects as defined in items.py
#   - The resulting Item lists are split into train/val/test sets elsewhere
#     (e.g. in a notebook) and then passed to:
#       * batch.py / preprocessor.py to generate AI summaries
#       * deep_neural_network.py for model training
#       * evaluator.py for evaluation
# =============================================================================

from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor
from pricer.parser import parse  # Item validation / construction logic (parsers.py)
import os

# Number of items per chunk sent to each worker process
CHUNK_SIZE = 1000

# Use all CPU cores minus one so the system remains responsive
cpu_count = os.cpu_count()
WORKERS = max(cpu_count - 1, 1)


class ItemLoader:
    """
    Downloads one Amazon product category from HuggingFace and converts the
    raw rows into validated Item objects using parallel processing.

    Typical usage:
        loader = ItemLoader("Electronics")
        items = loader.load()
        # items is now a list[Item] ready for train/val/test splitting
    """

    def __init__(self, category):
        """
        category : Amazon product category string (e.g. "Electronics", "Clothing")
                   Used both as a label on each Item and to select the correct
                   HuggingFace dataset split.
        """
        self.category = category
        self.dataset = None  # Populated by load()

    def from_datapoint(self, datapoint):
        """
        Attempt to create a single Item from one raw HuggingFace dataset row.
        Delegates to parse() in parsers.py, which validates the price range,
        cleans the text, and constructs the Item.
        Returns None if the datapoint is invalid (wrong price range, too short,
        missing fields, etc.); these are filtered out in from_chunk().
        """
        return parse(datapoint, self.category)

    def from_chunk(self, chunk):
        """
        Process a chunk (slice) of the dataset — a list of raw datapoints.
        Calls from_datapoint() on each row and discards None results.
        This method is designed to run in a separate worker process.
        """
        batch = [self.from_datapoint(datapoint) for datapoint in chunk]
        return [item for item in batch if item is not None]

    def chunk_generator(self):
        """
        Yield consecutive slices of size CHUNK_SIZE from the full dataset.
        Each slice is a HuggingFace Dataset object (not a plain list) so that
        it can be efficiently serialised and sent to worker processes.
        """
        size = len(self.dataset)
        for i in range(0, size, CHUNK_SIZE):
            yield self.dataset.select(range(i, min(i + CHUNK_SIZE, size)))

    def load_in_parallel(self, workers):
        """
        Farm out chunk processing to a pool of worker processes.
        ProcessPoolExecutor is used (rather than ThreadPoolExecutor) because
        the work is CPU-bound (regex, JSON parsing) and benefits from true
        parallelism without the GIL.

        Returns a flat list of all valid Items across all chunks.
        """
        results = []
        chunk_count = (len(self.dataset) // CHUNK_SIZE) + 1
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in tqdm(pool.map(self.from_chunk, self.chunk_generator()), total=chunk_count):
                results.extend(batch)
        return results

    def load(self, workers=WORKERS):
        """
        Full pipeline for one category:
          1. Download the raw HuggingFace dataset for this category.
          2. Process it in parallel (load_in_parallel → from_chunk → parse).
          3. Return a list of valid Item objects.

        The "full" split means we get all rows (not just a sampled subset).
        trust_remote_code=True is required by this particular HuggingFace dataset.
        """
        start = datetime.now()
        print(f"Loading dataset {self.category}", flush=True)
        self.dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{self.category}",  # Dataset config name encodes the category
            split="full",
            trust_remote_code=True,
        )
        results = self.load_in_parallel(workers)
        finish = datetime.now()
        print(
            f"Completed {self.category} with {len(results):,} datapoints in {(finish - start).total_seconds() / 60:.1f} mins",
            flush=True,
        )
        return results

