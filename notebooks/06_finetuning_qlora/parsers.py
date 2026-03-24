
# =============================================================================
# Program: parsers.py  (embedded below for reference — same file in the repo)
#
# PURPOSE:
#   Responsible for validating and cleaning individual raw Amazon dataset rows
#   before they become Item objects. Acts as the quality gate for the pipeline:
#   only rows with a valid price, sufficient text, and parseable fields pass
#   through. Also extracts the item weight from the raw details dictionary.
#
# CONNECTIONS TO OTHER PROGRAMS:
#   - Called exclusively by loaders.py (ItemLoader.from_datapoint → parse())
#   - Creates Item objects as defined in items.py
#   - The "full" field it builds (cleaned product text) is later consumed by
#     batch.py and preprocessor.py as the input to the AI summariser
# =============================================================================

from pricer.items import Item  # The central data model (items.py)
import json
import re

# --- Quality thresholds ---
MIN_CHARS = 600      # Minimum characters in cleaned text — short entries lack detail
MIN_PRICE = 0.5      # Minimum price in USD (filters free / placeholder items)
MAX_PRICE = 999.49   # Maximum price in USD (filters outliers / data errors)
MAX_TEXT_EACH = 3000 # Max chars taken from each text field before combining
MAX_TEXT_TOTAL = 4000 # Max chars of the final combined "full" text sent to the model

# Detail keys to remove: internal Amazon metadata that won't help the model
REMOVALS = [
    "Part Number",
    "Best Sellers Rank",
    "Batteries Included?",
    "Batteries Required?",
    "Item model number",
]


def simplify(text_list) -> str:
    """
    Collapse a raw text field (often a Python list repr from the dataset) into
    a single clean string:
      - Remove newlines, carriage returns, and tabs
      - Collapse double spaces
      - Strip leading/trailing whitespace
      - Truncate to MAX_TEXT_EACH characters so one verbose field can't crowd
        out others in the combined "full" text
    """
    return (
        str(text_list)
        .replace("\n", " ")
        .replace("\r", "")
        .replace("\t", "")
        .replace("  ", " ")
        .strip()[:MAX_TEXT_EACH]
    )


def scrub(title, description, features, details) -> str:
    """
    Build the combined "full" product text from all available fields and remove
    noise that could confuse the AI summariser or the pricing model:
      1. Remove unwanted detail keys (REMOVALS list above)
      2. Concatenate title, description, features, and JSON-encoded details
      3. Strip alphanumeric part numbers (regex: 7+ char sequences mixing letters
         and digits, like "B07XJ8C8F5") — these aren't useful for pricing

    The returned string becomes item.full and is later sent to the AI summariser
    in batch.py / preprocessor.py.
    """
    for remove in REMOVALS:
        details.pop(remove, None)

    result = title + "\n"
    if description:
        result += simplify(description) + "\n"
    if features:
        result += simplify(features) + "\n"
    if details:
        result += json.dumps(details) + "\n"

    # Remove part-number-like tokens: 7+ chars, must contain both letters and digits
    pattern = r"\b(?=[A-Z0-9]{7,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+\b"
    return re.sub(pattern, "", result).strip()[:MAX_TEXT_TOTAL]


def get_weight(details):
    """
    Extract item weight in pounds from the "Item Weight" detail string.
    Handles multiple unit formats found in Amazon data:
      pounds, ounces, grams, milligrams, kilograms,
      and the unusual "hundredths pounds" format.
    Returns 0 if the field is missing or unparseable (weight is optional).
    The returned value is stored as item.weight in items.py.
    """
    weight_str = details.get("Item Weight")
    if weight_str:
        parts = weight_str.split(" ")
        amount = float(parts[0])
        unit = parts[1].lower()
        if unit == "pounds":
            return amount
        elif unit == "ounces":
            return amount / 16
        elif unit == "grams":
            return amount / 453.592
        elif unit == "milligrams":
            return amount / 453592
        elif unit == "kilograms":
            return amount / 0.453592
        elif unit == "hundredths" and parts[2].lower() == "pounds":
            return amount / 100
    return 0


def parse(datapoint, category):
    """
    The main entry point called by loaders.py for every row in the dataset.
    Applies all quality filters and, if the row passes, constructs and returns
    an Item object.

    Filters applied (return None to discard):
      1. Price must be parseable as a float
      2. Price must be within [MIN_PRICE, MAX_PRICE]
      3. Cleaned full text must be at least MIN_CHARS characters long
         (short entries typically lack enough product information)

    On success, creates an Item (items.py) with:
      - title, category, price : basic product metadata
      - full  : cleaned combined text (from scrub()) for the AI summariser
      - weight: optional weight in pounds (from get_weight())
    """
    try:
        price = float(datapoint["price"])
    except ValueError:
        return None  # Discard rows where price is not a valid number

    if MIN_PRICE <= price <= MAX_PRICE:
        title = datapoint["title"]
        description = datapoint["description"]
        features = datapoint["features"]
        details = json.loads(datapoint["details"])
        weight = get_weight(details)  # Optional; 0 if not present
        full = scrub(title, description, features, details)  # Build cleaned text
        if len(full) >= MIN_CHARS:
            # Construct and return the Item (see items.py for field definitions)
            return Item(
                title=title,
                category=category,
                price=price,
                full=full,
                weight=weight,
            )
    # Return None implicitly if price is out of range or text is too short