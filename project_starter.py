import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
import json
import contextvars
import logging
import re
import unicodedata
from difflib import get_close_matches
from sqlalchemy.sql import text
from datetime import date as DateType, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, model_validator
from sqlalchemy import create_engine, Engine
from smolagents import ToolCallingAgent, OpenAIServerModel, load_dotenv, tool
from uuid import uuid4

HANDOFF_TRACE_FILE = "handoff_trace.jsonl"
_HANDOFF_RUN_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "handoff_run_id", default=None
)
_HANDOFF_REQUEST_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "handoff_request_id", default=None
)
_HANDOFF_SEQUENCE: contextvars.ContextVar[int] = contextvars.ContextVar(
    "handoff_sequence", default=0
)


def _get_handoff_logger() -> logging.Logger:
    logger = logging.getLogger("beavers_choice.handoff_trace")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    trace_path = os.path.abspath(HANDOFF_TRACE_FILE)
    if not any(
        isinstance(handler, logging.FileHandler) and handler.baseFilename == trace_path
        for handler in logger.handlers
    ):
        handler = logging.FileHandler(trace_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    return logger


def _record_handoff(
    source: str,
    destination: str,
    request: Any,
    response: Any,
    outcome: str,
    started_at: datetime,
    started_clock: float,
) -> None:
    sequence = _HANDOFF_SEQUENCE.get() + 1
    _HANDOFF_SEQUENCE.set(sequence)
    completed_at = datetime.now(timezone.utc)
    record = {
        "event": "handoff",
        "event_id": uuid4().hex,
        "run_id": _HANDOFF_RUN_ID.get(),
        "request_id": _HANDOFF_REQUEST_ID.get(),
        "sequence": sequence,
        "source": source,
        "destination": destination,
        "request": request,
        "response": response,
        "outcome": outcome,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_ms": round((time.perf_counter() - started_clock) * 1000, 3),
    }
    _get_handoff_logger().info(json.dumps(record, ensure_ascii=False, default=str))

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]


def _normalize_product_text(value: str) -> str:
    """Lowercase, drop punctuation and singularize each word ("Paper Cups!" -> "paper cup")."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    words = re.sub(r"[^\w]+", " ", normalized).split()
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words)


# Closed-world synonyms: every target must be a catalog item name.
PRODUCT_NAME_ALIASES = {
    "a4 sheet": "A4 paper",
    "a4 copy paper": "A4 paper",
    "printer paper": "Standard copy paper",
    "printing paper": "Standard copy paper",
    "copy paper": "Standard copy paper",
    "white paper": "Standard copy paper",
    "colour paper": "Colored paper",
    "color paper": "Colored paper",
    "colored sheet": "Colored paper",
    "colour sheet": "Colored paper",
    "glossy sheet": "Glossy paper",
    "gloss paper": "Glossy paper",
    "recycled sheet": "Recycled paper",
    "card stock": "Cardstock",
    "napkin": "Paper napkins",
    "plate": "Paper plates",
    "envelope": "Envelopes",
    "streamer": "Party streamers",
    "washi tape": "Decorative adhesive tape (washi tape)",
    "banner": "Banner paper",
    "poster": "Poster paper",
    "folder": "Presentation folders",
}
_CANONICAL_PRODUCTS = {
    _normalize_product_text(item["item_name"]): item["item_name"]
    for item in paper_supplies
}
_NORMALIZED_PRODUCT_ALIASES = {
    _normalize_product_text(alias): canonical_name
    for alias, canonical_name in PRODUCT_NAME_ALIASES.items()
}

_GENERIC_WORDS = {"paper", "sheet", "of", "and", "the", "with", "a"}
FUZZY_CUTOFF = 0.85  # Typo tolerance only.
_SIZE_PRODUCTS = {"A4 paper", "Letter-sized paper", "Legal-size paper"}


def _product_tokens(normalized_name: str) -> set:
    return {t for t in normalized_name.split() if t not in _GENERIC_WORDS}


def _resolve_product_name(product_name: str) -> Dict[str, Any]:
    """Map text to one catalog item: exact, alias, typo, then unique word match; else ambiguous/unknown."""
    normalized_name = _normalize_product_text(product_name)

    def result(status: str, canonical: Optional[str] = None, match_type: Optional[str] = None, candidates=()):
        return {
            "status": status,
            "requested_name": product_name,
            "normalized_name": normalized_name,
            "canonical_name": canonical,
            "match_type": match_type,
            "candidates": list(candidates),
        }

    if normalized_name in _CANONICAL_PRODUCTS:
        return result("matched", _CANONICAL_PRODUCTS[normalized_name], "canonical")
    if normalized_name in _NORMALIZED_PRODUCT_ALIASES:
        return result("matched", _NORMALIZED_PRODUCT_ALIASES[normalized_name], "alias")

    # Catalog name contained in the request, e.g. "heavy cardstock (white)" -> Cardstock.
    request_words = set(normalized_name.split())
    contained = {
        name: len(key.split()) for key, name in _CANONICAL_PRODUCTS.items()
        if set(key.split()) <= request_words
    }
    best = max(contained.values(), default=0)
    candidates = [name for name, size in contained.items() if size == best]
    # A size word like "A4" only describes the other product ("A4 glossy paper").
    specific = [name for name in candidates if name not in _SIZE_PRODUCTS]
    candidates = specific or candidates

    if not candidates:
        candidates = [
            _CANONICAL_PRODUCTS[n]
            for n in get_close_matches(normalized_name, list(_CANONICAL_PRODUCTS), n=3, cutoff=FUZZY_CUTOFF)
        ]
    if not candidates:
        # Request is a part of a catalog name, e.g. "cups".
        requested_tokens = _product_tokens(normalized_name)
        if requested_tokens:
            candidates = [
                name for key, name in _CANONICAL_PRODUCTS.items()
                if requested_tokens <= _product_tokens(key)
            ]

    if len(candidates) == 1:
        return result("matched", candidates[0], "partial")
    if candidates:
        return result("ambiguous", candidates=candidates)
    return result("unknown")


def _normalize_request_products(request_model: BaseModel) -> Tuple[BaseModel, Optional[str], str]:
    if hasattr(request_model, "item"):
        original_names = [request_model.item]
        field_name = "item"
    elif hasattr(request_model, "items"):
        original_names = [item.item for item in request_model.items]
        field_name = "items"
    else:
        return request_model, None, ""

    canonical_names = []
    for original_name in original_names:
        resolution = _resolve_product_name(original_name)
        if resolution["status"] != "matched":
            candidates = resolution["candidates"]
            if not candidates:
                return request_model, (
                    f"Product '{original_name}' is not in our portfolio. "
                    "Reject the whole request with a brief apology."
                ), "rejected"
            return request_model, (
                f"Could not safely identify product '{original_name}'. "
                f"Possible matches: {', '.join(candidates)}. "
                "Ask the customer to clarify the product name before checking stock, quoting, or recording a sale."
            ), "needs_info"
        canonical_names.append(resolution["canonical_name"])

    if field_name == "item":
        return request_model.model_copy(update={"item": canonical_names[0]}), None, ""

    normalized_items = [
        item.model_copy(update={"item": canonical_name})
        for item, canonical_name in zip(request_model.items, canonical_names)
    ]
    return request_model.model_copy(update={"items": normalized_items}), None, ""

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: Matching quotes, each represented as a dictionary with these fields:
            - original_request (str): The customer's request text from `quote_requests.response`.
            - total_amount (int or float): The quoted total amount in dollars.
            - quote_explanation (str): The explanation of the quote, including pricing details.
            - job_type (str): Customer's role. Values observed in `quotes.csv` include
                business owner, city hall clerk, event manager, hotel manager,
                law firm receptionist, non-profit director, office manager,
                restaurant manager, retail store manager, school board resouorce manager,
                school principal, and school teacher.
            - order_size (str): Order-size category; observed values are small, medium,
                and large.
            - event_type (str): Event category. Values observed in `quotes.csv` include
                assembly, celebration, ceremony, concert, conference, demonstration,
                exhibition, festival, gathering, meeting, parade, party, performance,
                reception, and show.
            - order_date (str): Quote date as an ISO 8601 datetime string. The current
                database initialization assigns 2025-01-01T00:00:00 to historical quotes.
            The category values describe the supplied CSV data and are not enforced
            enumerations.
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.

load_dotenv()

model = OpenAIServerModel(
    model_id="gpt-4o-mini",  # Small, efficient model
    api_key=os.getenv("UDACITY_OPENAI_API_KEY"),
    api_base="https://openai.vocareum.com/v1",
)


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


@tool
def tool_resolve_product_name(product_name: str) -> str:
    """Resolve one customer product phrase to a canonical catalog name.

    Args:
        product_name: One extracted product name, without quantity or request text.

    Returns:
        JSON with status, canonical_name, match_type, and clarification candidates.
    """
    return json.dumps(_resolve_product_name(product_name), ensure_ascii=False)


# Tools for inventory agent
@tool
def tool_check_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Tool for checking the current inventory levels of all items as of a given date.

    Args:
        as_of_date (str): The date (in ISO format) to check the inventory levels.
    
    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    inventory_info = get_all_inventory(as_of_date)
    return {item: int(stock) for item, stock in inventory_info.items()}

@tool
def tool_check_stock_level(item_name: str, as_of_date: str) -> Dict[str, Any]:
    """
    Tool for checking the stock level of a specific item as of a given date.

    Args:
        item_name (str): The name of the item to check.
        as_of_date (str): The date (in ISO format) to check the stock level.

    Returns:
        Dict[str, Any]: The item name and its current stock level, e.g. {"item": "A4 paper", "stock": 272}.
    """
    stock_info = get_stock_level(item_name, as_of_date)
    stock = int(stock_info["current_stock"].iloc[0]) if not stock_info.empty else 0
    return {"item": item_name, "stock": stock}

@tool
def tool_check_delivery_timeline(input_date: str, quantity: int) -> str:
    """
    Tool for estimating the supplier delivery date based on the requested order quantity and a starting date.

    Args:
        input_date (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    return get_supplier_delivery_date(input_date, quantity)

@tool
def tool_get_unit_price(item_name: str) -> float:
    """
    Tool for looking up the unit price of an item in the inventory table.

    Args:
        item_name (str): The exact name of the item.

    Returns:
        float: The price per unit.
    """
    result = pd.read_sql(
        "SELECT unit_price FROM inventory WHERE item_name = :item_name",
        db_engine,
        params={"item_name": item_name},
    )
    if result.empty:
        raise ValueError(f"Unknown item: {item_name}")
    return float(result["unit_price"].iloc[0])

_CALC_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.USub: lambda a: -a,
}

def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("max", "min", "round"):
        return {"max": max, "min": min, "round": round}[node.func.id](*[_eval_node(a) for a in node.args])
    raise ValueError("Unsupported expression")

@tool
def tool_calculator(expression: str) -> float:
    """
    Tool for evaluating an arithmetic expression, e.g. "max(500 - 272, 0)" or "0.1 * 200 + 5.5".

    Args:
        expression (str): Expression using numbers, + - * /, parentheses, max, min and round.

    Returns:
        float: The result of the expression.
    """
    return _eval_node(ast.parse(expression, mode="eval").body)

# Tools for quoting agent
@tool
def tool_read_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Tool for retrieving historical quotes that match given search terms.

    Args:
        search_terms (List[str]): List of keywords to search in past quotes.
        limit (int, optional): Maximum number of results to return. Default is 5.

    Returns:
        List[Dict]: Matching quotes, each represented as a dictionary with these fields:
            - original_request (str): The customer's request text from `quote_requests.response`.
            - total_amount (int or float): The quoted total amount in dollars.
            - quote_explanation (str): The explanation of the quote, including pricing details.
            - job_type (str): Customer's role. Examples are: 
                business owner, city hall clerk, event manager, ...
            - order_size (str): Order-size category; observed values are small, medium,
                and large.
            - event_type (str): Event category. Examples are: 
                assembly, celebration, ceremony, concert, conference,...
            - order_date (str): Quote date as an ISO 8601 datetime string. The current
                database initialization assigns 2025-01-01T00:00:00 to historical quotes.
    """
    return search_quote_history(search_terms, limit)


# Tools for ordering agent
@tool
def tool_execute_order(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: str,
) -> int:
    """
    Tool for recording a transaction in the company's database and booking system.

    Args:
        item_name (str): The item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): The number of units involved.
        price (float): The total transaction price.
        date (str): The transaction date in ISO format.

    Returns:
        int: The ID of the created transaction.
    """
    return create_transaction(item_name, transaction_type, quantity, price, date)

# Tools for the reporting agent
@tool
def tool_get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Tool for checking the current cash balance of the company as of a given date.

    Args:
        as_of_date (str or datetime): The date to check the cash balance for.

    Returns:
        float: The current cash balance at or before the specified date.
    """
    return get_cash_balance(as_of_date)

@tool
def tool_generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Tool for generating the full financial report of the company as of a given date.

    Args:
        as_of_date (str or datetime): The date to generate the report for.

    Returns:
        Dict: A financial report with the following structure:
            - as_of_date (str): The report cutoff date. Datetime inputs are converted
                to ISO 8601 strings; string inputs are returned unchanged.
            - cash_balance (float): Cash balance at or before the cutoff date,
                calculated as sales revenue minus stock purchase costs.
            - inventory_value (float): Total value of inventory at the cutoff date,
                calculated from each item's stock multiplied by its unit price.
            - total_assets (float): The sum of cash_balance and inventory_value.
            - inventory_summary (List[Dict[str, str or int or float]]): One record
                per item in the inventory table. Each record contains:
                    - item_name (str): Inventory item name.
                    - stock (int or float): Units in stock at the cutoff date.
                    - unit_price (float): Price per unit from the inventory table.
                    - value (float): The item's stock multiplied by its unit price.
            - top_selling_products (List[Dict[str, str or int or float or None]]):
                Up to five sales records grouped by item and ordered by total_revenue.
                Each record contains:
                    - item_name (str or None): Item name; None is possible for the
                        opening cash-balance transaction, which has no item name.
                    - total_units (int, float, or None): Total units sold. This can be
                        None when the grouped transaction has no units recorded.
                    - total_revenue (int or float): Sum of transaction prices for the
                        item through the cutoff date.
    """
    return generate_financial_report(as_of_date)


# Set up your agents and create an orchestration agent that will manage them.

ModelT = TypeVar("ModelT", bound=BaseModel)
STRICT_JSON_FINAL_ANSWER_INSTRUCTION = (
    "Return exactly one JSON object and nothing else: no prose, Markdown fences, or "
    "text before or after the object."
)

def parse_model(model_cls: Type[ModelT], answer: Any) -> ModelT:
    """Parse an agent's final answer (dict or JSON string) into model_cls."""
    data = answer if isinstance(answer, dict) else json.loads(answer)
    return model_cls.model_validate(data)

def _has_side_effecting_tool_call(memory: Any) -> bool:
    return any(
        getattr(tool_call, "name", None) == "tool_execute_order"
        for step in getattr(memory, "steps", [])
        for tool_call in (getattr(step, "tool_calls", None) or [])
    )

def make_final_answer_check(model_cls: Type[BaseModel]) -> Callable[..., bool]:
    """Validate structured output and allow at most one safe correction attempt."""
    def check_final_answer(final_answer: Any, memory: Any, agent: Any = None) -> bool:
        try:
            parse_model(model_cls, final_answer)
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            error_message = (
                f"Final answer must be {STRICT_JSON_FINAL_ANSWER_INSTRUCTION} "
                f"It must validate as {model_cls.__name__}. Validation error: {e}"
            )
            if agent is None:
                raise ValueError(error_message) from e

            retry_count = getattr(agent, "_structured_response_retry_count", 0)
            if retry_count >= 1 or _has_side_effecting_tool_call(memory):
                agent._structured_response_retry_exhausted = True
                agent.interrupt_switch = True
                if _has_side_effecting_tool_call(memory):
                    error_message += " No format retry was attempted after tool_execute_order ran."
                else:
                    error_message += " The single format-correction attempt was exhausted."
            else:
                agent._structured_response_retry_count = retry_count + 1
                error_message += " Correct the format once; do not repeat completed tool actions."
            raise ValueError(error_message) from e
        return True
    return check_final_answer

def delegate(
    agent: ToolCallingAgent,
    request: Any,
    request_cls: Type[BaseModel],
    response_cls: Type[BaseModel],
) -> str:
    """Validate the request, run the agent, and return its validated response as JSON."""
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    agent_name = getattr(agent, "name", type(agent).__name__)
    try:
        validated = request_cls.model_validate(request)
    except ValidationError as e:
        schema = json.dumps(request_cls.model_json_schema())
        response_json = response_cls(
            status="rejected",
            result=(
                f"Request must conform to this JSON schema: {schema}. "
                f"Errors: {e.errors(include_url=False, include_input=False)}"
            ),
        ).model_dump_json()
        _record_handoff(
            "orchestrator",
            agent_name,
            request,
            response_json,
            "rejected",
            started_at,
            started_clock,
        )
        return response_json
    validated, product_resolution_error, resolution_status = _normalize_request_products(validated)
    if product_resolution_error is not None:
        response_json = response_cls(
            status=resolution_status,
            result=product_resolution_error,
        ).model_dump_json()
        _record_handoff(
            "orchestrator",
            agent_name,
            request,
            response_json,
            "rejected" if resolution_status == "rejected" else "needs_clarification",
            started_at,
            started_clock,
        )
        return response_json
    agent_request = validated.model_dump_json()
    agent._structured_response_retry_count = 0
    agent._structured_response_retry_exhausted = False
    try:
        answer = agent.run(agent_request)
    except Exception as e:
        invalid_response = getattr(agent, "_structured_response_retry_exhausted", False)
        failure_type = "invalid structured response" if invalid_response else "execution failure"
        response_json = response_cls(
            status="failed",
            result=f"{agent_name} {failure_type}: {type(e).__name__}: {e}",
        ).model_dump_json()
        _record_handoff(
            "orchestrator",
            agent_name,
            agent_request,
            response_json,
            "invalid_response" if invalid_response else "failed",
            started_at,
            started_clock,
        )
        return response_json

    # Every agent uses make_final_answer_check, so the answer is already validated here.
    response_json = parse_model(response_cls, answer).model_dump_json()

    _record_handoff(
        "orchestrator",
        agent_name,
        agent_request,
        response_json,
        "completed",
        started_at,
        started_clock,
    )
    return response_json

class InventoryItemRequest(BaseModel):
    """One requested item in an inventory manager request."""

    model_config = ConfigDict(extra="forbid")

    item: str = Field(min_length=1, description="Exact item name.")
    quantity: StrictInt = Field(gt=0, description="Number of units, as an integer.")

class InventoryRequest(BaseModel):
    """Structured request sent from the orchestrator to the inventory manager."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, description="What the inventory manager should do.")
    items: List[InventoryItemRequest] = Field(
        min_length=1, description="Requested items with quantities."
    )
    date: DateType = Field(
        description="Request date in YYYY-MM-DD format; use it as the stock-check cutoff and transaction posting date."
    )

INVENTORY_REQUEST_SCHEMA = json.dumps(InventoryRequest.model_json_schema())

class InventoryItemResponse(BaseModel):
    """Inventory outcome for one requested item."""

    model_config = ConfigDict(extra="forbid")

    item: str = Field(description="Item name from the request.")
    quantity: StrictInt = Field(gt=0, description="Requested units.")
    stock_on_hand: Optional[StrictInt] = Field(
        default=None, description="Units on hand as of the request date, if checked."
    )
    shortfall: Optional[StrictInt] = Field(
        default=None, description="Requested units not covered by stock on hand, if calculated."
    )
    estimated_delivery_date: Optional[DateType] = Field(
        default=None,
        description="Estimated supplier arrival date (YYYY-MM-DD) for the shortfall; this is not the transaction posting date.",
    )
    order_placed: bool = Field(
        default=False, description="True only if a stock order was recorded successfully."
    )
    transaction_id: Optional[StrictInt] = Field(
        default=None, description="ID of the recorded stock order transaction, if placed."
    )
    total_price: Optional[float] = Field(
        default=None, description="Total purchase price of the stock order, if placed."
    )

class InventoryResponse(BaseModel):
    """Structured response sent from the inventory manager back to the orchestrator."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "needs_info", "rejected", "failed"] = Field(
        description="Outcome of the request."
    )
    result: str = Field(description="Description of the work done.")
    items: List[InventoryItemResponse] = Field(
        default_factory=list, description="One inventory result per requested item."
    )

INVENTORY_RESPONSE_SCHEMA = json.dumps(InventoryResponse.model_json_schema())

class QuoteItem(BaseModel):
    """One requested item of a quote request."""

    model_config = ConfigDict(extra="forbid")

    item: str = Field(min_length=1, description="Exact item name.")
    quantity: StrictInt = Field(gt=0, description="Number of units, as an integer.")
    stock_on_hand: Optional[StrictInt] = Field(
        default=None, description="Units on hand, from the inventory manager's response."
    )
    estimated_delivery_date: Optional[DateType] = Field(
        default=None, description="Estimated supplier delivery date (YYYY-MM-DD), from the inventory manager."
    )

class QuoteRequest(BaseModel):
    """Structured request sent from the orchestrator to the quote generator."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, description="What the quote generator should do.")
    items: List[QuoteItem] = Field(min_length=1, description="Requested items with quantities.")
    date: DateType = Field(description="Request date in YYYY-MM-DD format.")
    job_type: Optional[str] = Field(default=None, description="Customer's role, e.g. event manager.")
    order_size: Optional[str] = Field(default=None, description="Order size: small, medium or large.")
    event_type: Optional[str] = Field(default=None, description="Event category, e.g. ceremony.")

QUOTE_REQUEST_SCHEMA = json.dumps(QuoteRequest.model_json_schema())

class QuoteLineItem(BaseModel):
    """Price of one item in a quote."""

    model_config = ConfigDict(extra="forbid")

    item: str
    quantity: StrictInt
    unit_price: Optional[float] = None
    line_total: Optional[float] = None

class QuoteResponse(BaseModel):
    """Structured response sent from the quote generator back to the orchestrator."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "needs_info", "rejected", "failed"] = Field(
        description="Outcome of the request."
    )
    result: str = Field(
        description="How the quote history was used, and what is assumed or missing."
    )
    is_estimate: bool = Field(default=True, description="True if the quote is only an estimate.")
    line_items: List[QuoteLineItem] = Field(
        default_factory=list, description="Per-item prices; required when status is ok."
    )

    @model_validator(mode="after")
    def check_line_items(self):
        if self.status == "ok" and not self.line_items:
            raise ValueError("line_items are required when status is ok")
        return self

QUOTE_RESPONSE_SCHEMA = json.dumps(QuoteResponse.model_json_schema())

class SaleItem(BaseModel):
    """One item of an authorized customer purchase."""

    model_config = ConfigDict(extra="forbid")

    item: str = Field(min_length=1, description="Exact item name.")
    quantity: StrictInt = Field(gt=0, description="Number of units, as an integer.")
    total_price: float = Field(gt=0, description="Total sale price for this item, from the quote.")

class SalesRequest(BaseModel):
    """Structured request sent from the orchestrator to the sales agent."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, description="What the sales agent should do.")
    items: List[SaleItem] = Field(min_length=1, description="Items to sell, with quoted prices.")
    date: DateType = Field(
        description=(
            "Request/booking date in YYYY-MM-DD format. Use this date for the stock check "
            "and sale transaction, including when a confirmed reorder has a later estimated arrival."
        )
    )
    quote_total: Optional[float] = Field(
        default=None, description="Quote total calculated by the orchestrator; must equal the sum of the item prices."
    )

    @model_validator(mode="after")
    def check_quote_total(self):
        if self.quote_total is not None:
            if abs(sum(i.total_price for i in self.items) - self.quote_total) > 0.01:
                raise ValueError("quote_total must equal the sum of the item total_price values")
        return self

SALES_REQUEST_SCHEMA = json.dumps(SalesRequest.model_json_schema())

class SaleResult(BaseModel):
    """Outcome for one requested item."""

    model_config = ConfigDict(extra="forbid")

    item: str
    quantity: StrictInt
    recorded: bool = Field(description="True only if the sale was recorded successfully.")
    transaction_id: Optional[StrictInt] = Field(default=None, description="Set if recorded.")
    total_price: Optional[float] = Field(default=None, description="Recorded price, set if recorded.")
    reason: Optional[str] = Field(default=None, description="Why the item was not recorded.")

    @model_validator(mode="after")
    def check_recorded(self):
        if self.recorded and (self.transaction_id is None or self.total_price is None):
            raise ValueError("recorded items need transaction_id and total_price")
        return self

class SalesResponse(BaseModel):
    """Structured response sent from the sales agent back to the orchestrator."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "needs_info", "rejected", "failed"] = Field(
        description="ok: all items recorded; otherwise no item is recorded."
    )
    result: str = Field(description="Description of the work done.")
    items: List[SaleResult] = Field(default_factory=list, description="One entry per requested item.")
    total_recorded: float = Field(default=0.0, description="Sum of the recorded item prices.")

    @model_validator(mode="after")
    def check_consistency(self):
        recorded = [i for i in self.items if i.recorded]
        if self.status == "ok" and (not self.items or len(recorded) != len(self.items)):
            raise ValueError("status ok requires all items to be recorded")
        if self.status != "ok" and recorded:
            raise ValueError("sales must be all-or-nothing: only status ok may have recorded items")
        if abs(sum(i.total_price for i in recorded) - self.total_recorded) > 0.01:
            raise ValueError("total_recorded must equal the sum of recorded item prices")
        return self

SALES_RESPONSE_SCHEMA = json.dumps(SalesResponse.model_json_schema())

class ReportRequest(BaseModel):
    """Structured request sent from the orchestrator to the reporting agent."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, description="Which report or figures are wanted.")
    date: DateType = Field(description="As-of date in YYYY-MM-DD format.")

REPORT_REQUEST_SCHEMA = json.dumps(ReportRequest.model_json_schema())

class ReportResponse(BaseModel):
    """Structured response sent from the reporting agent back to the orchestrator."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "needs_info", "rejected", "failed"] = Field(
        description="Outcome of the request."
    )
    result: str = Field(
        description="The report text, including itemized inventory and top sellers if requested."
    )
    as_of_date: Optional[DateType] = Field(default=None, description="Date the report covers.")
    cash_balance: Optional[float] = Field(default=None, description="Cash balance, if reported.")
    inventory_value: Optional[float] = Field(default=None, description="Inventory value, if reported.")
    total_assets: Optional[float] = Field(default=None, description="Total assets, if reported.")

    @model_validator(mode="after")
    def check_report(self):
        if self.status == "ok" and self.as_of_date is None:
            raise ValueError("as_of_date is required when status is ok")
        figures = (self.cash_balance, self.inventory_value, self.total_assets)
        if None not in figures and abs(self.cash_balance + self.inventory_value - self.total_assets) > 0.01:
            raise ValueError("total_assets must equal cash_balance + inventory_value")
        return self

REPORT_RESPONSE_SCHEMA = json.dumps(ReportResponse.model_json_schema())

class OrchestrationAgent(ToolCallingAgent):
    """
    Coordinates customer inquiries by selecting and calling the specialist best suited
    to each task. The inventory manager checks on-hand stock, estimates delivery dates,
    and places explicitly requested stock orders; the quote generator uses historical
    quotes as context; the sales agent records authorized customer purchases; and the
    reporting agent prepares dated financial reports.

    For customer requests that require specific items or quantities, the orchestrator
    lets check inventory before quoting or completing a sale. If on-hand stock is
    insufficient, it explicitly asks the inventory manager to place a stock order for
    the shortage, then communicates any missing order details or estimated delivery
    timing without presenting incoming stock as already available.

    Attributes:
        model (OpenAIServerModel): The language model used for reasoning and decision-making.
        inventory_manager_agent (InventoryManagerAgent): Handles inventory-related tasks.
        quote_generator_agent (QuoteGeneratorAgent): Handles customer quote requests.
        sales_agent (SalesAgent): Handles customer orders and sales transactions.
        reporting_agent (ReportingAgent): Handles financial reporting requests.
    """

    PROMPT = (
        "You coordinate customer inquiries for a paper supply company. Classify each "
        "request, forward requests to the relevant specialists, accept their responses, "
        "process thema and use them for further actions. "
        "Use the input from the specialists' responses for your reply to the customer."
        "Delegate only the work that is relevant to the request."
        "Requests are all-or-nothing: if every item of a customer request can be fulfilled, "
        "COMPLETE THE SALE for all items; if any item cannot be fulfilled, reject the whole "
        "request with a brief apology and do not quote or sell any part of it.\n\n"
        "Before calling the inventory manager, quote generator, or sales agent for a "
        "product, extract each product name and call tool_resolve_product_name. "
        "Proceed only with a resolved canonical name. If the status is unknown, do not "
        "delegate; reject the whole request with a brief apology. If it is "
        "ambiguous, ask the customer which of the candidates they mean.\n"
        "Choose a specialist as follows:\n"
        "- Call the inventory manager for on-hand stock, item availability, supplier "
        "delivery estimates, and stock reorders. Every inventory manager request must "
        "be a JSON object that conforms to this JSON schema (no extra fields):\n"
        f"{INVENTORY_REQUEST_SCHEMA}\n"
        'Example: {"task": "Check on-hand stock", "item": "A4 paper", '
        '"quantity": 500, "date": "2025-04-01"}\n'
        "The inventory manager replies with a JSON object that conforms to this JSON "
        "schema:\n"
        f"{INVENTORY_RESPONSE_SCHEMA}\n"
        "Check status first; match each response entry in items by item name and use "
        "its stock_on_hand, shortfall, estimated_delivery_date, order_placed, and "
        "total_price instead of inferring them from result.\n"
        "- Call the quote generator when a customer asks for a quote or pricing "
        "guidance. Call the inventory manager once with all requested items first, then "
        "send the quote generator a JSON object that conforms to this JSON schema, "
        "filling stock_on_hand and estimated_delivery_date per item from the "
        "inventory manager's responses:\n"
        f"{QUOTE_REQUEST_SCHEMA}\n"
        "The quote generator replies with a JSON object that "
        "conforms to this JSON schema:\n"
        f"{QUOTE_RESPONSE_SCHEMA}\n"
        "Check status first; take is_estimate and line_items from the response. The "
        "quote generator does not give a total: calculate the total_amount yourself as "
        "the sum of the line_total values, using tool_calculator.\n"
        "- Call the sales agent when a purchase is possible. "
        "Send a JSON object that conforms to this JSON schema, taking each item's "
        "total_price from the quote's line_items and quote_total from the total_amount "
        "you calculated. If the quote has no per-item prices, ask the quote generator "
        "for them first:\n"
        f"{SALES_REQUEST_SCHEMA}\n"
        "The sales agent replies with a JSON object that conforms "
        "to this JSON schema:\n"
        f"{SALES_RESPONSE_SCHEMA}\n"
        "Check status first; a sale is recorded only if status is ok. Otherwise tell the "
        "customer that nothing was ordered and why.\n"
        "- Call the reporting agent for cash balances, inventory valuations, total "
        "assets, or top-selling product reports. Send a JSON object that conforms to "
        "this JSON schema:\n"
        f"{REPORT_REQUEST_SCHEMA}\n"
        "The reporting agent replies with a JSON object that "
        "conforms to this JSON schema:\n"
        f"{REPORT_RESPONSE_SCHEMA}\n"
        "Check status first; take cash_balance, inventory_value, and total_assets from "
        "the structured fields and any other details from result.\n\n"
        "When forwarding a task to a specialist, always formulate a clear request that includes all relevant details and context. "
        "For a customer quote or purchase request that names one or more items and "
        "quantities, call the inventory manager exactly once with all items in its "
        "items array before quoting availability or asking the sales agent to record "
        "a sale. Include in the task any delivery deadline and whether shortfall "
        "reorders are authorized. The inventory manager checks stock, estimates any "
        "shortfall delivery, and places authorized orders in that same run, but only "
        "when the estimate meets the customer's requested delivery date. If any "
        "out-of-stock item would arrive after that date, do not quote or record any "
        "part of the purchase; reject the request with a brief apology and explain "
        "that it cannot arrive by the requested date. Never call the inventory "
        "manager again for individual items or follow-up orders. Use the request date "
        "as the posting date for stock orders and sales.\n\n"
        "Do not pass a price; the inventory "
        "manager determines the purchase price itself. Do not say the order was placed "
        "unless the inventory manager confirms successful execution.\n\n"
        "For this example, after the inventory manager confirms a reorder that meets "
        "the customer's requested delivery date, treat the ordered quantity as "
        "available for a sale posted on the request date, even if the estimated "
        "arrival is later than the request date. Never use this assumption when the "
        "estimated arrival misses the customer's requested delivery date. Include "
        "the stock and delivery findings (including dates) in any follow-up call to "
        "the quote generator.\n\n"
        "Always use tool_calculator for any arithmetic; never calculate in your head.\n\n"
        "Historical quotes are examples, not guarantees of current prices, discounts, "
        "or availability. Do not invent prices, discounts, stock, or delivery promises. "
        "Report actions as "
        "completed only after the responsible agent confirms them; otherwise explain "
        "what is pending or what information is needed.\n\n"
        "Your reply to the customer is short and only contains what matters to them: "
        "the outcome, items, quantities, prices and delivery dates. Never mention "
        "transaction IDs, tools, agents, or other internal details."
    )

    def __init__(self, model: OpenAIServerModel):
        self.model = model
        self.inventory_manager_agent = InventoryManagerAgent(model)
        self.quote_generator_agent = QuoteGeneratorAgent(model)
        self.sales_agent = SalesAgent(model)
        self.reporting_agent = ReportingAgent(model)

        @tool
        def call_inventory_manager(request: dict) -> str:
            """Delegate inventory and supplier delivery questions to the inventory manager.

            Args:
                request: Object with task (str), items (list of objects with item (str) and quantity (int)), and date (YYYY-MM-DD).

            Returns:
                The inventory manager's response as a JSON string conforming to InventoryResponse, with one result per item.
            """
            return delegate(self.inventory_manager_agent, request, InventoryRequest, InventoryResponse)

        @tool
        def call_quote_generator(request: dict) -> str:
            """Delegate customer quote requests to the quote generator.

            Args:
                request: Object with the fields task (str), items (list of objects with item (str) and quantity (int)), date (YYYY-MM-DD), and optionally job_type, order_size, event_type.

            Returns:
                The quote generator's response as a JSON string conforming to QuoteResponse.
            """
            return delegate(self.quote_generator_agent, request, QuoteRequest, QuoteResponse)

        @tool
        def call_sales_agent(request: dict) -> str:
            """Delegate authorized customer purchases to the sales agent.

            Args:
                request: Object with the fields task (str), items (list of objects with item (str), quantity (int), and total_price (float)), date (YYYY-MM-DD), and optionally quote_total (float).

            Returns:
                The sales agent's response as a JSON string conforming to SalesResponse.
            """
            return delegate(self.sales_agent, request, SalesRequest, SalesResponse)

        @tool
        def call_reporting_agent(request: dict) -> str:
            """Delegate financial reporting requests to the reporting agent.

            Args:
                request: Object with exactly the fields task (str) and date (YYYY-MM-DD).

            Returns:
                The reporting agent's response as a JSON string conforming to ReportResponse.
            """
            return delegate(self.reporting_agent, request, ReportRequest, ReportResponse)

        super().__init__(
            model=model,
            tools=[
                tool_resolve_product_name,
                tool_calculator,
                call_inventory_manager,
                call_quote_generator,
                call_sales_agent,
                call_reporting_agent,
            ],
            instructions=self.PROMPT,
        )

class InventoryManagerAgent(ToolCallingAgent):
    """
    Checks item stock levels, estimates supplier delivery dates, and places explicitly
    requested stock reorders when the required order details are available.

    Attributes:
        model (OpenAIServerModel): The language model used for reasoning and decision-making.
        tools (List[Callable]): A list of tools available to the agent for performing actions.
    """

    PROMPT = (
        "You are an inventory manager.\n\n"
        "## Input\n"
        "Each request is a JSON object that conforms to this JSON schema:\n"
        f"{INVENTORY_REQUEST_SCHEMA}\n"
        "Use date as the request date and stock-check cutoff. If placing a stock order, "
        "also use this date as its transaction date.\n\n"
        "## Process\n"
        "1. Process every entry in items during this run, using its exact item name and "
        "quantity. Check stock as of the requested date and return exactly one result "
        "for each requested item. Do not repeat completed tool calls.\n"
        "2. For each item whose stock is below its requested quantity, calculate the "
        "shortfall once with tool_calculator and estimate delivery for that shortfall from the request date. "
        "For items with sufficient stock, do not calculate a shortfall or delivery date.\n"
        "3. Place stock reorders only when the task explicitly authorizes them, and order "
        "only each item's shortfall. If the task includes the customer's requested "
        "delivery date and any item's estimated arrival is later, place no order for "
        "any item. Leave order_placed false for all items and report the shortfalls and "
        "estimated arrivals so the orchestrator can reject the whole request. Before ordering, "
        "look up each item's unit price with the unit price "
        "tool and use unit price times shortfall as the total purchase price; never ask "
        "the requester for a price. Ask for any other missing details instead of "
        "guessing.\n"
        "4. Record the order with tool_execute_order using "
        'transaction_type="stock_orders" and the request date as the transaction date. '
        "Keep the estimated delivery date only as an arrival estimate; for this example, "
        "the reorder is treated as available on its request-date posting.\n"
        "5. Report that the reorder was placed only after the tool confirms it.\n"
        "Do not use this process to record customer sales.\n\n"
        "## Output\n"
        "Finish by calling final_answer with one JSON string that conforms to this JSON "
        "schema. Return one entry in items for each request item, copying its item name "
        "and quantity:\n"
        f"{INVENTORY_RESPONSE_SCHEMA}\n"
        f"{STRICT_JSON_FINAL_ANSWER_INSTRUCTION} "
        "Put any human-readable summary in the result field; never pass plain text to "
        "final_answer.\n"
        "- Set order_placed to true only after tool_execute_order succeeds, and then also "
        "set transaction_id and total_price. Keep estimated_delivery_date as the "
        "supplier's estimated arrival date, which may differ from the transaction date.\n"
        "- Use status needs_info if required details are missing.\n\n"
        "Example final_answer argument for a stock check without a shortfall:\n"
        '{"status": "ok", "result": "Stock check complete.", "items": '
        '[{"item": "A4 paper", "quantity": 200, "stock_on_hand": 272, '
        '"shortfall": 0, "estimated_delivery_date": null, "order_placed": false, '
        '"transaction_id": null, "total_price": null}]}'
    )

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            model=model,
            tools=[
                tool_check_inventory,
                tool_check_stock_level,
                tool_check_delivery_timeline,
                tool_get_unit_price,
                tool_calculator,
                tool_execute_order,
            ],
            instructions=self.PROMPT,
            final_answer_checks=[make_final_answer_check(InventoryResponse)],
            max_steps=10,
            name="inventory_manager_agent_01",
            description="Checks inventory, estimates delivery timelines, and places authorized stock reorders.",
        )

class QuoteGeneratorAgent(ToolCallingAgent):
    """
    The QuoteGeneratorAgent handles tasks related to generating quotes for customer requests.
    It can search historical quotes, analyze customer requirements, and produce new quotes
    based on available inventory and pricing.

    Attributes:
        model (OpenAIServerModel): The language model used for reasoning and decision-making.
        tools (List[Callable]): A list of tools available to the agent for performing actions.
    """

    PROMPT = (
        "You are an expert for generating customer quotes using relevant historical quotes.\n\n"
        "## Input\n"
        "Each request is a JSON object that conforms to this JSON schema:\n"
        f"{QUOTE_REQUEST_SCHEMA}\n\n"
        "## Rules\n"
        "- Search quote history using the requested item names and any supplied "
        "job_type, order_size, or event_type.\n"
        "- Use the most relevant past quotes to estimate a clear total for this request. "
        "Use past per-item prices when available; otherwise use a similar past quote "
        "as the estimate and briefly explain the comparison.\n"
        "- Use tool_calculator for every calculation, e.g. line_total = unit_price * quantity. "
        "Do not give a total for the whole quote; the orchestrator calculates it.\n"
        "- Treat every price based on history as an estimate. Do not invent discounts. "
        "Always give a price for every item: if the history has none for an item, "
        "search for similar products and estimate from their prices, and as a last "
        "resort use the item's catalog unit price. Say in result how it was estimated.\n\n"
        "## Output\n"
        "Finish by calling final_answer with a JSON string that conforms to this JSON "
        "schema:\n"
        f"{QUOTE_RESPONSE_SCHEMA}\n"
        f"{STRICT_JSON_FINAL_ANSWER_INSTRUCTION} "
        "Put any human-readable explanation in the result field; never pass plain text "
        "to final_answer.\n"
        "- For a quote based on history, use status ok and is_estimate true. Include "
        "per-item line_items with unit_price and line_total.\n\n"
        "Example final_answer argument for an estimate based on quote history:\n"
        '{"status": "ok", "result": "Based on similar past ceremony quotes, the '
        'estimate is $20.00 for 200 sheets of A4 paper; no discount was supplied.", '
        '"is_estimate": true, "line_items": [{"item": "A4 paper", '
        '"quantity": 200, "unit_price": 0.1, "line_total": 20.0}]}'
    )

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            model=model,
            tools=[tool_read_quote_history, tool_get_unit_price, tool_calculator],
            instructions=self.PROMPT,
            final_answer_checks=[make_final_answer_check(QuoteResponse)],
            max_steps=6,
            name="quote_generator_agent_01",
            description="Generates customer quotes using request details and relevant historical quotes.",
        )

class SalesAgent(ToolCallingAgent):
    """
    Checks stock and records authorized customer sales transactions.

    Attributes:
        model (OpenAIServerModel): The language model used for reasoning and decision-making.
        tools (List[Callable]): A list of tools available to the agent for performing actions.
    """

    PROMPT = (
        "You are a sales agent who records authorized customer purchases.\n\n"
        "## Input\n"
        "Each request is a JSON object that conforms to this JSON schema:\n"
        f"{SALES_REQUEST_SCHEMA}\n"
        "Every request is an authorized customer purchase.\n\n"
        "## Rules\n"
        "- A request is all-or-nothing. First check the stock of every item as of the "
        "request date in the date field. If any item lacks the requested quantity, "
        "record nothing, use status rejected and give a reason for each item. Only if "
        "all items are available, record all of them, using that same date for the "
        "sale transactions, including when stock was reordered with that date.\n"
        "- Record each item as its own transaction with transaction_type=\"sales\" "
        "and the given total_price unchanged.\n"
        "- Use tool_calculator for any calculation, e.g. total_recorded.\n"
        "- Never invent or change prices, and do not promise a delivery date that has "
        "not been confirmed.\n"
        "- Report an item as recorded only after tool_execute_order succeeds.\n\n"
        "## Output\n"
        "Finish by calling final_answer with a JSON string that conforms to this JSON "
        "schema:\n"
        f"{SALES_RESPONSE_SCHEMA}\n"
        f"{STRICT_JSON_FINAL_ANSWER_INSTRUCTION} "
        "Put any human-readable summary in the result field; never pass plain text to "
        "final_answer.\n"
        "- Add one entry in items per requested item. A recorded item needs "
        "transaction_id and total_price; a non-recorded item needs a reason.\n"
        "- Status ok: all items recorded; rejected: none recorded. "
        "total_recorded is the sum of the recorded prices (0 if none).\n\n"
        "Example final_answer argument when one of two items was out of stock:\n"
        '{"status": "rejected", "result": "Nothing recorded; poster paper is out of '
        'stock.", "items": [{"item": "A4 paper", "quantity": 200, "recorded": false, '
        '"transaction_id": null, "total_price": null, "reason": "Not recorded because another item is unavailable"}, '
        '{"item": "Poster paper", "quantity": 50, "recorded": false, '
        '"transaction_id": null, "total_price": null, "reason": "Insufficient stock"}], '
        '"total_recorded": 0.0}'
    )

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            model=model,
            tools=[tool_execute_order, tool_check_stock_level, tool_calculator],
            instructions=self.PROMPT,
            final_answer_checks=[make_final_answer_check(SalesResponse)],
            max_steps=12,
            name="sales_agent_01",
            description="Processes customer orders, records sales, and checks stock availability.",
        )

class ReportingAgent(ToolCallingAgent):
    """
    Generates dated financial reports covering cash, inventory value, total assets,
    and top-selling products.

    Attributes:
        model (OpenAIServerModel): The language model used for reasoning and decision-making.
        tools (List[Callable]): A list of tools available to the agent for performing actions.
    """

    PROMPT = (
        "You are a reporting agent who prepares dated financial reports.\n\n"
        "## Input\n"
        "Each request is a JSON object that conforms to this JSON schema:\n"
        f"{REPORT_REQUEST_SCHEMA}\n"
        "Use date as the as-of date.\n\n"
        "## Rules\n"
        "- Use tool_calculator for any calculation, e.g. total assets.\n"
        "- Base every figure on the financial report and cash balance tools; do not "
        "infer or invent values the tools do not provide.\n"
        "- Summarize cash balance, inventory value and total assets, and include "
        "itemized inventory or top-selling products when relevant to the request.\n"
        "- Clearly label the report date and keep reported figures separate from any "
        "interpretation.\n\n"
        "## Output\n"
        "Finish by calling final_answer with a JSON string that conforms to this JSON "
        "schema:\n"
        f"{REPORT_RESPONSE_SCHEMA}\n"
        f"{STRICT_JSON_FINAL_ANSWER_INSTRUCTION} "
        "Put the report text, including itemized inventory and top-selling products, in "
        "the result field; never pass plain text to final_answer.\n"
        "- Set as_of_date for every status ok report.\n"
        "- Fill cash_balance, inventory_value and total_assets only from the tools; "
        "total_assets must equal cash_balance plus inventory_value.\n\n"
        "Example final_answer argument for a summary report:\n"
        '{"status": "ok", "result": "Report as of 2025-04-01: cash balance $45,000.00, '
        'inventory value $5,000.00, total assets $50,000.00.", '
        '"as_of_date": "2025-04-01", "cash_balance": 45000.0, '
        '"inventory_value": 5000.0, "total_assets": 50000.0}'
    )

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            model=model,
            tools=[tool_generate_financial_report, tool_get_cash_balance, tool_calculator],
            instructions=self.PROMPT,
            final_answer_checks=[make_final_answer_check(ReportResponse)],
            max_steps=6,
            name="reporting_agent_01",
            description="Generates financial reports on cash, inventory, and company assets.",
        )


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        #quote_requests_sample = quote_requests_sample.sort_values("request_date")
        quote_requests_sample = quote_requests_sample.sort_values("request_date").iloc[:11]
        #quote_requests_sample = quote_requests_sample.sort_values("request_date").iloc[[2]]
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    multi_agent_system_orchestrator = OrchestrationAgent(model)
    trace_run_id = uuid4().hex

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        run_token = _HANDOFF_RUN_ID.set(trace_run_id)
        request_token = _HANDOFF_REQUEST_ID.set(str(idx + 1))
        sequence_token = _HANDOFF_SEQUENCE.set(0)
        started_at = datetime.now(timezone.utc)
        started_clock = time.perf_counter()
        try:
            response = multi_agent_system_orchestrator.run(request_with_date)
        except Exception as e:
            _record_handoff(
                "test_runner",
                "orchestrator",
                request_with_date,
                {"error_type": type(e).__name__, "error": str(e)},
                "failed",
                started_at,
                started_clock,
            )
            raise
        else:
            _record_handoff(
                "test_runner",
                "orchestrator",
                request_with_date,
                response,
                "completed",
                started_at,
                started_clock,
            )
        finally:
            _HANDOFF_SEQUENCE.reset(sequence_token)
            _HANDOFF_REQUEST_ID.reset(request_token)
            _HANDOFF_RUN_ID.reset(run_token)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )
        pd.DataFrame(results).to_csv("test_results.csv", index=False)

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    return results


if __name__ == "__main__":
    results = run_test_scenarios()
