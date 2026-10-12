import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
import json
from sqlalchemy.sql import text
from datetime import date as DateType, datetime, timedelta
from typing import Any, Callable, Dict, List, Literal, Optional, Type, TypeVar, Union
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, model_validator
from sqlalchemy import create_engine, Engine
from smolagents import ToolCallingAgent, OpenAIServerModel, load_dotenv, tool

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
def tool_check_stock_level(item_name: str, as_of_date: str) -> int:
    """
    Tool for checking the stock level of a specific item as of a given date.

    Args:
        item_name (str): The name of the item to check.
        as_of_date (str): The date (in ISO format) to check the stock level.

    Returns:
        int: The current stock level of the specified item.
    """
    stock_info = get_stock_level(item_name, as_of_date)
    return int(stock_info["current_stock"].iloc[0]) if not stock_info.empty else 0

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

@tool
def tool_calculate_shortfall(requested_quantity: int, stock_on_hand: int) -> int:
    """
    Tool for calculating how many units are missing to cover a requested quantity.

    Args:
        requested_quantity (int): The number of units requested.
        stock_on_hand (int): The number of units on hand.

    Returns:
        int: The missing units, or 0 if the stock covers the request.
    """
    return max(requested_quantity - stock_on_hand, 0)

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

def parse_model(model_cls: Type[ModelT], answer: Any) -> ModelT:
    """Parse an agent's final answer (dict or JSON string) into model_cls."""
    data = answer if isinstance(answer, dict) else json.loads(answer)
    return model_cls.model_validate(data)

def make_final_answer_check(model_cls: Type[BaseModel]) -> Callable[..., bool]:
    """Build a smolagents final-answer check; a failure is fed back to the agent so it retries."""
    def check_final_answer(final_answer: Any, memory: Any, agent: Any = None) -> bool:
        parse_model(model_cls, final_answer)
        return True
    return check_final_answer

def delegate(
    agent: ToolCallingAgent,
    request: Any,
    request_cls: Type[BaseModel],
    response_cls: Type[BaseModel],
) -> str:
    """Validate the request, run the agent, and return its validated response as JSON."""
    try:
        validated = request_cls.model_validate(request)
    except ValidationError as e:
        schema = json.dumps(request_cls.model_json_schema())
        return response_cls(
            status="rejected",
            result=(
                f"Request must conform to this JSON schema: {schema}. "
                f"Errors: {e.errors(include_url=False, include_input=False)}"
            ),
        ).model_dump_json()
    try:
        answer = agent.run(validated.model_dump_json())
        return parse_model(response_cls, answer).model_dump_json()
    except Exception as e:
        return response_cls(
            status="failed",
            result=f"{agent.name} did not return a valid response: {e}",
        ).model_dump_json()

class InventoryRequest(BaseModel):
    """Structured request sent from the orchestrator to the inventory manager."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(
        min_length=1,
        description="What the inventory manager should do.",
    )
    item: str = Field(min_length=1, description="Exact item name.")
    quantity: StrictInt = Field(gt=0, description="Number of units, as an integer.")
    date: DateType = Field(description="Transaction or as-of date in YYYY-MM-DD format.")

INVENTORY_REQUEST_SCHEMA = json.dumps(InventoryRequest.model_json_schema())

class InventoryResponse(BaseModel):
    """Structured response sent from the inventory manager back to the orchestrator."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "needs_info", "rejected", "failed"] = Field(
        description="Outcome of the request."
    )
    result: str = Field(description="Description of the work done.")
    item: str = Field(default="", description="Item name from the request.")
    stock_on_hand: Optional[StrictInt] = Field(
        default=None, description="Units on hand as of the request date, if checked."
    )
    shortfall: Optional[StrictInt] = Field(
        default=None, description="Requested units not covered by stock on hand, if calculated."
    )
    estimated_delivery_date: Optional[DateType] = Field(
        default=None, description="Estimated supplier delivery date (YYYY-MM-DD) for the shortfall, if estimated."
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
    total_amount: Optional[float] = Field(
        default=None, description="Total quoted amount in dollars; required when status is ok."
    )
    line_items: List[QuoteLineItem] = Field(
        default_factory=list, description="Per-item prices, only if they can be derived from the context."
    )

    @model_validator(mode="after")
    def check_totals(self):
        if self.status == "ok" and self.total_amount is None:
            raise ValueError("total_amount is required when status is ok")
        totals = [li.line_total for li in self.line_items]
        if self.total_amount is not None and totals and None not in totals:
            if abs(sum(totals) - self.total_amount) > 0.01:
                raise ValueError("line_total values must sum to total_amount")
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
        description="Transaction date in YYYY-MM-DD format; the goods must be available on this date."
    )
    quote_total: Optional[float] = Field(
        default=None, description="total_amount of the quote; must equal the sum of the item prices."
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

    status: Literal["ok", "partial", "needs_info", "rejected", "failed"] = Field(
        description="ok: all items recorded; partial: some recorded."
    )
    result: str = Field(description="Description of the work done.")
    items: List[SaleResult] = Field(default_factory=list, description="One entry per requested item.")
    total_recorded: float = Field(default=0.0, description="Sum of the recorded item prices.")

    @model_validator(mode="after")
    def check_consistency(self):
        recorded = [i for i in self.items if i.recorded]
        if self.status == "ok" and (not self.items or len(recorded) != len(self.items)):
            raise ValueError("status ok requires all items to be recorded")
        if self.status == "partial" and (not recorded or len(recorded) == len(self.items)):
            raise ValueError("status partial requires some, but not all, items to be recorded")
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
        "Delegate only the work that is relevant to the request.\n\n"
        "Choose a specialist as follows:\n"
        "- Call the inventory manager for on-hand stock, item availability, supplier "
        "delivery estimates, and stock reorders. Every inventory manager request must "
        "be a JSON object that conforms to this JSON schema (no extra fields):\n"
        f"{INVENTORY_REQUEST_SCHEMA}\n"
        'Example: {"task": "Check on-hand stock", "item": "A4 paper", '
        '"quantity": 500, "date": "2025-04-01"}\n'
        "Requests with missing, extra, or wrongly typed fields are rejected; correct "
        "them and call again.\n"
        "The inventory manager replies with a JSON object that conforms to this JSON "
        "schema:\n"
        f"{INVENTORY_RESPONSE_SCHEMA}\n"
        "Check status first; use stock_on_hand, shortfall, estimated_delivery_date, "
        "order_placed, and total_price from the response instead of inferring them "
        "from result.\n"
        "- Call the quote generator when a customer asks for a quote or pricing "
        "guidance. Call the inventory manager once per requested item first, then "
        "send the quote generator a JSON object that conforms to this JSON schema, "
        "filling stock_on_hand and estimated_delivery_date per item from the "
        "inventory manager's responses:\n"
        f"{QUOTE_REQUEST_SCHEMA}\n"
        "Requests with missing, extra, or wrongly typed fields are rejected; correct "
        "them and call again. The quote generator replies with a JSON object that "
        "conforms to this JSON schema:\n"
        f"{QUOTE_RESPONSE_SCHEMA}\n"
        "Check status first; take total_amount and is_estimate from the response "
        "instead of inferring them from result.\n"
        "- Call the sales agent only when a customer clearly authorizes a purchase. "
        "Send a JSON object that conforms to this JSON schema, taking each item's "
        "total_price from the quote's line_items and quote_total from its "
        "total_amount. If the quote has no per-item prices, ask the quote generator "
        "for them first:\n"
        f"{SALES_REQUEST_SCHEMA}\n"
        "Requests with missing, extra, or wrongly typed fields are rejected; correct "
        "them and call again. The sales agent replies with a JSON object that conforms "
        "to this JSON schema:\n"
        f"{SALES_RESPONSE_SCHEMA}\n"
        "Check status first; treat a sale as recorded only for items with recorded=true.\n"
        "- Call the reporting agent for cash balances, inventory valuations, total "
        "assets, or top-selling product reports. Send a JSON object that conforms to "
        "this JSON schema:\n"
        f"{REPORT_REQUEST_SCHEMA}\n"
        "Requests with missing, extra, or wrongly typed fields are rejected; correct "
        "them and call again. The reporting agent replies with a JSON object that "
        "conforms to this JSON schema:\n"
        f"{REPORT_RESPONSE_SCHEMA}\n"
        "Check status first; take cash_balance, inventory_value, and total_assets from "
        "the structured fields and any other details from result.\n\n"
        "When forwarding a task to a specialist, always formulate a clear request that includes all relevant details and context. "
        "For a customer quote or purchase request that names an item and quantity, "
        "check on-hand stock with the inventory manager before quoting availability or "
        "asking the sales agent to record a sale. Reject a purchase request if the item is not in stock and the expected delivery date is higher than the customer-specified date. If the item is in stock, confirm the available quantity and any delivery estimate before quoting a price or recording a sale.\n\n"
        "If delivery in time is realistic, call the inventory manager again with the "
        "item, the full requested quantity, and the request date, and tell it to place a "
        "stock order for the shortfall. It calculates the shortfall itself and dates the "
        "order at the estimated delivery date. Do not pass a price; the inventory "
        "manager determines the purchase price itself. Do not say the order was placed "
        "unless the inventory manager confirms successful execution.\n\n"
        "Ordered stock becomes available only on its estimated delivery date. Use the "
        "inventory manager's delivery estimate when explaining when it may become "
        "available. Ask the sales agent to record a sale only when the goods are "
        "available on the sale date: use the request date as date if the stock on hand "
        "covers the quantity, or the estimated delivery date after a stock order was "
        "placed. "
        "Include the stock and delivery findings (also dates) in any follow-up call to "
        "the quote generator.\n\n"
        "Historical quotes are examples, not guarantees of current prices, discounts, "
        "or availability. Do not invent prices, discounts, stock, or delivery promises. "
        "Do not record a customer sale without clear authorization. Report actions as "
        "completed only after the responsible agent confirms them; otherwise explain "
        "what is pending or what information is needed."
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
                request: Object with exactly the fields task (str), item (str), quantity (int), and date (YYYY-MM-DD).

            Returns:
                The inventory manager's response as a JSON string conforming to InventoryResponse.
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
        "Each request is a JSON object that conforms to this JSON schema:\n"
        f"{INVENTORY_REQUEST_SCHEMA}\n"
        "Use date as the as-of date. "
        "Answer inventory questions using the inventory tools and the requested as-of "
        "date, and report stock quantities accurately. If the stock on hand is below "
        "the requested quantity, calculate the shortfall with the shortfall tool and "
        "estimate the delivery date for the shortfall quantity, not the requested "
        "quantity, with the delivery timeline tool, starting from the request date. "
        "Place a stock reorder only when explicitly asked to do so, and order only the "
        "shortfall. Before ordering, look up the item's unit price with the unit price "
        "tool and use unit price times shortfall as the total purchase price; never "
        "ask the requester for a price. Ask for any other missing details instead of "
        "guessing. Record the order with tool_execute_order using "
        'transaction_type="stock_orders" and the estimated delivery date as the '
        "transaction date, not the request date, so that the stock and the cash "
        "outflow take effect on delivery. Report that the reorder was placed only "
        "after the tool confirms it. Do not use this process to record customer sales.\n"
        "Return your final answer as a JSON object that conforms to this JSON schema, "
        "with item copied from the request:\n"
        f"{INVENTORY_RESPONSE_SCHEMA}\n"
        "Set order_placed to true only after tool_execute_order succeeds, and then also "
        "set transaction_id and total_price, with estimated_delivery_date equal to the "
        "order's transaction date. Use status needs_info if required details "
        "are missing."
    )

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            model=model,
            tools=[
                tool_check_inventory,
                tool_check_stock_level,
                tool_check_delivery_timeline,
                tool_get_unit_price,
                tool_calculate_shortfall,
                tool_execute_order,
            ],
            instructions=self.PROMPT,
            final_answer_checks=[make_final_answer_check(InventoryResponse)],
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
        "Each request is a JSON object that conforms to this JSON schema:\n"
        f"{QUOTE_REQUEST_SCHEMA}\n"
        "Use the quote history tool to find relevant examples, using the item names "
        "and any job_type, order_size, and event_type as search terms, then explain how they "
        "relate to the customer's request. Treat historical quotes as context, not as "
        "current prices. Apply only discount rules and prices provided in the request "
        "or other supplied context; do not invent prices or discounts. Ask for missing "
        "details needed to prepare a reliable quote, and make clear when a quote is "
        "only an estimate.\n"
        "Return your final answer as a JSON object that conforms to this JSON schema:\n"
        f"{QUOTE_RESPONSE_SCHEMA}\n"
        "Set total_amount only from the history and supplied context, and fill "
        "line_items only if per-item prices can be derived from them. Use status "
        "needs_info if details are missing or if the task asks for per-item prices "
        "that cannot be derived."
    )

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            model=model,
            tools=[tool_read_quote_history],
            instructions=self.PROMPT,
            final_answer_checks=[make_final_answer_check(QuoteResponse)],
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
        "Each request is a JSON object that conforms to this JSON schema:\n"
        f"{SALES_REQUEST_SCHEMA}\n"
        "Every request is an authorized customer purchase. "
        "For each item, use the stock tool to verify the requested quantity is "
        "available as of the date field. Record only the items that are in stock, "
        "each with its own transaction, using the given total_price unchanged, and skip "
        'the others with a reason. Use transaction_type="sales" for customer purchases. '
        "Do not invent or change prices, and do not promise a delivery date that has "
        "not been confirmed.\n"
        "Return your final answer as a JSON object that conforms to this JSON schema:\n"
        f"{SALES_RESPONSE_SCHEMA}\n"
        "Set recorded to true only after tool_execute_order succeeds, and then also set "
        "transaction_id and total_price. Use status partial if only some items were "
        "recorded."
    )

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            model=model,
            tools=[tool_execute_order, tool_check_stock_level],
            instructions=self.PROMPT,
            final_answer_checks=[make_final_answer_check(SalesResponse)],
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
        "Each request is a JSON object that conforms to this JSON schema:\n"
        f"{REPORT_REQUEST_SCHEMA}\n"
        "Generate reports using the financial report tool and the requested date as "
        "the as-of date. "
        "Summarize cash balance, inventory value, and total assets; include itemized "
        "inventory or top-selling products when relevant. Clearly label the report date "
        "and distinguish reported figures from any interpretation. Do not infer values "
        "the report does not provide.\n"
        "Return your final answer as a JSON object that conforms to this JSON schema:\n"
        f"{REPORT_RESPONSE_SCHEMA}\n"
        "Fill cash_balance, inventory_value, and total_assets only from the tools, and "
        "put itemized inventory and top-selling products in result."
    )

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            model=model,
            tools=[tool_generate_financial_report, tool_get_cash_balance],
            instructions=self.PROMPT,
            final_answer_checks=[make_final_answer_check(ReportResponse)],
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
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
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

        response = multi_agent_system_orchestrator.run(request_with_date)

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

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
