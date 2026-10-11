import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
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
    return {row["item_name"]: int(row["current_stock"]) for _, row in inventory_info.iterrows()}

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
        "delivery estimates, and stock reorders.\n"
        "- Call the quote generator when a customer asks for a quote or pricing "
        "guidance. Include relevant request details and any inventory findings.\n"
        "- Call the sales agent only when a customer clearly authorizes a purchase and "
        "the item, quantity, total price, and transaction date are known.\n"
        "- Call the reporting agent for cash balances, inventory valuations, total "
        "assets, or top-selling product reports.\n\n"
        "When forwarding a task to a specialist, always formulate a clear request that includes all relevant details and context. "
        "For a customer quote or purchase request that names an item and quantity, "
        "check on-hand stock with the inventory manager before quoting availability or "
        "asking the sales agent to record a sale. Reject a purchase request if the item is not in stock and the expected delivery date is higher than the customer-specified date. If the item is in stock, confirm the available quantity and any delivery estimate before quoting a price or recording a sale.\n\n"
        "If delivery in time is realistic, call the inventory manager again and explicitly tell "
        "it to place a stock order for the shortfall using tool_execute_order with "
        'transaction_type="stock_orders". Pass along the item, shortfall quantity, '
        "request or transaction date, and any supplier purchase price provided in the "
        "request or context. Do not invent a missing price or date; have the inventory "
        "manager request missing details, and do not say the order was placed unless "
        "it confirms successful execution.\n\n"
        "Treat newly ordered stock as incoming, not as currently on-hand. Use the "
        "inventory manager's delivery estimate when explaining when it may become "
        "available."
        "Include the stock and delivery findings (also dates) in any follow-up call to "
        "the quote generator or sales agent.\n\n"
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
        def call_inventory_manager(request: str) -> str:
            """Delegate inventory and supplier delivery questions to the inventory manager.

            Args:
                request: The inventory-related request.

            Returns:
                The inventory manager's response.
            """
            return self.inventory_manager_agent.run(request)

        @tool
        def call_quote_generator(request: str) -> str:
            """Delegate customer quote requests to the quote generator.

            Args:
                request: The customer's quote request.

            Returns:
                The quote generator's response.
            """
            return self.quote_generator_agent.run(request)

        @tool
        def call_sales_agent(request: str) -> str:
            """Delegate customer orders to the sales agent.

            Args:
                request: The order or sales task to process.

            Returns:
                The sales agent's response.
            """
            return self.sales_agent.run(request)

        @tool
        def call_reporting_agent(request: str) -> str:
            """Delegate financial reporting requests to the reporting agent.

            Args:
                request: The financial report request.

            Returns:
                The reporting agent's response.
            """
            return self.reporting_agent.run(request)

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
        "Answer inventory questions using the inventory tools and the requested as-of "
        "date. Report stock quantities accurately, and use the delivery timeline tool "
        "when estimating arrival dates. Place a stock reorder when explicitly asked to "
        "do so. Before ordering, make sure the item name, quantity, total purchase "
        "price, and transaction date are known; ask for any missing details instead of "
        "guessing. Record the order with tool_execute_order using "
        'transaction_type="stock_orders". Report that the reorder was placed only '
        "after the tool confirms it. Do not use this process to record customer sales."
    )

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            model=model,
            tools=[
                tool_check_inventory,
                tool_check_stock_level,
                tool_check_delivery_timeline,
                tool_execute_order,
            ],
            instructions=self.PROMPT,
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
        "Use the quote history tool to find relevant examples, then explain how they "
        "relate to the customer's request. Treat historical quotes as context, not as "
        "current prices. Apply only discount rules and prices provided in the request "
        "or other supplied context; do not invent prices or discounts. Ask for missing "
        "details needed to prepare a reliable quote, and make clear when a quote is "
        "only an estimate."
    )

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            model=model,
            tools=[tool_read_quote_history],
            instructions=self.PROMPT,
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
        "Before recording a sale, use the stock tool to verify the requested quantity "
        "is available. Record a sale only when the customer has clearly authorized it "
        "and the item, quantity, total price, and transaction date are known. Use "
        'transaction_type="sales" for customer purchases. Do not invent prices or '
        "promise a delivery date that has not been confirmed."
    )

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            model=model,
            tools=[tool_execute_order, tool_check_stock_level],
            instructions=self.PROMPT,
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
        "Generate reports using the financial report tool and the requested as-of date. "
        "Summarize cash balance, inventory value, and total assets; include itemized "
        "inventory or top-selling products when relevant. Clearly label the report date "
        "and distinguish reported figures from any interpretation. Do not infer values "
        "the report does not provide."
    )

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            model=model,
            tools=[tool_generate_financial_report],
            instructions=self.PROMPT,
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
