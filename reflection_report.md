# Reflection Report: Beaver's Choice Paper Company Multi-Agent System

## 1. Agent Workflow and Architecture

This text-based system uses five `smolagents` agents to manage customer
enquiries, stock, quotes, sales, and reporting without exceeding the project
limit. The design separates customer coordination and reorder decisions from
inventory execution, pricing, sales transactions, and financial reporting.

The workflow is shown in [architecture_diagram.mmd](./docs/architecture_diagram.mmd)
and its rendered [architecture diagram](./architecture_diagram.png). Blue
nodes are agents, green nodes are tools, yellow nodes are underlying helper
functions, and the red node is the shared SQLite database.

### Agent Roles and Design Rationale

A single general-purpose agent would mix read-only checks with financial and
inventory updates. Instead, the system uses one customer-facing coordinator and
four focused specialists:

1. **Orchestrator Agent:** Receives every request, resolves product names,
   selects the required agents, decides whether a shortfall may be reordered,
   and produces the final customer response.
2. **Inventory Manager Agent:** Performs read-only stock and supplier-delivery
   checks, then executes a reorder only when the orchestrator explicitly
   authorizes it.
3. **Quote Generator Agent:** Creates item-level, history-informed price
   estimates.
4. **Sales Agent:** Records an all-or-nothing customer sale after a final
   availability check.
5. **Reporting Agent:** Produces dated cash, inventory, asset, and sales
   reports.

This allocation keeps authority clear: Inventory supplies evidence and carries
out approved stock purchases; the Orchestrator makes the business decision;
Quote prices the request; Sales records the completed transaction. It also
assigns all starter helpers: stock and inventory functions to Inventory,
historical quote searching to Quote, transaction creation to Inventory and
Sales, and financial functions to Reporting.

The design deliberately separates a stock decision from its execution. The
Inventory Manager has the operational detail needed to identify a shortage and
forecast supplier lead time, but it cannot independently spend company funds
during its first check. The Orchestrator considers that evidence alongside the
customer's delivery commitment before authorizing a reorder. This avoids a
quote or sale being based on assumed incoming stock and keeps the customer
interaction, stock-purchase authority, and transaction execution auditable.

### Quote and Purchase Workflow

For a customer request containing products and quantities, the agents work
together in the following order:

1. The **Orchestrator** classifies the request and resolves every product name
   with `tool_resolve_product_name`. Alias and fuzzy matching are supported.
   Unknown products are rejected; ambiguous products require clarification.
2. The Orchestrator converts any requested delivery deadline to a standard date
   and sends all items to **Inventory** with `action: "check"`. This call is
   read-only.
3. **Inventory** checks stock through `tool_check_stock_level`, calculates any
   shortfall, and estimates a supplier arrival date with
   `tool_check_delivery_timeline`. It returns itemized stock, shortfall, and
   delivery information, but it cannot place an order in check mode.
4. The **Orchestrator** evaluates the returned evidence. If all items are in
   stock, it continues. If there is a shortfall, it compares estimated arrivals
   to the customer's deadline with `tool_check_deadline`; arrival on the
   deadline is acceptable. A late item causes the complete request to be
   rejected before quoting or selling.
5. When a shortfall has no deadline, or can meet its deadline, the
   **Orchestrator** decides to reorder. It sends Inventory a separate
   `action: "reorder"` request containing only the missing quantities.
6. **Inventory** retrieves unit prices, calculates purchase costs, and records
   the approved stock orders using `tool_execute_order`. It confirms a reorder
   only after the database write succeeds. A failed reorder rejects the entire
   customer request.
7. When stock is sufficient or replenishment is confirmed, **Quote** receives
   the normalized items and inventory findings. It searches historical quotes
   with `tool_read_quote_history` and returns item-level estimates. When an
   item is not present in the stocked inventory table, its unit price can fall
   back to the `paper_supplies` product catalogue. The Orchestrator calculates
   the total.
8. For a purchase, the **Sales Agent** receives the quoted prices and final
   total. It performs a final all-item stock check and records every sale with
   `tool_execute_order` only if all items are available. In this project,
   confirmed deadline-safe replenishment is treated as available for the sale.
9. The **Orchestrator** returns a concise response containing confirmed
   outcomes, items, prices, and delivery dates. It does not expose internal
   agents, tools, or transaction identifiers.

Standalone inventory questions use only Inventory's read-only check. Financial
or operational requests use the **Reporting Agent**, which calls
`tool_get_cash_balance` and `tool_generate_financial_report`.

The workflow uses the request date consistently as the stock-check and
transaction-posting date. A confirmed reorder that can meet the requested
delivery deadline is treated as available for the associated project sale,
even when the supplier's estimated arrival is later than the booking date.
This project-specific rule enables the system to reserve deadline-safe incoming
stock while still rejecting requests that cannot be fulfilled on time.

### Controls and Reliability

- **All-or-nothing fulfillment:** A product-resolution failure, late delivery,
  failed reorder, missing price, or failed final stock check rejects the whole
  purchase. No partial sale is recorded.
- **Structured handoffs:** Pydantic models validate specialist requests and
  responses before they are used.
- **Controlled state changes:** Inventory can write only after an explicit
  reorder authorization; Sales can write only after its final availability
  check.
- **Bounded and traceable execution:** Agent tool calls and delegations have
  soft limits, while every handoff is recorded in `handoff_trace.jsonl`.
- **Safe calculation:** Pricing and order arithmetic use a restricted
  calculator tool rather than arbitrary expression evaluation.

Together, these controls ensure that customer-facing statements are grounded
in specialist responses and database actions. A transaction is not reported as
completed merely because an agent proposed it: the responsible specialist must
confirm the relevant database operation first.

## 2. Project Structure and Files

```text
beavers_choice_paper_company/
├── docs/
│   ├── architecture_diagram.mmd       # Editable Mermaid workflow source
│   └── architecture_diagram.svg       # Scalable diagram rendering
├── architecture_diagram.png           # Raster workflow rendering
├── reflection_report.md               # This report
├── project_starter.py                 # Database helpers, tools, agents, and runner
├── requirements.txt                   # Python dependencies
├── README.md                          # Setup and project guidance
├── quote_requests.csv                 # Historical customer requests
├── quotes.csv                         # Historical quotes for pricing context
├── quote_requests_sample.csv          # Input scenarios for evaluation
├── test_results.csv                   # Generated scenario results
├── munder_difflin.db                  # Generated SQLite business data
├── handoff_trace.jsonl                # Generated agent-handoff trace
├── run.log                            # Generated runtime output
├── .env.example                       # Environment-variable template
├── .gitignore                         # Local and generated file exclusions
└── .vscode/launch.json                # VS Code debugging configuration
```

### Key Files

- [project_starter.py](./project_starter.py) contains database initialization,
  helper functions, tool wrappers, Pydantic contracts, the five agent classes,
  handoff tracing, and `run_test_scenarios()`.
- [architecture_diagram.mmd](./docs/architecture_diagram.mmd) is the editable
  architecture source. [architecture_diagram.png](./architecture_diagram.png)
  and [architecture_diagram.svg](./docs/architecture_diagram.svg) are its
  rendered versions.
- [quote_requests.csv](./quote_requests.csv) and [quotes.csv](./quotes.csv)
  provide historical data for quote context;
  [quote_requests_sample.csv](./quote_requests_sample.csv) supplies the dated
  evaluation scenarios.
- [test_results.csv](./test_results.csv) records each request, its resulting
  financial state, and its customer response. [munder_difflin.db](./munder_difflin.db)
  stores inventory and transactions, while [handoff_trace.jsonl](./handoff_trace.jsonl)
  records the agent interactions.
- [.env.example](./.env.example) documents the API-key configuration format.
  The real `.env`, generated database, logs, and traces remain excluded from
  version control through [.gitignore](./.gitignore).

The test runner in [project_starter.py](./project_starter.py) initializes the
database, orders the sample requests by date, processes each scenario through
the Orchestrator, and writes the accumulated results to
[test_results.csv](./test_results.csv). This makes the evaluation repeatable
and preserves the operational state required to inspect the effect of earlier
transactions on later requests.

## 3. Evaluation Results

The generated [test_results.csv](./test_results.csv) contains 20 dated
scenarios from 1 to 17 April 2025. Each row records the request, the
post-scenario cash balance and inventory value, and the customer response.
The tests cover successful orders, stock shortfalls, delivery deadlines,
and unsupported products.

| Outcome | Requests | Evidence |
| --- | ---: | --- |
| Completed or successfully placed | 9 | Requests 1, 5-12 |
| Rejected | 11 | Unsupported items, insufficient stock, or impossible deadlines |

Successful requests demonstrate full processing. Request 1 provided an
itemized order for Glossy paper, Cardstock, and Colored paper, including an
estimated delivery date for replenished Glossy paper. Request 5 confirmed 500
sheets of Colored paper, 300 sheets of Cardstock, and 200 rolls of decorative
adhesive tape with a total and delivery commitment. Requests 6-12 further show
that in-stock items and confirmed replenishment can be handled together.

The rejections are also valuable evidence. The system declined unsupported
products such as balloons, A3 paper, cardboard signage, and tickets, and it rejected
complete requests when required supplies could not arrive in time. It therefore
does not portray an incomplete order as fulfillable merely because some items
are available.

This result mix demonstrates both required behaviours of the project: the
system completes feasible orders and actively leaves impossible ones
unfulfilled with a customer-readable reason. The latter is especially
important for physical inventory because accepting an order without a viable
delivery path would create an incorrect commitment and undermine customer
trust.

### Observed Strengths

- **Detailed feasibility information:** Responses name the relevant products,
  quantities, stock situation, shortfalls, and limiting item rather than giving
  unexplained rejections.
- **Complete processing when feasible:** Successful scenarios show the full
  path from availability confirmation through pricing and order placement.
- **Clear delivery communication:** The system gave customer-facing arrival
  dates, including 5 April for Glossy paper in request 1, 11 and 14 April for
  replenishments in request 8, and 12 April for Cardstock in request 10.
- **Helpful communication:** Both successful and rejected responses are polite
  and explain the business reason for the outcome.
- **Traceable business updates:** Cash and inventory values are recorded after
  every scenario. After successful request 5, cash rose from $55,460.70 to
  $55,588.70 and inventory value changed from $4,974.30 to $4,837.55.

## 4. Improvement Suggestions

### 1. Introduce Data-Driven, Customer-Aware Pricing

The Quote Generator should extend its use of historical examples into a
controlled pricing strategy. It could assess purchase frequency, order volume,
accepted quotes, and customer relationship status to recommend loyalty or
volume discounts. Analysis of completed sales should identify the prices,
discount levels, products, and customer segments associated with successful
transactions, producing defined pricing ranges rather than relying on isolated
examples. A configurable seasonal-discount programme could also respond to
demand, promotions, and inventory objectives. This would improve
competitiveness and repeat business while protecting margins.

Implementation should begin with a transparent pricing policy rather than
unrestricted price generation. The system can record the reason and source data
for every adjustment, apply manager-defined discount ceilings, and measure
acceptance rate, realised margin, and repeat purchases. These controls would
allow the company to test pricing changes safely and refine the strategy using
measurable commercial results.

### 2. Strengthen Product Resolution with a Knowledge Service

The current resolver protects transactions through aliases, fuzzy matching, and
clarification of ambiguous names. A dedicated Product Resolution Agent or
service could make this process more flexible, subject to the five-agent limit
or consolidation of another role. It could use a RAG knowledge base containing
the product catalogue, aliases, synonyms, paper sizes, weights, finishes, and
other product background data.

An LLM could retrieve relevant records and return a canonical name, confidence
level, and explanation. High-confidence matches could continue automatically;
low-confidence or genuinely ambiguous cases would still request clarification.
This would reduce avoidable rejections while retaining safe uncertainty
handling, helping convert more valid enquiries into sales.

The knowledge base should be maintained as a governed catalogue resource, with
approved synonyms, source references, and regular review of unresolved terms.
Resolution results should remain structured and include the selected canonical
name and confidence level, so that downstream inventory, quote, and sales
agents continue to operate on validated product identifiers.
