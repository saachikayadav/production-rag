"""SQLite demo warehouse with realistic, synthetic demand-planning data."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS regions (
    region_id INTEGER PRIMARY KEY,
    region_name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS weekly_demand (
    week_start TEXT NOT NULL,
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    region_id INTEGER NOT NULL REFERENCES regions(region_id),
    forecast_units INTEGER NOT NULL,
    actual_units INTEGER NOT NULL,
    promotion_active INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (week_start, product_id, region_id)
);
CREATE TABLE IF NOT EXISTS inventory (
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    region_id INTEGER NOT NULL REFERENCES regions(region_id),
    on_hand_units INTEGER NOT NULL,
    safety_stock_units INTEGER NOT NULL,
    inbound_units INTEGER NOT NULL,
    lead_time_days INTEGER NOT NULL,
    PRIMARY KEY (product_id, region_id)
);
CREATE TABLE IF NOT EXISTS planning_documents (
    document_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    section TEXT NOT NULL,
    effective_date TEXT NOT NULL,
    content TEXT NOT NULL
);
"""


DOCUMENTS = [
    (
        "forecast-metrics-1",
        "Demand Planning Metrics Handbook",
        "Forecast bias",
        "2026-01-01",
        "Forecast bias is calculated as SUM(forecast_units - actual_units) divided by SUM(actual_units). Positive bias means over-forecasting; negative bias means under-forecasting. Review absolute bias above 10 percent.",
    ),
    (
        "forecast-metrics-2",
        "Demand Planning Metrics Handbook",
        "Weighted absolute percentage error",
        "2026-01-01",
        "WAPE is SUM(ABS(actual_units - forecast_units)) divided by SUM(actual_units). Lower values indicate better forecast accuracy. Aggregate the numerator and denominator before division; do not average row-level percentage errors.",
    ),
    (
        "inventory-policy-1",
        "Inventory Risk Policy",
        "Shortage risk",
        "2026-02-15",
        "A product-region is at shortage risk when on-hand plus confirmed inbound inventory is below safety stock plus expected four-week demand. High risk requires planner review and supplier escalation.",
    ),
    (
        "promotion-guide-1",
        "Promotion Forecasting Guide",
        "Promotional uplift",
        "2025-11-01",
        "Promotional weeks should be evaluated separately from baseline demand. Planners should compare promoted demand with the preceding four non-promotional weeks and record the uplift assumption.",
    ),
    (
        "governance-1",
        "Analytics Governance Standard",
        "AI-generated SQL",
        "2026-03-01",
        "AI-generated queries must use a read-only role, expose the generated SQL, enforce a row limit and timeout, and reject data-definition or data-modification statements before execution.",
    ),
]


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    connection = sqlite3.connect(str(path), check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA)
    if connection.execute("SELECT COUNT(*) FROM products").fetchone()[0]:
        return

    products = [
        (1, "Model Alpha", "EV", 42990),
        (2, "Model Beta", "EV", 52990),
        (3, "PowerCell Home", "Energy", 9990),
    ]
    regions = [(1, "West"), (2, "Central"), (3, "East")]
    connection.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", products)
    connection.executemany("INSERT INTO regions VALUES (?, ?)", regions)

    start = date(2026, 1, 5)
    demand_rows = []
    for week in range(12):
        week_start = (start + timedelta(days=7 * week)).isoformat()
        for product_id in range(1, 4):
            for region_id in range(1, 4):
                baseline = 180 + product_id * 55 + region_id * 28 + week * 5
                promotion = int(week in (4, 9) and region_id == 1)
                actual = baseline + (70 if promotion else 0) + ((week + product_id) % 5 - 2) * 11
                # West/Model Beta is deliberately under-forecast to create a useful anomaly.
                error = -58 if product_id == 2 and region_id == 1 else ((week + region_id) % 4 - 1) * 14
                forecast = max(1, actual + error)
                demand_rows.append((week_start, product_id, region_id, forecast, actual, promotion))
    connection.executemany("INSERT INTO weekly_demand VALUES (?, ?, ?, ?, ?, ?)", demand_rows)

    inventory_rows = []
    for product_id in range(1, 4):
        for region_id in range(1, 4):
            on_hand = 1050 - product_id * 120 - region_id * 65
            if product_id == 2 and region_id == 1:
                on_hand = 310
            inventory_rows.append((product_id, region_id, on_hand, 420, 180, 21 + product_id * 3))
    connection.executemany("INSERT INTO inventory VALUES (?, ?, ?, ?, ?, ?)", inventory_rows)
    connection.executemany("INSERT INTO planning_documents VALUES (?, ?, ?, ?, ?)", DOCUMENTS)
    connection.commit()


def schema_catalog(connection: sqlite3.Connection) -> list[dict[str, str]]:
    """Convert database metadata into documents used for schema retrieval."""
    docs: list[dict[str, str]] = []
    tables = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    for table_row in tables:
        table = table_row["name"]
        columns = connection.execute(f"PRAGMA table_info({table})").fetchall()
        column_text = ", ".join(
            f"{column['name']} {column['type']}{' primary key' if column['pk'] else ''}"
            for column in columns
        )
        docs.append(
            {
                "id": f"schema-{table}",
                "title": f"Database schema: {table}",
                "section": "Schema catalog",
                "content": f"Table {table} contains: {column_text}.",
                "source_type": "schema",
            }
        )
    return docs
