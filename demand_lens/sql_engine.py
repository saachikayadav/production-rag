"""Natural-language planning plus defense-in-depth read-only SQL execution."""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QueryPlan:
    sql: str
    parameters: tuple[Any, ...]
    description: str


class UnsupportedQuestion(ValueError):
    pass


class SafeSQLExecutor:
    BANNED = re.compile(
        r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|reindex|trigger)\b",
        re.IGNORECASE,
    )

    def __init__(self, connection: sqlite3.Connection, timeout_ms: int = 750, row_limit: int = 100):
        self.connection = connection
        self.timeout_ms = timeout_ms
        self.row_limit = row_limit

    def validate(self, sql: str) -> None:
        normalized = sql.strip().rstrip(";")
        if ";" in normalized or not re.match(r"^(select|with)\b", normalized, re.IGNORECASE):
            raise ValueError("Only a single SELECT or WITH query is allowed")
        if self.BANNED.search(normalized) or "--" in normalized or "/*" in normalized:
            raise ValueError("Unsafe SQL token detected")

    def execute(self, plan: QueryPlan) -> dict[str, Any]:
        self.validate(plan.sql)
        started = time.perf_counter()
        deadline = started + self.timeout_ms / 1000

        def authorizer(action: int, _arg1: str, _arg2: str, _db: str, _source: str) -> int:
            allowed = {
                sqlite3.SQLITE_SELECT,
                sqlite3.SQLITE_READ,
                sqlite3.SQLITE_FUNCTION,
                sqlite3.SQLITE_RECURSIVE,
            }
            return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

        self.connection.set_authorizer(authorizer)
        self.connection.set_progress_handler(
            lambda: 1 if time.perf_counter() > deadline else 0,
            1000,
        )
        try:
            explain = self.connection.execute("EXPLAIN QUERY PLAN " + plan.sql, plan.parameters).fetchall()
            cursor = self.connection.execute(plan.sql, plan.parameters)
            rows = [dict(row) for row in cursor.fetchmany(self.row_limit + 1)]
        finally:
            self.connection.set_progress_handler(None, 0)
            self.connection.set_authorizer(None)
        truncated = len(rows) > self.row_limit
        rows = rows[: self.row_limit]
        return {
            "columns": list(rows[0]) if rows else [],
            "rows": rows,
            "row_count": len(rows),
            "truncated": truncated,
            "execution_ms": round((time.perf_counter() - started) * 1000, 2),
            "query_plan": [row[3] for row in explain],
        }


def compile_question(question: str) -> QueryPlan:
    """Deterministic text-to-SQL baseline for reproducible evaluation and offline demos."""
    q = question.lower()
    product_filter = ""
    parameters: list[Any] = []
    for product in ("Model Alpha", "Model Beta", "PowerCell Home"):
        if product.lower() in q:
            product_filter = " AND p.product_name = ?"
            parameters.append(product)
            break

    region_filter = ""
    for region in ("West", "Central", "East"):
        if re.search(rf"\b{region.lower()}\b", q):
            region_filter = " AND r.region_name = ?"
            parameters.append(region)
            break

    if any(term in q for term in ("wape", "forecast accuracy", "forecast miss", "forecast error")):
        sql = f"""
            SELECT r.region_name, p.product_name,
                   ROUND(100.0 * SUM(ABS(d.actual_units - d.forecast_units)) / SUM(d.actual_units), 2) AS wape_pct,
                   SUM(d.actual_units - d.forecast_units) AS missed_units
            FROM weekly_demand d
            JOIN products p ON p.product_id = d.product_id
            JOIN regions r ON r.region_id = d.region_id
            WHERE 1=1 {product_filter} {region_filter}
            GROUP BY r.region_name, p.product_name
            ORDER BY wape_pct DESC
            LIMIT 20
        """
        return QueryPlan(sql, tuple(parameters), "Forecast accuracy by product and region")

    if "bias" in q or "over-forecast" in q or "under-forecast" in q:
        sql = f"""
            SELECT r.region_name, p.product_name,
                   ROUND(100.0 * SUM(d.forecast_units - d.actual_units) / SUM(d.actual_units), 2) AS forecast_bias_pct
            FROM weekly_demand d
            JOIN products p ON p.product_id = d.product_id
            JOIN regions r ON r.region_id = d.region_id
            WHERE 1=1 {product_filter} {region_filter}
            GROUP BY r.region_name, p.product_name
            ORDER BY ABS(forecast_bias_pct) DESC
            LIMIT 20
        """
        return QueryPlan(sql, tuple(parameters), "Forecast bias by product and region")

    if any(term in q for term in ("shortage", "inventory risk", "stock risk", "at risk")):
        sql = f"""
            WITH recent AS (
                SELECT product_id, region_id, AVG(actual_units) * 4 AS expected_four_week_demand
                FROM weekly_demand
                WHERE week_start >= (SELECT date(MAX(week_start), '-28 days') FROM weekly_demand)
                GROUP BY product_id, region_id
            )
            SELECT r.region_name, p.product_name, i.on_hand_units, i.inbound_units,
                   i.safety_stock_units, ROUND(recent.expected_four_week_demand, 0) AS expected_four_week_demand,
                   ROUND(i.on_hand_units + i.inbound_units - i.safety_stock_units - recent.expected_four_week_demand, 0) AS projected_surplus
            FROM inventory i
            JOIN recent USING (product_id, region_id)
            JOIN products p ON p.product_id = i.product_id
            JOIN regions r ON r.region_id = i.region_id
            WHERE 1=1 {product_filter} {region_filter}
            ORDER BY projected_surplus ASC
            LIMIT 20
        """
        return QueryPlan(sql, tuple(parameters), "Inventory position relative to four-week demand and safety stock")

    if any(term in q for term in ("demand", "actual units", "sales")):
        sql = f"""
            SELECT d.week_start, r.region_name, p.product_name,
                   SUM(d.actual_units) AS actual_units, SUM(d.forecast_units) AS forecast_units
            FROM weekly_demand d
            JOIN products p ON p.product_id = d.product_id
            JOIN regions r ON r.region_id = d.region_id
            WHERE 1=1 {product_filter} {region_filter}
            GROUP BY d.week_start, r.region_name, p.product_name
            ORDER BY d.week_start DESC
            LIMIT 30
        """
        return QueryPlan(sql, tuple(parameters), "Weekly actual and forecast demand")

    raise UnsupportedQuestion("The deterministic SQL planner does not support this data question yet")
