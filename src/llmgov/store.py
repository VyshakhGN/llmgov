from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE customers (
    customer_id           TEXT PRIMARY KEY,
    customer_since_months INTEGER,
    total_orders          INTEGER,
    refunds_last_12m      INTEGER,
    account_flags         TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE orders (
    order_id           TEXT PRIMARY KEY,
    customer_id        TEXT REFERENCES customers(customer_id),
    order_status       TEXT,
    category           TEXT,
    delivered_days_ago INTEGER,
    is_faulty          INTEGER NOT NULL DEFAULT 0,
    return_condition   TEXT,
    order_value_eur    REAL
);
"""

CUSTOMER_FIELDS = (
    "customer_since_months",
    "total_orders",
    "refunds_last_12m",
    "account_flags",
)
ORDER_FIELDS = (
    "customer_id",
    "order_status",
    "category",
    "delivered_days_ago",
    "is_faulty",
    "return_condition",
    "order_value_eur",
)


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def create(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _row_to_customer(row: sqlite3.Row) -> dict[str, Any]:
    out = {k: row[k] for k in CUSTOMER_FIELDS if k != "account_flags"}
    out["account_flags"] = json.loads(row["account_flags"] or "[]")
    return out


def read_customers(path: str | Path) -> dict[str, dict[str, Any]]:
    with connect(path) as conn:
        rows = conn.execute("SELECT * FROM customers ORDER BY customer_id").fetchall()
    return {row["customer_id"]: _row_to_customer(row) for row in rows}


def read_orders(path: str | Path) -> dict[str, dict[str, Any]]:
    """Order rows keyed by id, shaped like the YAML the loader used to read."""
    with connect(path) as conn:
        rows = conn.execute("SELECT * FROM orders ORDER BY order_id").fetchall()
    orders: dict[str, dict[str, Any]] = {}
    for row in rows:
        facts = {k: row[k] for k in ORDER_FIELDS}
        facts["is_faulty"] = bool(facts["is_faulty"])
        orders[row["order_id"]] = {k: v for k, v in facts.items() if v is not None}
    return orders


def seed(
    path: str | Path,
    customers: dict[str, dict[str, Any]],
    orders: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    conn = create(path)
    with conn:
        conn.executemany(
            "INSERT INTO customers (customer_id, customer_since_months, total_orders,"
            " refunds_last_12m, account_flags) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    cid,
                    row.get("customer_since_months"),
                    row.get("total_orders"),
                    row.get("refunds_last_12m"),
                    json.dumps(row.get("account_flags") or []),
                )
                for cid, row in customers.items()
            ],
        )
        conn.executemany(
            "INSERT INTO orders (order_id, customer_id, order_status, category,"
            " delivered_days_ago, is_faulty, return_condition, order_value_eur)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    oid,
                    row.get("customer_id"),
                    row.get("order_status"),
                    row.get("category"),
                    row.get("delivered_days_ago"),
                    int(bool(row.get("is_faulty"))),
                    row.get("return_condition"),
                    row.get("order_value_eur"),
                )
                for oid, row in orders.items()
            ],
        )
    conn.close()
    return len(customers), len(orders)
