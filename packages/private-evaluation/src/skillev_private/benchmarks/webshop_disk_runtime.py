#!/usr/bin/env python3
# mypy: ignore-errors
"""Disk-backed data surfaces for the pinned official WebShop simulator.

The official simulator expects a product sequence, an ASIN mapping, a price
mapping, a shuffled goal list, and a Lucene searcher.  This module supplies
those same surfaces from the immutable SQLite/JSONL assets produced by
``webshop_bulk_preparation`` without loading the multi-gigabyte product array
into RAM.  It intentionally imports no SKILLEV package so the standalone
official-environment worker can load it in the pinned WebShop interpreter.
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections.abc import Mapping, Sequence


def _product(row):
    if row is None:
        raise KeyError("unknown WebShop product")
    value = json.loads(row[0])
    if not isinstance(value, dict):
        raise TypeError("stored WebShop product must be an object")
    return value


class DiskProductSequence(Sequence):
    """Official ``all_products`` sequence backed by sequential source IDs."""

    def __init__(self, connection, product_count):
        self._connection = connection
        self._product_count = product_count

    def __len__(self):
        return self._product_count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(self._product_count))]
        if not isinstance(index, int):
            raise TypeError("WebShop product index must be an integer")
        if index < 0:
            index += self._product_count
        if index < 0 or index >= self._product_count:
            raise IndexError(index)
        row = self._connection.execute(
            "SELECT product_json FROM products WHERE source_id = ?",
            (index,),
        ).fetchone()
        return _product(row)

    def __iter__(self):
        rows = self._connection.execute("SELECT product_json FROM products ORDER BY source_id")
        for row in rows:
            yield _product(row)


class DiskProductMapping(Mapping):
    """Official ``product_item_dict`` mapping backed by the ASIN index."""

    def __init__(self, connection, product_count):
        self._connection = connection
        self._product_count = product_count

    def __len__(self):
        return self._product_count

    def __iter__(self):
        rows = self._connection.execute("SELECT asin FROM products ORDER BY source_id")
        for row in rows:
            yield row[0]

    def __getitem__(self, asin):
        if not isinstance(asin, str):
            raise KeyError(asin)
        row = self._connection.execute(
            "SELECT product_json FROM products WHERE asin = ?",
            (asin,),
        ).fetchone()
        return _product(row)


class DiskPriceMapping(Mapping):
    """Official ``product_prices`` mapping backed by the same product row."""

    def __init__(self, connection, product_count):
        self._connection = connection
        self._product_count = product_count

    def __len__(self):
        return self._product_count

    def __iter__(self):
        rows = self._connection.execute("SELECT asin FROM products ORDER BY source_id")
        for row in rows:
            yield row[0]

    def __getitem__(self, asin):
        if not isinstance(asin, str):
            raise KeyError(asin)
        row = self._connection.execute(
            "SELECT price FROM products WHERE asin = ?",
            (asin,),
        ).fetchone()
        if row is None:
            raise KeyError(asin)
        return float(row[0])


class DiskProductStore:
    """Read-only owner of the three official product collection views."""

    def __init__(self, store_path):
        uri = f"file:{store_path}?mode=ro&immutable=1"
        self.connection = sqlite3.connect(uri, uri=True, timeout=60.0)
        metadata = dict(self.connection.execute("SELECT key, value FROM build_meta"))
        if metadata.get("phase") != "complete":
            self.connection.close()
            raise ValueError("WebShop product store is not complete")
        product_count = int(metadata["accepted"])
        actual_count = self.connection.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        if product_count <= 0 or actual_count != product_count:
            self.connection.close()
            raise ValueError("WebShop product-store count differs from metadata")
        self.products = DiskProductSequence(self.connection, product_count)
        self.product_mapping = DiskProductMapping(self.connection, product_count)
        self.price_mapping = DiskPriceMapping(self.connection, product_count)

    def close(self):
        self.connection.close()


def load_goals(goals_path):
    goals = []
    with open(goals_path, encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("stored WebShop goal must be an object")
            goals.append(value)
    if not goals:
        raise ValueError("WebShop goal stream is empty")
    return goals


def create_disk_backed_server(
    *,
    web_agent_module,
    product_store_path,
    goals_path,
    search_index_path,
    searcher_factory,
    show_attrs=False,
):
    """Build an official ``SimServer`` instance without its eager loader."""

    goals = load_goals(goals_path)
    random.seed(233)
    random.shuffle(goals)
    weights = [goal["weight"] for goal in goals]
    cumulative = 0.0
    cumulative_weights = [0.0]
    for weight in weights:
        cumulative += float(weight)
        cumulative_weights.append(cumulative)
    search_engine = searcher_factory(search_index_path)
    store = DiskProductStore(product_store_path)

    server = web_agent_module.SimServer.__new__(web_agent_module.SimServer)
    server.base_url = "http://127.0.0.1:3000"
    server.all_products = store.products
    server.product_item_dict = store.product_mapping
    server.product_prices = store.price_mapping
    server.search_engine = search_engine
    server.goals = goals
    server.show_attrs = show_attrs
    server.weights = weights
    server.cum_weights = cumulative_weights
    server.user_sessions = {}
    server.search_time = 0
    server.render_time = 0
    server.sample_time = 0
    server.assigned_instruction_text = None
    server._skillev_product_store = store
    return server


def close_disk_backed_server(server):
    """Close the immutable SQLite handle owned by a disk-backed server."""

    store = server._skillev_product_store
    store.close()


__all__ = [
    "DiskPriceMapping",
    "DiskProductMapping",
    "DiskProductSequence",
    "DiskProductStore",
    "close_disk_backed_server",
    "create_disk_backed_server",
    "load_goals",
]
