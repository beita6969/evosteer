"""Memory-bounded preparation of the pinned full WebShop product corpus.

The official WebShop loader materialises the 5.48 GB product JSON array and
all derived product objects in one Python process.  That exceeds the bounded
local WSL memory budget.  This module preserves the official transformation
and RNG order while streaming one raw product at a time into:

* a sequential SQLite product store used by the disk-backed official runtime;
* a Pyserini ``JsonCollection`` document stream; and
* the unshuffled official human-goal stream.

The base-table phase is resumable because the earlier local retrieval build
was genuinely lost after a WSL failure.  Checkpoints commit source position,
RNG state, SQLite rows, and the exact durable document-stream offset together.
No published source or completed output is removed or replaced.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import BinaryIO, cast

import ijson  # type: ignore[import-untyped]

from skillev.contracts import JsonValue, canonical_json, stable_hash
from skillev.experiments import FIXED_SEED

from .acquisition import (
    _read_published_canonical_record,
    load_benchmark_acquisition_lock,
)
from .webshop_bulk_types import (
    WEBSHOP_BULK_PREPARATION_FORMAT,
    WEBSHOP_PRODUCT_STORE_FORMAT,
    WebShopBulkPreparationReceipt,
    WebShopPreparationProgress,
    WebShopSourceIdentity,
    _exact_object,
    _optional_positive_int,
    _positive_int,
    _sha256_file,
    _text,
    locked_webshop_sources,
)

_STORE_NAME = "products.sqlite3"
_DOCUMENTS_NAME = "documents.jsonl"
_GOALS_NAME = "goals.jsonl"
_MANIFEST_NAME = "preparation.manifest.json"
_BUILDING_SUFFIX = ".building"
_PRICE_RANGE = tuple(10.0 * index for index in range(1, 100))
_DROP_PRODUCT_KEYS = (
    "product_information",
    "brand",
    "brand_url",
    "list_price",
    "availability_quantity",
    "availability_status",
    "total_reviews",
    "total_answered_questions",
    "seller_id",
    "seller_name",
    "fulfilled_by_amazon",
    "fast_track_message",
    "aplus_present",
    "small_description_old",
)


def _tuple_tree(value: object) -> object:
    if type(value) is list:
        return tuple(_tuple_tree(item) for item in cast(list[object], value))
    return value


def _rng_state_value(rng: random.Random) -> str:
    return json.dumps(rng.getstate(), separators=(",", ":"))


def _restore_rng_state(rng: random.Random, value: str) -> None:
    state = _tuple_tree(json.loads(value))
    if type(state) is not tuple:
        raise TypeError("WebShop RNG checkpoint must decode to a tuple")
    rng.setstate(cast(tuple[int, tuple[int, ...], float | None], state))


class _CountingReader:
    """Count bytes consumed by ijson without retaining source content."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        data = self._stream.read(size)
        self.bytes_read += len(data)
        return data


def _load_attribute_map(path: Path) -> dict[str, list[object]]:
    attributes: dict[str, list[object]] = {}
    with path.open("rb") as stream:
        for asin, raw in ijson.kvitems(stream, "", use_float=True):
            if type(asin) is not str:
                raise TypeError("WebShop attribute ASIN must be text")
            row = _exact_object(raw, label="WebShop attribute row")
            value = row.get("attributes")
            if value is None:
                continue
            if type(value) is not list:
                raise TypeError("WebShop attributes must be an array")
            attributes[asin] = cast(list[object], value)
    if not attributes:
        raise ValueError("WebShop attribute source is empty")
    return attributes


def _load_human_instruction_map(path: Path) -> dict[str, list[object]]:
    instructions: dict[str, list[object]] = {}
    with path.open("rb") as stream:
        for asin, raw in ijson.kvitems(stream, "", use_float=True):
            if type(asin) is not str or type(raw) is not list:
                raise TypeError("WebShop human instructions must map ASINs to arrays")
            instructions[asin] = cast(list[object], raw)
    if not instructions:
        raise ValueError("WebShop human-instruction source is empty")
    return instructions


def _normalize_product(
    raw: object,
    *,
    attributes: Mapping[str, list[object]],
    human_instructions: Mapping[str, list[object]],
) -> dict[str, object] | None:
    product = dict(_exact_object(raw, label="WebShop product"))
    for key in _DROP_PRODUCT_KEYS:
        product.pop(key, None)

    asin = _text(product["asin"], field="WebShop product ASIN")
    if asin == "nan" or len(asin) > 10:
        return None

    product["category"] = product["category"]
    product["query"] = product["query"]
    product["product_category"] = product["product_category"]
    product["Title"] = product["name"]
    product["Description"] = product["full_description"]
    product["Reviews"] = []
    product["Rating"] = "N.A."
    small_description = product["small_description"]
    product["BulletPoints"] = (
        small_description if type(small_description) is list else [small_description]
    )

    pricing_raw = product.get("pricing")
    if pricing_raw is None or not pricing_raw:
        pricing = [100.0]
        price_tag = "$100.0"
    else:
        pricing_text = _text(pricing_raw, field="WebShop pricing")
        pricing = [
            float(Decimal(re.sub(r"[^\d.]", "", component)))
            for component in pricing_text.split("$")[1:]
        ]
        if len(pricing) == 1:
            price_tag = f"${pricing[0]}"
        else:
            price_tag = f"${pricing[0]} to ${pricing[1]}"
            pricing = pricing[:2]
    product["pricing"] = pricing
    product["Price"] = price_tag

    options: dict[str, list[str]] = {}
    option_to_image: dict[str, object] = {}
    customization_options = product["customization_options"]
    if customization_options:
        custom = _exact_object(customization_options, label="WebShop customization options")
        for option_name_raw, option_contents in custom.items():
            if option_contents is None:
                continue
            option_name = option_name_raw.lower()
            if type(option_contents) is not list:
                raise TypeError("WebShop option contents must be an array")
            option_values: list[str] = []
            for raw_content in option_contents:
                content = _exact_object(raw_content, label="WebShop option content")
                option_value = (
                    _text(content["value"], field="WebShop option value")
                    .strip()
                    .replace("/", " | ")
                    .lower()
                )
                option_values.append(option_value)
                option_to_image[option_value] = content.get("image")
            options[option_name] = option_values
    product["options"] = options
    product["option_to_image"] = option_to_image
    product["Attributes"] = attributes.get(asin, ["DUMMY_ATTR"])
    if asin in human_instructions:
        product["instructions"] = human_instructions[asin]
    images = product["images"]
    if type(images) is not list or not images:
        raise ValueError("WebShop product images must be a non-empty array")
    product["MainImage"] = images[0]
    product["query"] = (
        _text(
            product["query"],
            field="WebShop query",
            allow_empty=True,
        )
        .lower()
        .strip()
    )
    return product


def _product_price(product: Mapping[str, object], rng: random.Random) -> float:
    pricing = product["pricing"]
    if type(pricing) is not list:
        raise TypeError("normalized WebShop pricing must be an array")
    if not pricing:
        return 100.0
    if len(pricing) == 1:
        return float(pricing[0])
    return rng.uniform(float(pricing[0]), float(pricing[1]))


def _search_document(product: Mapping[str, object]) -> dict[str, object]:
    options = _exact_object(product.get("options"), label="normalized WebShop options")
    option_texts: list[str] = []
    for option_name, option_contents in options.items():
        if type(option_contents) is not list or any(
            type(value) is not str for value in option_contents
        ):
            raise TypeError("normalized WebShop option values must be text arrays")
        option_texts.append(f"{option_name}: {', '.join(cast(list[str], option_contents))}")
    bullet_points = product["BulletPoints"]
    if type(bullet_points) is not list or not bullet_points:
        raise ValueError("normalized WebShop BulletPoints must be non-empty")
    contents = " ".join(
        (
            _text(product["Title"], field="WebShop Title", allow_empty=True),
            _text(product["Description"], field="WebShop Description", allow_empty=True),
            _text(bullet_points[0], field="WebShop first BulletPoint", allow_empty=True),
            ", and ".join(option_texts),
        )
    ).lower()
    return {
        "contents": contents,
        "id": _text(product["asin"], field="WebShop product ASIN"),
    }


def _configure_connection(path: Path, *, create: bool) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=120.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-524288")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    connection.execute("PRAGMA wal_autocheckpoint=0")
    if create:
        connection.executescript(
            """
            CREATE TABLE build_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE products (
                source_id INTEGER PRIMARY KEY,
                asin TEXT NOT NULL,
                product_json TEXT NOT NULL,
                price REAL NOT NULL,
                category TEXT NOT NULL,
                query TEXT NOT NULL
            );
            CREATE TABLE goal_products (
                source_id INTEGER PRIMARY KEY,
                asin TEXT NOT NULL,
                category TEXT NOT NULL,
                query TEXT NOT NULL,
                name TEXT NOT NULL,
                product_category TEXT NOT NULL,
                instructions_json TEXT NOT NULL,
                price REAL NOT NULL
            );
            """
        )
    return connection


def _meta(connection: sqlite3.Connection) -> dict[str, str]:
    rows = connection.execute("SELECT key, value FROM build_meta").fetchall()
    return dict(cast(list[tuple[str, str]], rows))


def _set_meta(connection: sqlite3.Connection, values: Mapping[str, str]) -> None:
    connection.executemany(
        """
        INSERT INTO build_meta(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        tuple(values.items()),
    )


def _checkpoint(
    *,
    connection: sqlite3.Connection,
    documents: BinaryIO,
    raw_seen: int,
    accepted: int,
    rng: random.Random,
) -> int:
    documents.flush()
    os.fsync(documents.fileno())
    offset = documents.tell()
    _set_meta(
        connection,
        {
            "accepted": str(accepted),
            "documents_offset": str(offset),
            "raw_seen": str(raw_seen),
            "rng_state": _rng_state_value(rng),
        },
    )
    connection.commit()
    connection.execute("PRAGMA wal_checkpoint(PASSIVE)")
    return offset


def _rebuild_checkpointed_documents(
    connection: sqlite3.Connection,
    documents_path: Path,
    *,
    expected_offset: int,
) -> None:
    with documents_path.open("xb") as documents:
        rows = connection.execute("SELECT product_json FROM products ORDER BY source_id")
        for (product_json,) in rows:
            product = _exact_object(
                json.loads(cast(str, product_json)),
                label="stored WebShop product",
            )
            documents.write(
                (json.dumps(_search_document(product), ensure_ascii=True) + "\n").encode("utf-8")
            )
        documents.flush()
        os.fsync(documents.fileno())
        if documents.tell() != expected_offset:
            raise ValueError("rebuilt WebShop document stream differs from its durable checkpoint")


def _goal_records(
    connection: sqlite3.Connection,
    rng: random.Random,
) -> tuple[dict[str, object], ...]:
    goals: list[dict[str, object]] = []
    rows = connection.execute(
        """
        SELECT asin, category, query, name, product_category, instructions_json, price
        FROM goal_products
        ORDER BY source_id
        """
    )
    for asin, category, query, name, product_category, instructions_json, price in rows:
        instructions = json.loads(cast(str, instructions_json))
        if type(instructions) is not list:
            raise TypeError("WebShop stored instructions must be an array")
        for raw_instruction in instructions:
            instruction = _exact_object(raw_instruction, label="WebShop human instruction")
            attributes = instruction["instruction_attributes"]
            if type(attributes) is not list:
                raise TypeError("WebShop goal attributes must be an array")
            if not attributes:
                continue
            price_range = [candidate for candidate in _PRICE_RANGE if candidate > price][:4]
            if len(price_range) >= 2:
                _, price_upper = sorted(rng.sample(price_range, 2))
                price_text = f", and price lower than {price_upper:.2f} dollars"
            else:
                price_upper = 1_000_000
                price_text = ""
            goals.append(
                {
                    "asin": asin,
                    "attributes": attributes,
                    "category": category,
                    "goal_options": instruction["instruction_options"],
                    "instruction_text": (
                        _text(
                            instruction["instruction"],
                            field="WebShop human instruction text",
                        ).strip(".")
                        + price_text
                    ),
                    "name": name,
                    "price_upper": price_upper,
                    "product_category": product_category,
                    "query": query,
                    "weight": 1,
                }
            )
    if not goals:
        raise ValueError("WebShop human goal inventory is empty")
    return tuple(goals)


def _progress(
    callback: Callable[[WebShopPreparationProgress], None] | None,
    *,
    phase: str,
    raw_seen: int,
    accepted: int,
    goals: int,
    phase_started: float,
    remaining_units: float | None,
) -> None:
    if callback is None:
        return
    elapsed = time.monotonic() - phase_started
    rate = raw_seen / elapsed if elapsed > 0.0 else 0.0
    eta = None
    if remaining_units is not None and rate > 0.0:
        eta = max(0.0, remaining_units / rate)
    callback(
        WebShopPreparationProgress(
            phase=phase,
            raw_products_seen=raw_seen,
            accepted_products=accepted,
            goals_written=goals,
            elapsed_seconds=float(elapsed),
            records_per_second=float(rate),
            eta_seconds=(None if eta is None else float(eta)),
        )
    )


def _source_identity_values(
    sources: Sequence[WebShopSourceIdentity],
) -> tuple[dict[str, JsonValue], ...]:
    return tuple(source.to_value() for source in sources)


def prepare_webshop_bulk_assets(
    *,
    products_source: WebShopSourceIdentity,
    attributes_source: WebShopSourceIdentity,
    human_instructions_source: WebShopSourceIdentity,
    output_root: Path,
    seed: int = FIXED_SEED,
    raw_product_limit: int | None = None,
    checkpoint_interval: int = 50_000,
    progress_interval: int = 10_000,
    progress_callback: Callable[[WebShopPreparationProgress], None] | None = None,
) -> WebShopBulkPreparationReceipt:
    """Prepare one exact full or bounded WebShop corpus without loading it."""

    sources = (products_source, attributes_source, human_instructions_source)
    if not isinstance(output_root, Path) or not output_root.is_absolute():
        raise ValueError("output_root must be an absolute Path")
    if type(seed) is not int or seed != FIXED_SEED:
        raise ValueError("WebShop preparation must use the frozen experiment seed")
    _optional_positive_int(raw_product_limit, field="raw_product_limit")
    _positive_int(checkpoint_interval, field="checkpoint_interval")
    _positive_int(progress_interval, field="progress_interval")
    if output_root.exists():
        return verify_webshop_bulk_assets(output_root)
    if not output_root.parent.is_dir():
        raise NotADirectoryError(output_root.parent)

    for source in sources:
        source.verify()
    source_values = _source_identity_values(sources)
    build_identity = stable_hash(
        {
            "format": WEBSHOP_BULK_PREPARATION_FORMAT,
            "raw_product_limit": raw_product_limit,
            "seed": seed,
            "source_identities": list(source_values),
        }
    )
    staging_root = output_root.with_name(f".{output_root.name}{_BUILDING_SUFFIX}")
    store_path = staging_root / _STORE_NAME
    documents_path = staging_root / _DOCUMENTS_NAME
    create = not staging_root.exists()
    if create:
        staging_root.mkdir()
    elif not staging_root.is_dir():
        raise NotADirectoryError(staging_root)
    if create and (store_path.exists() or documents_path.exists()):
        raise FileExistsError("new WebShop staging directory is not empty")
    if not create and (not store_path.is_file() or not documents_path.is_file()):
        raise ValueError("WebShop resumable staging is incomplete")

    connection = _configure_connection(store_path, create=create)
    try:
        if create:
            rng = random.Random(seed)  # noqa: S311 - frozen benchmark RNG
            _set_meta(
                connection,
                {
                    "accepted": "0",
                    "build_identity": build_identity,
                    "documents_offset": "0",
                    "phase": "products",
                    "raw_seen": "0",
                    "rng_state": _rng_state_value(rng),
                },
            )
            connection.commit()
        metadata = _meta(connection)
        required_meta = {
            "accepted",
            "build_identity",
            "documents_offset",
            "phase",
            "raw_seen",
            "rng_state",
        }
        if not required_meta.issubset(metadata):
            raise ValueError("WebShop staging checkpoint is incomplete")
        if metadata["build_identity"] != build_identity:
            raise ValueError("WebShop staging belongs to another exact build")
        rng = random.Random()  # noqa: S311 - restored frozen benchmark RNG
        _restore_rng_state(rng, metadata["rng_state"])
        raw_seen = int(metadata["raw_seen"])
        accepted = int(metadata["accepted"])
        documents_offset = int(metadata["documents_offset"])
        if min(raw_seen, accepted, documents_offset) < 0 or accepted > raw_seen:
            raise ValueError("WebShop staging counters are invalid")

        if metadata["phase"] == "products":
            attributes = _load_attribute_map(attributes_source.path)
            human_instructions = _load_human_instruction_map(human_instructions_source.path)
            asins = {cast(str, row[0]) for row in connection.execute("SELECT asin FROM products")}
            if len(asins) != accepted:
                raise ValueError("WebShop checkpoint ASIN count differs from accepted count")
            documents_mode = "r+b" if documents_path.exists() else "w+b"
            phase_started = time.monotonic()
            insert_rows: list[tuple[object, ...]] = []
            goal_rows: list[tuple[object, ...]] = []
            if documents_path.exists() and documents_path.stat().st_size > documents_offset:
                preserved_documents = output_root.parent / (
                    f".{output_root.name}.documents-uncommitted-"
                    f"{os.getpid()}-{time.time_ns()}.preserved"
                )
                documents_path.rename(preserved_documents)
                _rebuild_checkpointed_documents(
                    connection,
                    documents_path,
                    expected_offset=documents_offset,
                )
            with documents_path.open(documents_mode) as documents:
                documents.seek(0, os.SEEK_END)
                if documents.tell() < documents_offset:
                    raise ValueError("WebShop document stream is shorter than checkpoint")
                if documents.tell() != documents_offset:
                    raise ValueError("WebShop document stream differs from checkpoint")
                documents.seek(documents_offset)
                with products_source.path.open("rb") as raw_stream:
                    counting = _CountingReader(raw_stream)
                    try:
                        products = ijson.items(counting, "item", use_float=True)
                        for raw_index, raw_product in enumerate(products):
                            if raw_product_limit is not None and raw_index >= raw_product_limit:
                                break
                            if raw_index < raw_seen:
                                continue
                            raw_seen = raw_index + 1
                            product = _normalize_product(
                                raw_product,
                                attributes=attributes,
                                human_instructions=human_instructions,
                            )
                            if product is not None:
                                asin = _text(product["asin"], field="WebShop product ASIN")
                                if asin not in asins:
                                    asins.add(asin)
                                    price = _product_price(product, rng)
                                    product_json = json.dumps(
                                        product,
                                        ensure_ascii=True,
                                        separators=(",", ":"),
                                    )
                                    insert_rows.append(
                                        (
                                            accepted,
                                            asin,
                                            product_json,
                                            price,
                                            _text(
                                                product["category"],
                                                field="WebShop category",
                                                allow_empty=True,
                                            ),
                                            _text(
                                                product["query"],
                                                field="WebShop query",
                                                allow_empty=True,
                                            ),
                                        )
                                    )
                                    instructions = product.get("instructions")
                                    if instructions is not None:
                                        goal_rows.append(
                                            (
                                                accepted,
                                                asin,
                                                _text(
                                                    product["category"],
                                                    field="WebShop category",
                                                    allow_empty=True,
                                                ),
                                                _text(
                                                    product["query"],
                                                    field="WebShop query",
                                                    allow_empty=True,
                                                ),
                                                _text(
                                                    product["name"],
                                                    field="WebShop name",
                                                    allow_empty=True,
                                                ),
                                                _text(
                                                    product["product_category"],
                                                    field="WebShop product_category",
                                                    allow_empty=True,
                                                ),
                                                json.dumps(
                                                    instructions,
                                                    ensure_ascii=True,
                                                    separators=(",", ":"),
                                                ),
                                                price,
                                            )
                                        )
                                    document = _search_document(product)
                                    documents.write(
                                        (json.dumps(document, ensure_ascii=True) + "\n").encode(
                                            "utf-8"
                                        )
                                    )
                                    accepted += 1
                            if len(insert_rows) >= 1_000:
                                connection.executemany(
                                    """
                                    INSERT INTO products(
                                        source_id, asin, product_json, price, category, query
                                    ) VALUES (?, ?, ?, ?, ?, ?)
                                    """,
                                    insert_rows,
                                )
                                insert_rows.clear()
                            if len(goal_rows) >= 1_000:
                                connection.executemany(
                                    """
                                    INSERT INTO goal_products(
                                        source_id, asin, category, query, name,
                                        product_category, instructions_json, price
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    goal_rows,
                                )
                                goal_rows.clear()
                            if raw_seen % checkpoint_interval == 0:
                                if insert_rows:
                                    connection.executemany(
                                        """
                                        INSERT INTO products(
                                            source_id, asin, product_json, price, category, query
                                        ) VALUES (?, ?, ?, ?, ?, ?)
                                        """,
                                        insert_rows,
                                    )
                                    insert_rows.clear()
                                if goal_rows:
                                    connection.executemany(
                                        """
                                        INSERT INTO goal_products(
                                            source_id, asin, category, query, name,
                                            product_category, instructions_json, price
                                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                        """,
                                        goal_rows,
                                    )
                                    goal_rows.clear()
                                _checkpoint(
                                    connection=connection,
                                    documents=cast(BinaryIO, documents),
                                    raw_seen=raw_seen,
                                    accepted=accepted,
                                    rng=rng,
                                )
                            if raw_seen % progress_interval == 0:
                                remaining: float | None
                                if raw_product_limit is not None:
                                    remaining = float(raw_product_limit - raw_seen)
                                elif counting.bytes_read > 0:
                                    estimated_total = (
                                        raw_seen * products_source.size_bytes / counting.bytes_read
                                    )
                                    remaining = max(0.0, estimated_total - raw_seen)
                                else:
                                    remaining = None
                                _progress(
                                    progress_callback,
                                    phase="products",
                                    raw_seen=raw_seen,
                                    accepted=accepted,
                                    goals=0,
                                    phase_started=phase_started,
                                    remaining_units=remaining,
                                )
                    except (ijson.JSONError, SystemError) as error:
                        raise ValueError("WebShop product JSON is invalid") from error
                if insert_rows:
                    connection.executemany(
                        """
                        INSERT INTO products(
                            source_id, asin, product_json, price, category, query
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        insert_rows,
                    )
                if goal_rows:
                    connection.executemany(
                        """
                        INSERT INTO goal_products(
                            source_id, asin, category, query, name,
                            product_category, instructions_json, price
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        goal_rows,
                    )
                _checkpoint(
                    connection=connection,
                    documents=cast(BinaryIO, documents),
                    raw_seen=raw_seen,
                    accepted=accepted,
                    rng=rng,
                )
            if raw_product_limit is not None and raw_seen != raw_product_limit:
                raise ValueError("WebShop product source ended before the requested limit")
            if accepted == 0:
                raise ValueError("WebShop product preparation accepted no products")
            _set_meta(connection, {"phase": "indexes"})
            connection.commit()
            metadata = _meta(connection)

        if metadata["phase"] == "indexes":
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS products_asin_uq ON products(asin)"
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS products_category_idx
                    ON products(category, source_id)
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS products_query_idx
                    ON products(query, source_id)
                    """
                )
                _set_meta(connection, {"phase": "goals"})
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            metadata = _meta(connection)

        if metadata["phase"] == "goals":
            rng = random.Random()  # noqa: S311 - restored frozen benchmark RNG
            _restore_rng_state(rng, metadata["rng_state"])
            goals = _goal_records(connection, rng)
            goals_temp = output_root.parent / (
                f".{output_root.name}.goals-{os.getpid()}-{time.time_ns()}.part"
            )
            with goals_temp.open("xb") as stream:
                for goal in goals:
                    stream.write((json.dumps(goal, ensure_ascii=True) + "\n").encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            goals_path = staging_root / _GOALS_NAME
            if goals_path.exists():
                if goals_path.stat().st_size != goals_temp.stat().st_size or _sha256_file(
                    goals_path
                ) != _sha256_file(goals_temp):
                    raise ValueError(
                        "existing WebShop goal stream differs from regenerated official goals"
                    )
            else:
                goals_temp.rename(goals_path)
            _set_meta(
                connection,
                {
                    "goal_count": str(len(goals)),
                    "phase": "complete",
                },
            )
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            metadata = _meta(connection)

        if metadata["phase"] != "complete":
            raise ValueError("WebShop preparation did not reach the complete phase")
        raw_seen = int(metadata["raw_seen"])
        accepted = int(metadata["accepted"])
        goal_count = int(metadata["goal_count"])
    finally:
        connection.close()

    store_size = store_path.stat().st_size
    documents_size = documents_path.stat().st_size
    goals_path = staging_root / _GOALS_NAME
    goals_size = goals_path.stat().st_size
    receipt = WebShopBulkPreparationReceipt(
        seed=seed,
        raw_product_limit=raw_product_limit,
        raw_products_seen=raw_seen,
        product_count=accepted,
        goal_count=goal_count,
        product_store_size_bytes=store_size,
        product_store_sha256=_sha256_file(store_path),
        documents_size_bytes=documents_size,
        documents_sha256=_sha256_file(documents_path),
        goals_size_bytes=goals_size,
        goals_sha256=_sha256_file(goals_path),
        source_identities=source_values,
    )
    manifest_path = staging_root / _MANIFEST_NAME
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(
            canonical_json(
                {
                    **receipt.to_value(),
                    "content_hash": receipt.content_hash,
                }
            )
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    staging_root.rename(output_root)
    return receipt


def verify_webshop_bulk_assets(output_root: Path) -> WebShopBulkPreparationReceipt:
    """Verify a published bulk preparation without reading private task text."""

    if not isinstance(output_root, Path) or not output_root.is_absolute():
        raise ValueError("output_root must be an absolute Path")
    if not output_root.is_dir():
        raise NotADirectoryError(output_root)
    receipt = WebShopBulkPreparationReceipt.from_manifest_value(
        _read_published_canonical_record(
            output_root / _MANIFEST_NAME,
            label="WebShop bulk preparation manifest",
        )
    )
    paths = (
        (
            output_root / _STORE_NAME,
            receipt.product_store_size_bytes,
            receipt.product_store_sha256,
        ),
        (
            output_root / _DOCUMENTS_NAME,
            receipt.documents_size_bytes,
            receipt.documents_sha256,
        ),
        (
            output_root / _GOALS_NAME,
            receipt.goals_size_bytes,
            receipt.goals_sha256,
        ),
    )
    for path, size, digest in paths:
        if not path.is_file() or path.stat().st_size != size or _sha256_file(path) != digest:
            raise ValueError(f"prepared WebShop asset differs from manifest: {path.name}")
    connection = sqlite3.connect(output_root / _STORE_NAME)
    try:
        product_count = cast(
            int,
            connection.execute("SELECT COUNT(*) FROM products").fetchone()[0],
        )
        goal_product_count = cast(
            int,
            connection.execute("SELECT COUNT(*) FROM goal_products").fetchone()[0],
        )
        integrity = cast(str, connection.execute("PRAGMA quick_check").fetchone()[0])
    finally:
        connection.close()
    if product_count != receipt.product_count or goal_product_count <= 0:
        raise ValueError("prepared WebShop store counts differ from manifest")
    if integrity != "ok":
        raise ValueError("prepared WebShop product store failed quick_check")
    with (output_root / _GOALS_NAME).open("rb") as stream:
        line_count = sum(1 for line in stream if line)
    if line_count != receipt.goal_count:
        raise ValueError("prepared WebShop goal count differs from manifest")
    return receipt


def _print_progress(progress: WebShopPreparationProgress) -> None:
    eta = "unknown" if progress.eta_seconds is None else f"{progress.eta_seconds:.1f}"
    print(
        " ".join(
            (
                f"phase={progress.phase}",
                f"raw={progress.raw_products_seen}",
                f"accepted={progress.accepted_products}",
                f"goals={progress.goals_written}",
                f"elapsed_s={progress.elapsed_seconds:.1f}",
                f"records_s={progress.records_per_second:.3f}",
                f"eta_s={eta}",
            )
        ),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--target-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--raw-product-limit", type=int)
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--progress-interval", type=int, default=10_000)
    args = parser.parse_args(argv)
    lock = load_benchmark_acquisition_lock(args.lock.resolve())
    products, attributes, human = locked_webshop_sources(
        lock,
        args.target_root.resolve(),
    )
    receipt = prepare_webshop_bulk_assets(
        products_source=products,
        attributes_source=attributes,
        human_instructions_source=human,
        output_root=args.output_root.resolve(),
        seed=FIXED_SEED,
        raw_product_limit=args.raw_product_limit,
        checkpoint_interval=args.checkpoint_interval,
        progress_interval=args.progress_interval,
        progress_callback=_print_progress,
    )
    print(
        canonical_json(
            {
                "content_hash": receipt.content_hash,
                "goal_count": receipt.goal_count,
                "product_count": receipt.product_count,
                "raw_products_seen": receipt.raw_products_seen,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "WEBSHOP_BULK_PREPARATION_FORMAT",
    "WEBSHOP_PRODUCT_STORE_FORMAT",
    "WebShopBulkPreparationReceipt",
    "WebShopPreparationProgress",
    "WebShopSourceIdentity",
    "locked_webshop_sources",
    "prepare_webshop_bulk_assets",
    "verify_webshop_bulk_assets",
]
