from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from pathlib import Path
from typing import cast

import pytest
from skillev_private.benchmarks.acquisition import load_benchmark_acquisition_lock
from skillev_private.benchmarks.webshop_bulk_preparation import (
    WebShopPreparationProgress,
    WebShopSourceIdentity,
    locked_webshop_sources,
    prepare_webshop_bulk_assets,
    verify_webshop_bulk_assets,
)
from skillev_private.benchmarks.webshop_disk_runtime import (
    close_disk_backed_server,
    create_disk_backed_server,
)

from skillev.experiments import FIXED_SEED


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> WebShopSourceIdentity:
    path.write_text(json.dumps(value), encoding="utf-8")
    return WebShopSourceIdentity(
        path=path.resolve(),
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
    )


def _product(
    asin: str,
    *,
    pricing: str | None,
    name: str,
    product_category: str = "fixture > category",
) -> dict[str, object]:
    return {
        "aplus_present": False,
        "asin": asin,
        "availability_quantity": 1,
        "availability_status": "available",
        "brand": "fixture brand",
        "brand_url": "fixture",
        "category": "fixture-category",
        "customization_options": {
            "Color": [
                {
                    "image": None,
                    "value": "Red/Blue",
                }
            ]
        },
        "fast_track_message": "",
        "fulfilled_by_amazon": False,
        "full_description": f"{name} full description",
        "images": [f"https://invalid.example/{asin}.jpg"],
        "list_price": None,
        "name": name,
        "pricing": pricing,
        "product_category": product_category,
        "product_information": {"ASIN": asin},
        "query": "  Fixture Query  ",
        "seller_id": "seller",
        "seller_name": "seller",
        "small_description": [f"{name} bullet"],
        "small_description_old": "",
        "total_answered_questions": 0,
        "total_reviews": 0,
    }


def _sources(
    tmp_path: Path,
) -> tuple[WebShopSourceIdentity, WebShopSourceIdentity, WebShopSourceIdentity]:
    products = [
        _product(
            "ASIN000001",
            pricing="$10.00 - $20.00",
            name="First product",
            product_category="",
        ),
        _product("ASIN000001", pricing="$30.00", name="Duplicate product"),
        _product("nan", pricing="$40.00", name="Rejected product"),
        _product("ASIN000002", pricing=None, name="Second product"),
    ]
    attributes = {
        "ASIN000001": {
            "attributes": ["lightweight", "fixture attribute"],
            "instruction": "synthetic source field",
            "instruction_attributes": ["unused"],
        }
    }
    human = {
        "ASIN000001": [
            {
                "asin": "ASIN000001",
                "attributes": ["unused source alias"],
                "instruction": "Find the first fixture.",
                "instruction_attributes": ["lightweight"],
                "instruction_options": ["red | blue"],
                "options": ["unused"],
            }
        ],
        "ASIN000002": [
            {
                "asin": "ASIN000002",
                "attributes": [],
                "instruction": "This goal is skipped.",
                "instruction_attributes": [],
                "instruction_options": [],
                "options": [],
            }
        ],
    }
    return (
        _write_json(tmp_path / "items_shuffle.json", products),
        _write_json(tmp_path / "items_ins_v2.json", attributes),
        _write_json(tmp_path / "items_human_ins.json", human),
    )


def test_streaming_preparation_matches_official_transform_and_rng(tmp_path: Path) -> None:
    products, attributes, human = _sources(tmp_path)
    output = (tmp_path / "prepared").resolve()
    samples: list[WebShopPreparationProgress] = []

    receipt = prepare_webshop_bulk_assets(
        products_source=products,
        attributes_source=attributes,
        human_instructions_source=human,
        output_root=output,
        raw_product_limit=4,
        checkpoint_interval=2,
        progress_interval=1,
        progress_callback=samples.append,
    )

    assert receipt.raw_products_seen == 4
    assert receipt.product_count == 2
    assert receipt.goal_count == 1
    assert samples[-1].raw_products_seen == 4
    assert all(sample.records_per_second >= 0.0 for sample in samples)
    assert verify_webshop_bulk_assets(output) == receipt

    connection = sqlite3.connect(output / "products.sqlite3")
    try:
        rows = connection.execute(
            "SELECT source_id, asin, product_json, price FROM products ORDER BY source_id"
        ).fetchall()
    finally:
        connection.close()
    assert [(row[0], row[1]) for row in rows] == [
        (0, "ASIN000001"),
        (1, "ASIN000002"),
    ]
    first = cast(dict[str, object], json.loads(rows[0][2]))
    assert "product_information" not in first
    assert "brand" not in first
    assert first["query"] == "fixture query"
    assert first["BulletPoints"] == ["First product bullet"]
    assert first["options"] == {"color": ["red | blue"]}
    assert first["Attributes"] == ["lightweight", "fixture attribute"]
    assert first["MainImage"] == "https://invalid.example/ASIN000001.jpg"
    assert first["product_category"] == ""

    rng = random.Random(FIXED_SEED)  # noqa: S311 - reference benchmark RNG
    expected_price = rng.uniform(10.0, 20.0)
    assert rows[0][3] == expected_price
    assert rows[1][3] == 100.0
    price_range = [price for price in (10.0 * i for i in range(1, 100)) if price > expected_price][
        :4
    ]
    _, expected_upper = sorted(rng.sample(price_range, 2))
    goals = [
        json.loads(line)
        for line in (output / "goals.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert goals[0]["price_upper"] == expected_upper
    assert goals[0]["instruction_text"].endswith(f"price lower than {expected_upper:.2f} dollars")

    documents = [
        json.loads(line)
        for line in (output / "documents.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [document["id"] for document in documents] == ["ASIN000001", "ASIN000002"]
    assert set(documents[0]) == {"contents", "id"}
    assert "first product full description" in documents[0]["contents"]
    assert "color: red | blue" in documents[0]["contents"]


def test_interrupted_ingestion_resumes_and_preserves_uncommitted_bytes(
    tmp_path: Path,
) -> None:
    products, attributes, human = _sources(tmp_path)
    output = (tmp_path / "prepared").resolve()

    def stop_after_checkpoint(progress: WebShopPreparationProgress) -> None:
        if progress.raw_products_seen == 2:
            raise RuntimeError("simulated WSL interruption")

    with pytest.raises(RuntimeError, match="simulated WSL interruption"):
        prepare_webshop_bulk_assets(
            products_source=products,
            attributes_source=attributes,
            human_instructions_source=human,
            output_root=output,
            raw_product_limit=4,
            checkpoint_interval=1,
            progress_interval=1,
            progress_callback=stop_after_checkpoint,
        )

    staging = output.with_name(".prepared.building")
    assert staging.is_dir()
    with (staging / "documents.jsonl").open("ab") as stream:
        stream.write(b"uncommitted bytes remain preserved\n")

    receipt = prepare_webshop_bulk_assets(
        products_source=products,
        attributes_source=attributes,
        human_instructions_source=human,
        output_root=output,
        raw_product_limit=4,
        checkpoint_interval=1,
        progress_interval=1,
    )

    assert receipt.product_count == 2
    preserved = tuple(tmp_path.glob(".prepared.documents-uncommitted-*.preserved"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes().endswith(b"uncommitted bytes remain preserved\n")
    assert verify_webshop_bulk_assets(output) == receipt


def test_locked_source_resolution_uses_exact_committed_webshop_files(tmp_path: Path) -> None:
    lock = load_benchmark_acquisition_lock(
        (Path(__file__).parents[2] / "benchmark-acquisition-lock.json").resolve()
    )
    products, attributes, human = locked_webshop_sources(lock, tmp_path.resolve())

    assert products.path == tmp_path.resolve() / "webshop/raw/items_shuffle.json"
    assert attributes.path == tmp_path.resolve() / "webshop/raw/items_ins_v2.json"
    assert human.path == tmp_path.resolve() / "webshop/raw/items_human_ins.json"
    assert products.size_bytes == 5_479_720_229
    assert products.sha256 == "2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8"


def test_source_digest_mismatch_fails_before_staging(tmp_path: Path) -> None:
    products, attributes, human = _sources(tmp_path)
    wrong = WebShopSourceIdentity(
        path=products.path,
        size_bytes=products.size_bytes,
        sha256="0" * 64,
    )
    output = (tmp_path / "prepared").resolve()

    with pytest.raises(ValueError, match="digest differs"):
        prepare_webshop_bulk_assets(
            products_source=wrong,
            attributes_source=attributes,
            human_instructions_source=human,
            output_root=output,
            raw_product_limit=4,
        )

    assert not output.exists()
    assert not output.with_name(".prepared.building").exists()


def test_disk_runtime_exposes_official_collection_surfaces(tmp_path: Path) -> None:
    products, attributes, human = _sources(tmp_path)
    output = (tmp_path / "prepared").resolve()
    prepare_webshop_bulk_assets(
        products_source=products,
        attributes_source=attributes,
        human_instructions_source=human,
        output_root=output,
        raw_product_limit=4,
        checkpoint_interval=2,
        progress_interval=1,
    )

    class _SimServer:
        pass

    class _OfficialModule:
        SimServer = _SimServer

    search_index = (tmp_path / "index").resolve()
    search_index.mkdir()
    searcher = object()
    server = create_disk_backed_server(
        web_agent_module=_OfficialModule,
        product_store_path=output / "products.sqlite3",
        goals_path=output / "goals.jsonl",
        search_index_path=search_index,
        searcher_factory=lambda path: searcher if path == search_index else None,
    )
    try:
        assert len(server.all_products) == 2
        assert server.all_products[0]["asin"] == "ASIN000001"
        assert server.all_products[-1]["asin"] == "ASIN000002"
        assert [product["asin"] for product in server.all_products] == [
            "ASIN000001",
            "ASIN000002",
        ]
        assert server.product_item_dict["ASIN000002"]["Title"] == "Second product"
        assert server.product_prices["ASIN000002"] == 100.0
        assert server.search_engine is searcher
        assert len(server.goals) == 1
        assert server.cum_weights == [0.0, 1.0]
    finally:
        close_disk_backed_server(server)
