"""Acquire current benchmark assets privately; never generate, score or train.

Use the dedicated CPU environment documented in docs/current-datasets.md.
Existing completed files are retained. A failed transfer stays .partial and
does not become a ready dataset. Hugging Face credentials use the normal local
credential store, are never logged, and are sent only to the gated HF source.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.metadata
import json
import os
import shutil
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def fetch(url: str, path: Path) -> None:
    if path.is_file():
        return
    if not url.startswith("https://"):
        raise ValueError("dataset downloads require HTTPS")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    request = urllib.request.Request(url)  # noqa: S310 -- HTTPS checked above
    started = last = time.monotonic()
    size = 0
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 -- HTTPS above
        total = int(response.headers.get("Content-Length", 0))
        with path.with_name(path.name + ".partial").open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
                size += len(block)
                now = time.monotonic()
                if now - last >= 30:
                    rate = size / (now - started)
                    print(
                        json.dumps(
                            {
                                "file": path.name,
                                "bytes": size,
                                "bytes_per_second": rate,
                                "eta_seconds": (total - size) / rate if total else None,
                            }
                        ),
                        flush=True,
                    )
                    last = now
        if total and size != total:
            raise OSError("download did not receive the declared number of bytes")
    path.with_name(path.name + ".partial").rename(path)


def count_rows(path: Path) -> int:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return sum(1 for _ in csv.DictReader(stream))
    with path.open() as stream:
        prefix = stream.read(128).lstrip()
        stream.seek(0)
        if prefix.startswith("["):
            return len(json.load(stream))
        count = 0
        for line in stream:
            if line.strip():
                json.loads(line)
                count += 1
        return count


def prepare_gpqa(path: Path) -> dict[str, Any]:
    """Metadata-only subject selection, no grading/answer-dependent selection."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    biology = sorted(row["Record ID"] for row in rows if row["High-level domain"] == "Biology")
    organic = sorted(
        row["Record ID"]
        for row in rows
        if row["High-level domain"] == "Chemistry" and row["Subdomain"] == "Organic Chemistry"
    )
    if (len(rows), len({row["Record ID"] for row in rows}), len(biology), len(organic)) != (
        198,
        198,
        19,
        72,
    ):
        raise ValueError("GPQA Diamond source does not contain the approved 19+72 population")
    if set(biology) & set(organic):
        raise ValueError("GPQA subject populations overlap")
    allowlist = path.parent / "bioorganic-allowlist-private.json"
    value = {"biology_record_ids": biology, "organic_record_ids": organic}
    allowlist.write_text(json.dumps(value, indent=2) + "\n")
    return {
        "population_rows": 91,
        "strata": {"Biology": 19, "Organic Chemistry": 72},
        "allowlist_path": str(allowlist.resolve()),
    }


def acquire(name: str, spec: dict[str, Any], root: Path) -> dict[str, Any]:
    directory = root / name
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    result: dict[str, Any] = {"benchmark": name, "status": "incomplete", "split": spec["split"]}
    paths: list[Path] = []
    if spec["kind"] == "huggingface-files":
        from huggingface_hub import hf_hub_download

        meta = directory / "upstream-metadata.json"
        fetch("https://huggingface.co/api/datasets/" + spec["repository"], meta)
        info = json.loads(meta.read_text())
        revision = info["sha"]  # Upstream revision identity, not a computed content hash.
        for file in spec["files"]:
            path = directory / file
            if spec.get("auth_required"):
                # Keep credential and redirect handling in the official HF client.
                if not path.is_file():
                    cached = hf_hub_download(
                        spec["repository"], file, repo_type="dataset", revision=revision
                    )
                    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    partial = path.with_name(path.name + ".partial")
                    shutil.copyfile(cached, partial)
                    partial.rename(path)
            else:
                fetch(
                    f"https://huggingface.co/datasets/{spec['repository']}/resolve/{revision}/{file}",
                    path,
                )
            paths.append(path)
        result.update(repository=spec["repository"], source_revision=revision)
    elif spec["kind"] == "http-files":
        for item in spec["files"]:
            path = directory / item["name"]
            fetch(item["url"], path)
            if path.suffix == ".gz":
                expanded = path.with_suffix("")
                if not expanded.exists():
                    with gzip.open(path, "rb") as source, expanded.open("xb") as output:
                        while block := source.read(1024 * 1024):
                            output.write(block)
                path = expanded
            paths.append(path)
        result["source_urls"] = [item["url"] for item in spec["files"]]
    elif spec["kind"] == "google-drive":
        import gdown

        path = directory / spec["file"]
        if not path.exists():
            partial = path.with_name(path.name + ".partial")
            if not gdown.download(id=spec["file_id"], output=str(partial), quiet=False):
                raise OSError("official Google Drive dataset download failed")
            partial.rename(path)
        with zipfile.ZipFile(path) as archive:
            members = [
                n
                for n in archive.namelist()
                if n.endswith("/" + spec["split"]) or n == spec["split"]
            ]
            if len(members) != 1:
                raise ValueError("MuSiQue-Ans dev file is missing or ambiguous")
            target = directory / spec["split"]
            if not target.exists():
                with archive.open(members[0]) as source, target.open("xb") as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
            paths.append(target)
    elif spec["kind"] == "installed-environment":
        package, expected = spec["package"].split("==")
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise ValueError("installed environment version differs from the source catalog")
        result["package"] = f"{package}=={actual}"
    else:
        raise ValueError("unsupported dataset source")
    if name == "alfworld":
        # Native text-game assets only. Do not install the unrelated pretrained agent.
        for path in paths:
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    destination = (directory / member.filename).resolve()
                    if not destination.is_relative_to(directory.resolve()):
                        raise ValueError("environment archive member escapes the data directory")
                    if not destination.exists():
                        archive.extract(member, directory)
        result["data_directory"] = str(directory.resolve())
        result["package"] = "alfworld==" + importlib.metadata.version("alfworld")
    elif paths:
        result["rows"] = sum(count_rows(path) for path in paths)
        if spec.get("expected_rows") is not None and result["rows"] != spec["expected_rows"]:
            raise ValueError(f"{name} downloaded row count differs from the declared release")
    if name == "gpqa-diamond-bioorganic":
        result.update(prepare_gpqa(paths[0]))
    if name == "math-hard":
        rows = [row for path in paths for row in json.loads(path.read_text())]
        if any(row["level"] != "Level 5" for row in rows):
            raise ValueError("MATH-Hard has an unexpected difficulty level")
    if name == "apps-introductory":
        with paths[0].open() as stream:
            result["population_rows"] = sum(
                json.loads(line)["difficulty"] == "introductory" for line in stream if line.strip()
            )
        if result["population_rows"] != spec["population_rows"]:
            raise ValueError("APPS Introductory population differs from the expected release")
    result.update(
        status="ready",
        paths=[str(path.resolve()) for path in paths],
        bytes=sum(path.stat().st_size for path in paths),
        generation_started=False,
    )
    (directory / "inventory-private.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog", type=Path, default=Path("configs/evaluation/current_datasets.json")
    )
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    os.umask(0o077)
    catalog = json.loads(args.catalog.read_text())
    names = args.only or catalog["iid"] + catalog["ood"]
    if any(name not in catalog["iid"] + catalog["ood"] for name in names):
        parser.error("requested dataset is not in the current catalog")
    args.destination.mkdir(mode=0o700, parents=True, exist_ok=True)

    def job(name: str) -> dict[str, Any]:
        try:
            result = acquire(name, catalog["datasets"][name], args.destination)
        except Exception as error:
            result = {"benchmark": name, "status": "incomplete", "error": str(error)}
        print(json.dumps(result), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(job, names))
    (args.destination / "inventory-private.json").write_text(json.dumps(results, indent=2) + "\n")
    if any(result["status"] != "ready" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
