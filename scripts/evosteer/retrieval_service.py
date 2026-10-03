from __future__ import annotations

import gzip
import json
import mmap
import queue
import sys
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

DIM = 768
HEADER = 45


def _index_meta(root: Path) -> tuple[int, int]:
    head = (root / "part_aa").open("rb").read(HEADER)
    assert head[:4] == b"IxFI", head[:4]
    dim = int.from_bytes(head[4:8], "little")
    total = int.from_bytes(head[8:16], "little")
    floats = int.from_bytes(head[37:45], "little")
    assert dim == DIM and floats == total * dim and head[33:37] == b"\0\0\0\0"
    return dim, total


def prepare_corpus(root: Path) -> np.ndarray:
    root = Path(root)
    corpus = root / "wiki-18.jsonl"
    if not corpus.exists():
        with gzip.open(root / "wiki-18.jsonl.gz", "rb") as raw, tarfile.open(fileobj=raw, mode="r|") as tar:
            for member in tar:
                if member.isfile() and member.name.endswith(".jsonl"):
                    source = tar.extractfile(member)
                    with (root / "wiki-18.jsonl.partial").open("wb") as out:
                        while block := source.read(64 << 20):
                            out.write(block)
                    break
        (root / "wiki-18.jsonl.partial").rename(corpus)
        print("corpus extracted", corpus.stat().st_size, flush=True)
    offsets_path = root / "wiki-18.offsets.npy"
    if not offsets_path.exists():
        with corpus.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
            starts = [0]
            position = data.find(b"\n")
            while position != -1:
                starts.append(position + 1)
                position = data.find(b"\n", position + 1)
            if starts[-1] != len(data):
                starts.append(len(data))
        offsets = np.asarray(starts, dtype=np.uint64)
        np.save(offsets_path, offsets)
        print("offsets", len(offsets) - 1, flush=True)
    return np.load(offsets_path)


def prepare(root: Path) -> None:
    root = Path(root)
    offsets = prepare_corpus(root)
    lines = len(offsets) - 1
    _dim, total = _index_meta(root)
    assert lines == total, f"corpus lines {lines} != vectors {total}"
    target = root / "vectors_fp16.bin"
    if target.exists() and target.stat().st_size == total * DIM * 2:
        print("vectors ready", flush=True)
        return
    parts = [root / "part_aa", root / "part_ab"]
    sizes = [p.stat().st_size for p in parts]
    assert sum(sizes) == HEADER + total * DIM * 4
    out = np.lib.format.open_memmap(root / "vectors_fp16.partial.npy", mode="w+", dtype=np.float16, shape=(total, DIM))
    rows_per_block = 1 << 16
    block_bytes = rows_per_block * DIM * 4
    handles = [p.open("rb") for p in parts]
    handles[0].seek(HEADER)
    current, row = 0, 0
    pending = b""
    started = time.time()
    while row < total:
        need = min(block_bytes, (total - row) * DIM * 4) - len(pending)
        chunk = handles[current].read(need)
        if not chunk and current + 1 >= len(handles) and len(pending) < DIM * 4:
            raise RuntimeError(f"index ended at row {row} of {total}")
        pending += chunk
        if len(chunk) < need and current + 1 < len(handles):
            current += 1
            continue
        count = len(pending) // (DIM * 4)
        block = np.frombuffer(pending[: count * DIM * 4], dtype=np.float32).reshape(count, DIM)
        out[row : row + count] = block.astype(np.float16)
        pending = pending[count * DIM * 4 :]
        row += count
        if (row // rows_per_block) % 32 == 0:
            print(f"vectors {row}/{total} {time.time() - started:.0f}s", flush=True)
    out.flush()
    del out
    (root / "vectors_fp16.partial.npy").rename(root / "vectors_fp16.npy")
    target.write_text("see vectors_fp16.npy\n")
    print("vectors converted", flush=True)


def _exact_top(vectors: np.ndarray, candidates: np.ndarray, query: np.ndarray, k: int) -> tuple[list[float], list[int]]:
    order = np.sort(np.unique(candidates))
    exact = np.asarray(vectors[order], dtype=np.float32) @ query
    keep = np.argsort(-exact, kind="stable")[:k]
    return exact[keep].tolist(), order[keep].tolist()


class Retriever:
    def __init__(self, root: Path, model_dir: Path, device: str = "cuda:0") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        root = Path(root)
        vectors = np.load(root / "vectors_fp16.npy", mmap_mode="r")
        self.vectors = vectors
        self.total = vectors.shape[0]
        self.cpu = device == "cpu"
        self.dtype = torch.bfloat16 if self.cpu else torch.float16
        self.matrix = torch.empty((self.total, DIM), dtype=self.dtype, device=device)
        step = 1 << 20
        for start in range(0, self.total, step):
            block = torch.from_numpy(np.array(vectors[start : start + step]))
            self.matrix[start : start + step] = block.to(device=device, dtype=self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir, dtype=torch.float32 if self.cpu else torch.float16).to(device).eval()
        self.offsets = np.load(root / "wiki-18.offsets.npy")
        self._corpus_file = (root / "wiki-18.jsonl").open("rb")
        self.corpus = mmap.mmap(self._corpus_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.timing: dict[str, float] = {}

    BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)
    LENGTH_BUCKETS = (64, 128, 192, 256)

    def encode(self, texts: list[str]):
        torch = self.torch
        rows = next(b for b in self.BATCH_BUCKETS if b >= len(texts))
        padded = [f"query: {t}" for t in texts] + ["query: "] * (rows - len(texts))
        batch = self.tokenizer(padded, max_length=256, padding=True, truncation=True, return_tensors="pt")
        length = next(b for b in self.LENGTH_BUCKETS if b >= batch["input_ids"].shape[1])
        batch = self.tokenizer(padded, max_length=length, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            return torch.nn.functional.normalize(pooled.float(), dim=-1)

    def warm_up(self) -> None:
        if self.cpu:
            self.search(["warm up"], 3)
            return
        for rows in self.BATCH_BUCKETS:
            for length in self.LENGTH_BUCKETS:
                self.search(["warm up " + "x " * (length - 12)] * rows, 3)

    def search(self, texts: list[str], topk: int) -> list[list[dict]]:
        limit = self.BATCH_BUCKETS[-1]
        if len(texts) > limit:
            return [hits for start in range(0, len(texts), limit) for hits in self.search(texts[start : start + limit], topk)]
        torch = self.torch
        began = time.monotonic()
        final_k = topk
        if self.cpu:
            topk = max(50, 10 * topk)
        exact_queries = self.encode(texts)
        queries = exact_queries.to(self.dtype)
        if not self.cpu:
            torch.cuda.synchronize(self.device)
        encoded = time.monotonic()
        best_scores = best_ids = None
        with torch.inference_mode():
            for start in range(0, self.total, 1 << 22):
                scores = queries @ self.matrix[start : start + (1 << 22)].T
                value, index = torch.topk(scores.float(), k=min(topk, scores.shape[1]), dim=1)
                index = index + start
                if best_scores is None:
                    best_scores, best_ids = value, index
                else:
                    merged_scores = torch.cat([best_scores, value], 1)
                    merged_ids = torch.cat([best_ids, index], 1)
                    best_scores, order = torch.topk(merged_scores, k=topk, dim=1)
                    best_ids = torch.gather(merged_ids, 1, order)
        if self.cpu:
            exact_q = exact_queries[: len(texts)].numpy()
            ids = best_ids[: len(texts)].numpy()
            rescored_scores, rescored_ids = [], []
            for row, candidates in zip(exact_q, ids):
                scores, kept = _exact_top(self.vectors, candidates, row, final_k)
                rescored_scores.append(scores)
                rescored_ids.append(kept)
            best_scores, best_ids = rescored_scores, rescored_ids
        else:
            best_scores, best_ids = best_scores[: len(texts)].tolist(), best_ids[: len(texts)].tolist()
        searched = time.monotonic()
        results = []
        for row_scores, row_ids in zip(best_scores, best_ids):
            hits = []
            for score, pid in zip(row_scores, row_ids):
                line = self.corpus[int(self.offsets[pid]) : int(self.offsets[pid + 1])]
                record = json.loads(line)
                title, _, text = record["contents"].partition("\n")
                hits.append({"id": record["id"], "title": title.strip('"'), "text": text, "score": round(score, 5)})
            results.append(hits)
        fetched = time.monotonic()
        for key, value in (("encode_s", encoded - began), ("search_s", searched - encoded), ("fetch_s", fetched - searched)):
            self.timing[key] = self.timing.get(key, 0.0) + value
        return results


class Batcher:
    def __init__(self, retriever: Retriever, window_ms: float = 15.0, max_batch: int = 128) -> None:
        self.retriever, self.window, self.max_batch = retriever, window_ms / 1000, max_batch
        self.inbox: queue.Queue = queue.Queue()
        self.stats = {"requests": 0, "queries": 0, "batches": 0, "busy_seconds": 0.0, "started": time.time()}
        self.ready = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()
        self.ready.wait()

    def submit(self, queries: list[str], topk: int) -> list[list[dict]]:
        done = threading.Event()
        box: dict = {}
        self.inbox.put((queries, topk, done, box))
        done.wait()
        if "error" in box:
            raise RuntimeError(box["error"])
        return box["result"]

    def _loop(self) -> None:
        self.retriever.warm_up()
        self.retriever.timing = {}
        self.stats["started"] = time.time()
        self.ready.set()
        while True:
            items = [self.inbox.get()]
            deadline = time.monotonic() + self.window
            while sum(len(i[0]) for i in items) < self.max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    items.append(self.inbox.get(timeout=remaining))
                except queue.Empty:
                    break
            began = time.monotonic()
            try:
                topk = max(i[1] for i in items)
                flat = [q for i in items for q in i[0]]
                found = self.retriever.search(flat, topk)
                position = 0
                for queries, k, done, box in items:
                    box["result"] = [hits[:k] for hits in found[position : position + len(queries)]]
                    position += len(queries)
                    done.set()
            except Exception as exc:
                for *_rest, done, box in items:
                    box["error"] = repr(exc)
                    done.set()
            self.stats["busy_seconds"] += time.monotonic() - began
            self.stats["batches"] += 1
            self.stats["requests"] += len(items)
            self.stats["queries"] += sum(len(i[0]) for i in items)


def make_server(retriever: Retriever, batcher: Batcher, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, value: dict) -> None:
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.startswith("/health"):
                uptime = time.time() - batcher.stats["started"]
                self._send(200, {"ok": True, "passages": retriever.total, **batcher.stats,
                                 "timing": {k: round(v, 3) for k, v in retriever.timing.items()},
                                 "busy_fraction": round(batcher.stats["busy_seconds"] / max(1.0, uptime), 4)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if not self.path.startswith("/retrieve"):
                self._send(404, {"error": "not found"})
                return
            try:
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                queries = [str(q)[:2000] for q in request["queries"]]
                topk = max(1, min(20, int(request.get("topk", 3))))
                self._send(200, {"result": batcher.submit(queries, topk)})
            except Exception as exc:
                self._send(500, {"error": repr(exc)})

        def log_message(self, *args) -> None:
            return

    class Server(ThreadingHTTPServer):
        request_queue_size = 512
        daemon_threads = True

    server = Server((host, port), Handler)
    server.daemon_threads = True
    return server


def serve(root: Path, model_dir: Path, host: str, port: int, device: str = "cuda:0", threads: int = 8) -> None:
    if device == "cpu":
        import torch

        torch.set_num_threads(threads)
    retriever = Retriever(root, model_dir, device=device)
    batcher = Batcher(retriever)
    server = make_server(retriever, batcher, host, port)
    print(json.dumps({"event": "serving", "host": host, "port": port, "passages": retriever.total}), flush=True)
    server.serve_forever()


def embed(root: Path, model_dir: Path, device: str = "cuda:0", batch: int = 512) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    root = Path(root)
    offsets = prepare_corpus(root)
    total = len(offsets) - 1
    target, partial, progress = root / "vectors_fp16.npy", root / "vectors_fp16.partial.npy", root / "vectors_fp16.embed.json"
    if target.exists():
        print("vectors ready", flush=True)
        return
    out = np.lib.format.open_memmap(partial, mode="r+" if partial.exists() else "w+", dtype=np.float16, shape=(total, DIM))
    done = json.loads(progress.read_text())["rows"] if progress.exists() and partial.exists() else 0
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, dtype=torch.float16).to(device).eval()
    corpus_file = (root / "wiki-18.jsonl").open("rb")
    corpus = mmap.mmap(corpus_file.fileno(), 0, access=mmap.ACCESS_READ)

    def texts(start: int, stop: int) -> list[str]:
        rows = []
        for i in range(start, stop):
            line = corpus[int(offsets[i]) : int(offsets[i + 1])]
            rows.append("passage: " + json.loads(line)["contents"])
        return rows

    import concurrent.futures as futures

    def tokenized(start: int):
        stop = min(start + batch, total)
        return stop, tokenizer(texts(start, stop), max_length=256, padding=True, truncation=True, return_tensors="pt")

    started, row = time.time(), done
    with futures.ThreadPoolExecutor(2) as pool, torch.inference_mode():
        pending = pool.submit(tokenized, row) if row < total else None
        while pending is not None:
            stop, encoded = pending.result()
            pending = pool.submit(tokenized, stop) if stop < total else None
            encoded = encoded.to(device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            out[row:stop] = torch.nn.functional.normalize(pooled.float(), dim=-1).half().cpu().numpy()
            row = stop
            if (row // batch) % 2000 == 0 or row == total:
                out.flush()
                progress.write_text(json.dumps({"rows": row}))
                rate = (row - done) / max(1e-9, time.time() - started)
                print(f"embedded {row}/{total} {rate:.0f}/s eta {(total - row) / max(rate, 1e-9) / 60:.0f} min", flush=True)
    out.flush()
    del out
    partial.rename(target)
    (root / "vectors_fp16.bin").write_text("see vectors_fp16.npy (computed by embed)\n")
    print("vectors embedded", flush=True)


def check(root: Path, head_file: Path, rows: int) -> None:
    raw = Path(head_file).read_bytes()
    assert raw[:4] == b"IxFI", "not a faiss flat index prefix"
    have = min(rows, (len(raw) - HEADER) // (DIM * 4))
    official = np.frombuffer(raw[HEADER : HEADER + have * DIM * 4], dtype=np.float32).reshape(have, DIM)
    root = Path(root)
    name = "vectors_fp16.npy" if (root / "vectors_fp16.npy").exists() else "vectors_fp16.partial.npy"
    ours = np.load(root / name, mmap_mode="r")[:have].astype(np.float32)
    cos = (ours * official).sum(1) / (np.linalg.norm(ours, axis=1) * np.linalg.norm(official, axis=1))
    print(json.dumps({
        "rows": have, "cos_min": float(cos.min()), "cos_mean": float(cos.mean()),
        "max_abs_diff": float(np.abs(ours - official).max()),
        "official_norm_mean": float(np.linalg.norm(official, axis=1).mean()),
    }), flush=True)


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "prepare":
        prepare(Path(sys.argv[2]))
    elif command == "embed":
        args = sys.argv[2:]
        device = args[args.index("--device") + 1] if "--device" in args else "cuda:0"
        batch = int(args[args.index("--batch") + 1]) if "--batch" in args else 512
        embed(Path(args[0]), Path(args[1]), device, batch)
    elif command == "check":
        args = sys.argv[2:]
        rows = int(args[args.index("--rows") + 1]) if "--rows" in args else 20000
        check(Path(args[0]), Path(args[1]), rows)
    elif command == "serve":
        args = sys.argv[2:]
        port = int(args[args.index("--port") + 1]) if "--port" in args else 8010
        host = args[args.index("--host") + 1] if "--host" in args else "0.0.0.0"
        device = args[args.index("--device") + 1] if "--device" in args else "cuda:0"
        threads = int(args[args.index("--threads") + 1]) if "--threads" in args else 8
        serve(Path(args[0]), Path(args[1]), host, port, device, threads)
    else:
        raise SystemExit(__doc__)
