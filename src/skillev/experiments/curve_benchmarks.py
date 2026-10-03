from __future__ import annotations

import ast
import builtins
import json
import os
import re
import subprocess
import symtable
import sys
import tempfile
import textwrap
import uuid
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, NamedTuple

from skillev.contracts.evosteer import EvoTask
from skillev.evosteer_application import ResetReceipt, SessionRequest, TaskBinding, TaskSession
from skillev.orchestration.model_executor import FrozenTextExecutor
from skillev.rollout.evosteer_risk import ExecutionRiskPolicy

_FENCE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
_FENCE_LINE = re.compile(r"^\s*(`{3,})([^`]*)$")
_PYTHON_TAGS = frozenset({"", "python", "py", "python3", "py3"})


def _hotpot_norm(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text.lower())
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _f1(prediction: str, answer: str) -> float:
    normalized_prediction, normalized_answer = _hotpot_norm(prediction), _hotpot_norm(answer)
    special = {"yes", "no", "noanswer"}
    if (normalized_prediction in special or normalized_answer in special) and (
        normalized_prediction != normalized_answer
    ):
        return 0.0
    p, g = normalized_prediction.split(), normalized_answer.split()
    if not p or not g:
        return float(p == g)
    overlap = sum((__import__("collections").Counter(p) & __import__("collections").Counter(g)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


_NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\.\d+")


def _last_number(text: str) -> Decimal | None:
    numbers = _NUMBER.findall(text.replace("\\!", "").replace("{,}", ","))
    if not numbers:
        return None
    try:
        return Decimal(numbers[-1].replace(",", ""))
    except InvalidOperation:
        return None


_MATH_CUE = re.compile(r"(?:final answer|answer)[*_\s]*(?:is\b[*_\s]*[:：]?|[:：=])", re.IGNORECASE)


def _math_prediction(output: str) -> Decimal | None:
    for pattern in (r"\\boxed\{([^{}]+)\}", r"####\s*([^\n]+)"):
        spans = re.findall(pattern, output)
        if spans:
            value = _last_number(spans[-1])
            if value is not None:
                return value
    cues = list(_MATH_CUE.finditer(output))
    if cues:
        match = _NUMBER.search(output[cues[-1].end() :][:200].replace("{,}", ","))
        if match:
            try:
                return Decimal(match.group(0).replace(",", ""))
            except InvalidOperation:
                pass
    return _last_number(output)


_ANSWER_CUE = re.compile(r"(?:final answer|answer)\s*(?:is|:)\s*(.+)", re.IGNORECASE)


def _short_answer(output: str) -> str:
    lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    cues = list(_ANSWER_CUE.finditer(output))
    span = cues[-1].group(1) if cues else lines[-1]
    span = re.sub(r"[*_`]+", "", span.splitlines()[0])
    if cues and not span.strip("\"':. \t"):
        after = [
            re.sub(r"[*_`]+", "", line).strip()
            for line in output[cues[-1].end() :].splitlines()
        ]
        span = next((line for line in after if line.strip("\"':. \t")), "")
    return span.strip().strip("\"'").rstrip(".").strip()


def _hotpot_score(output: str, answer: str) -> float:
    short = _short_answer(output)
    gold = _hotpot_norm(answer)
    if gold in {"yes", "no"}:
        return float((_hotpot_norm(short).split() or [""])[0] == gold)
    return max(_f1(short, answer), _f1(output, answer))


QA_KINDS = frozenset({"hotpotqa", "nq_open"})


def _answer_em(output: str, answer: str) -> float:
    short = _hotpot_norm(_short_answer(output))
    gold = _hotpot_norm(answer)
    if gold in {"yes", "no"}:
        return float((short.split() or [""])[0] == gold)
    return float(bool(short) and short == gold)


def _qa_em(output: str, answers: list[str]) -> float:
    return max((_answer_em(output, answer) for answer in answers), default=0.0)


def _qa_answers(kind: str, row: dict[str, Any]) -> list[str]:
    answers = row.get("answers")
    if answers:
        return [answers] if isinstance(answers, str) else list(answers)
    if kind == "hotpotqa":
        return [row["answer"]]
    return list(row["answers"])


def qa_metrics(kind: str, output: str, row: dict[str, Any]) -> dict[str, float]:
    if kind not in QA_KINDS:
        raise ValueError(f"not a short-answer QA benchmark: {kind}")
    answers = _qa_answers(kind, row)
    f1 = max((_hotpot_score(output, answer) for answer in answers), default=0.0)
    return {"em": _qa_em(output, answers), "f1": f1}


_CHOICE_CUES = (
    re.compile(r"(?i:answer)[*_\s]*(?i:is)?[*_\s]*[:：]?[*_\s(\[]*([A-J])(?![A-Za-z])"),
    re.compile(r"(?i:option|choice)[*_\s]*(?i:is)?[*_\s]*[:：]?[*_\s(\[]*([A-J])(?![A-Za-z])"),
)


def _choice_letter(output: str, letters: list[str]) -> str | None:
    for cue in _CHOICE_CUES:
        found = [letter for letter in cue.findall(output) if letter in letters]
        if found:
            return found[-1]
    bare = re.fullmatch(r"\s*[*_(\[]*\s*([A-J])\s*[)\].:*_]*\s*", output)
    if bare and bare.group(1) in letters:
        return bare.group(1)
    return None


def _integer_score(output: str, answer: str) -> float:
    prediction = _math_prediction(output)
    return float(prediction is not None and prediction == Decimal(answer))


def _final_value_score(output: str, answer: str) -> float:
    prediction = _last_boxed(output)
    if prediction is None:
        cues = list(_MATH_CUE.finditer(output))
        tail = output[cues[-1].end() :].strip() if cues else ""
        if not tail:
            return 0.0
        prediction = tail.splitlines()[0].strip().strip("$").rstrip(".").strip()
    return float(_math_equivalent(prediction, str(answer)))


def _last_boxed(text: str) -> str | None:
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    rest = text[start:]
    if rest.startswith("\\boxed "):
        return rest[len("\\boxed ") :].split("$")[0].strip()
    open_at = rest.find("{")
    if open_at < 0:
        return None
    depth = 0
    for index in range(open_at, len(rest)):
        if rest[index] == "{":
            depth += 1
        elif rest[index] == "}":
            depth -= 1
            if depth == 0:
                return rest[open_at + 1 : index].strip()
    return None


def _fix_fracs(text: str) -> str:
    parts = text.split("\\frac")
    out = parts[0]
    for part in parts[1:]:
        out += "\\frac"
        if part[:1] == "{":
            out += part
            continue
        if len(part) < 2:
            return text
        a, b = part[0], part[1]
        if b != "{":
            out += "{" + a + "}{" + b + "}" + part[2:]
        else:
            out += "{" + a + "}" + b + part[2:]
    return out


def _fix_a_slash_b(text: str) -> str:
    if len(text.split("/")) != 2:
        return text
    a, b = text.split("/")
    try:
        a_value, b_value = int(a), int(b)
    except ValueError:
        return text
    if text != f"{a_value}/{b_value}":
        return text
    return "\\frac{" + str(a_value) + "}{" + str(b_value) + "}"


def _remove_right_units(text: str) -> str:
    if "\\text{ " in text:
        return text.split("\\text{ ")[0]
    return text


def _fix_sqrt(text: str) -> str:
    if "\\sqrt" not in text:
        return text
    parts = text.split("\\sqrt")
    out = parts[0]
    for part in parts[1:]:
        if part[:1] != "{" and part:
            out += "\\sqrt{" + part[0] + "}" + part[1:]
        else:
            out += "\\sqrt" + part
    return out


def _strip_math(text: str) -> str:
    text = text.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    text = text.replace("tfrac", "frac").replace("dfrac", "frac")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("^{\\circ}", "").replace("^\\circ", "")
    text = text.replace("\\$", "").replace("$", "")
    text = _remove_right_units(text)
    text = text.replace("\\%", "").replace("%", "")
    text = text.replace(" .", " 0.").replace("{.", "{0.")
    if not text:
        return text
    if text[0] == ".":
        text = "0" + text
    if len(text.split("=")) == 2 and len(text.split("=")[0]) <= 2:
        text = text.split("=")[1]
    text = _fix_sqrt(text)
    text = text.replace(" ", "")
    text = _fix_fracs(text)
    if text == "0.5":
        text = "\\frac{1}{2}"
    return _fix_a_slash_b(text)


_SIMPLE_FRACTION = re.compile(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}")


def _numeric_value(text: str) -> Fraction | None:
    text = text.replace(",", "") if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text) else text
    if re.fullmatch(r"-?(?:\d+(?:\.\d*)?|\.\d+)", text):
        return Fraction(text)
    match = _SIMPLE_FRACTION.fullmatch(text)
    if match is not None and int(match.group(3)) != 0:
        value = Fraction(int(match.group(2)), int(match.group(3)))
        return -value if match.group(1) else value
    return None


def _math_equivalent(prediction: str, answer: str) -> bool:
    left, right = _strip_math(prediction), _strip_math(answer)
    if left == right:
        return True
    left_value, right_value = _numeric_value(left), _numeric_value(right)
    return left_value is not None and left_value == right_value


def _mbpp_tests(row: dict[str, Any]) -> list[str]:
    tests = row.get("test_list", []) or []
    if isinstance(tests, str):
        try:
            tests = ast.literal_eval(tests)
        except (SyntaxError, ValueError):
            return []
    if not isinstance(tests, list) or any(not isinstance(t, str) for t in tests):
        return []
    return tests


_ASSERTION_CALL = re.compile(r"assertion\(\s*([A-Za-z_]\w*)\(\*inp\)")
_BUILTIN_NAMES = frozenset(dir(builtins))


def _entry_point(row: dict[str, Any]) -> str | None:
    found = _ASSERTION_CALL.findall(row.get("plus_test") or "")
    if found:
        return found[-1]
    tests = _mbpp_tests(row)
    tree = _parse(tests[0]) if tests else None
    if tree is None:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in _BUILTIN_NAMES:
                return node.func.id
    return None


class _Block(NamedTuple):
    source: str
    python: bool
    closed: bool


def _fenced_blocks(output: str) -> list[_Block]:
    blocks: list[_Block] = []

    def emit(tag: str, lines: list[str], closed: bool) -> None:
        source = textwrap.dedent("\n".join(lines)).strip()
        if source:
            blocks.append(_Block(source, tag in _PYTHON_TAGS, closed))

    current: tuple[str, int, list[str]] | None = None
    for line in _lines(output):
        fence = _FENCE_LINE.match(line)
        if fence is not None and (current is None or len(fence.group(1)) >= current[1]):
            info = fence.group(2).strip()
            tag = info.split()[0].lower() if info else ""
            if current is None:
                current = (tag, len(fence.group(1)), [])
            elif not info:
                emit(current[0], current[2], closed=True)
                current = None
            else:
                emit(current[0], current[2], closed=False)
                current = (tag, len(fence.group(1)), [])
        elif current is not None:
            current[2].append(line)
    if current is not None:
        emit(current[0], current[2], closed=False)
    return blocks


def _lines(text: str) -> list[str]:
    return re.split(r"\r\n|\r|\n", text)


def _parse(source: str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _defines(tree: ast.Module, name: str) -> bool:
    for node in tree.body:
        if isinstance(node, _DEFINITIONS) and node.name == name:
            return True
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return True
    return False


def _extract_code(output: str, entry: str | None = None) -> str:
    blocks = _fenced_blocks(output)
    parsed = [(block, _parse(block.source)) for block in blocks]
    usable = [(block, tree) for block, tree in parsed if tree is not None]
    if entry:
        defining = [block for block, tree in usable if _defines(tree, entry)]
        for picks in (
            [block for block in defining if block.closed and block.python],
            [block for block in defining if block.closed],
            defining,
        ):
            if picks:
                return _sanitize(picks[-1].source, entry)
    with_function = [
        block
        for block, tree in usable
        if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body)
    ]
    if with_function:
        source = max(with_function, key=lambda block: len(block.source)).source
    elif blocks:
        source = max(blocks, key=lambda block: len(block.source)).source
    else:
        inline = [match.group(1).strip() for match in _FENCE.finditer(output)]
        source = max(inline, key=len) if inline else output.strip()
        if not inline and _parse(source) is None:
            source = _unfenced_code(source, entry) or source
    return _sanitize(source, entry)


_CODE_START = re.compile(r"\b(?:async\s+def|def|class|import|from)\s+\w")


def _unfenced_code(text: str, entry: str | None) -> str | None:
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if _CODE_START.search(line)]
    if entry:
        defining = [i for i in starts if re.search(rf"\bdef\s+{re.escape(entry)}\s*\(", lines[i])]
        if defining:
            first = defining[0]
            while first - 1 in starts and re.search(r"\b(?:import|from)\s+\w", lines[first - 1]):
                first -= 1
            starts = [first]
    for start in starts[:1]:
        head = lines[start]
        head = head[_CODE_START.search(head).start():]
        body = [head, *lines[start + 1 :]]
        for end in range(len(body), 0, -1):
            candidate = "\n".join(body[:end]).rstrip()
            if candidate and _parse(candidate) is not None:
                return candidate
    return None


_DRIVER_CALLS = frozenset(
    {
        "print",
        "input",
        "open",
        "exit",
        "quit",
        "breakpoint",
        "help",
        "sys.exit",
        "os._exit",
        "unittest.main",
        "doctest.testmod",
        "pytest.main",
    }
)


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return ".".join([node.id, *reversed(parts)]) if isinstance(node, ast.Name) else None


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _is_main_guard(node: ast.stmt) -> bool:
    return isinstance(node, ast.If) and ast.unparse(node.test).replace('"', "'") in {
        "__name__ == '__main__'",
        "'__main__' == __name__",
    }


def _drives(node: ast.stmt, defined: set[str]) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Assert):
            return True
        if isinstance(child, ast.Call):
            name = _dotted(child.func)
            if name in _DRIVER_CALLS or name in defined:
                return True
        if isinstance(child, ast.Attribute) and _dotted(child) == "sys.stdin":
            return True
    return False


def _binds(node: ast.stmt) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        stores = isinstance(getattr(child, "ctx", None), (ast.Store, ast.Del))
        if isinstance(child, ast.Name) and stores:
            names.add(child.id)
        elif isinstance(child, (ast.Attribute, ast.Subscript)) and stores:
            names.add(_root_name(child) or "")
        elif isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            names.add(_root_name(child.func.value) or "")
        elif isinstance(child, _DEFINITIONS):
            names.add(child.name)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in child.names)
    return names - {""}


def _reads(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _definition_reads(node: ast.stmt) -> set[str]:
    try:
        table = symtable.symtable(ast.unparse(node), "<candidate>", "exec")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return _reads(node)
    names = {symbol.get_name() for symbol in table.get_symbols() if symbol.is_referenced()}
    scopes = list(table.get_children())
    while scopes:
        scope = scopes.pop()
        names.update(
            symbol.get_name()
            for symbol in scope.get_symbols()
            if symbol.is_global() and (symbol.is_referenced() or symbol.is_declared_global())
        )
        scopes.extend(scope.get_children())
    return names


def _sanitize(source: str, entry: str | None = None) -> str:
    tree = _parse(source)
    if tree is None:
        return source
    try:
        kept = _kept_statements(tree, entry)
        if len(kept) == len(tree.body):
            return source
        sanitized = _splice(source, tree, kept)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return source
    return sanitized if _parse(sanitized) is not None else source


def _kept_statements(tree: ast.Module, entry: str | None) -> set[int]:
    defined = {node.name for node in tree.body if isinstance(node, _DEFINITIONS)}
    imported: set[str] = set()
    kept: set[int] = set()
    pending: list[int] = []
    reads: set[str] = {entry} if entry else set()
    for index, node in enumerate(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            kept.add(index)
            imported |= _binds(node)
        elif isinstance(node, _DEFINITIONS):
            kept.add(index)
            reads |= _definition_reads(node)
        elif isinstance(node, ast.Assert) or _is_main_guard(node):
            continue
        elif _drives(node, defined):
            pending.append(index)
        else:
            kept.add(index)
            reads |= _reads(node)
    grown = True
    while grown:
        grown = False
        for index in [index for index in pending if (_binds(tree.body[index]) - imported) & reads]:
            pending.remove(index)
            kept.add(index)
            reads |= _reads(tree.body[index])
            grown = True
    return kept


def _splice(source: str, tree: ast.Module, kept: set[int]) -> str:
    def left(node: ast.stmt) -> list[str]:
        if not (isinstance(node, ast.If) and _is_main_guard(node)):
            return []
        imports = (ast.Import, ast.ImportFrom)
        return [ast.unparse(child) for child in node.body if isinstance(child, imports)]

    lines = _lines(source)
    replace: dict[int, list[str]] = {}
    for index, node in enumerate(tree.body):
        if index in kept:
            continue
        first, last = node.lineno - 1, (node.end_lineno or node.lineno) - 1
        before = lines[first].encode("utf-8")[: node.col_offset].decode("utf-8", "replace")
        after = lines[last].encode("utf-8")[node.end_col_offset :].decode("utf-8", "replace")
        before, after = before.strip(), after.strip()
        if before or (after and not after.startswith("#")):
            unparsed: list[str] = []
            for number, item in enumerate(tree.body):
                unparsed.extend([ast.unparse(item)] if number in kept else left(item))
            return "\n".join(unparsed)
        replace[first] = left(node)
        for number in range(first + 1, last + 1):
            replace[number] = []
    out: list[str] = []
    for number, line in enumerate(lines):
        out.extend(replace.get(number, [line]))
    return "\n".join(out).strip()


def _code_score(output: str, row: dict[str, Any]) -> float:
    source = _extract_code(output, _entry_point(row))
    if not source:
        return 0.0
    try:
        ast.parse(source)
    except SyntaxError:
        return 0.0
    imports = "\n".join(row.get("test_imports", []) or [])
    return float(_passes(source + "\n\n" + imports + "\n" + row["plus_test"], cpu_seconds=30))


_ONE_THREAD = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def _passes(program: str, *, cpu_seconds: int) -> bool:
    sentinel = f"evosteer-tests-passed-{uuid.uuid4().hex}"
    limits = (
        "import resource as _evosteer_resource\n"
        f"_evosteer_resource.setrlimit(_evosteer_resource.RLIMIT_CPU, ({cpu_seconds}, {cpu_seconds}))\n"
        "_evosteer_resource.setrlimit(_evosteer_resource.RLIMIT_AS, (4 << 30, 4 << 30))\n"
        "_evosteer_resource.setrlimit(_evosteer_resource.RLIMIT_FSIZE, (8 << 20, 8 << 20))\n"
        if sys.platform.startswith("linux")
        else ""
    )
    with tempfile.TemporaryDirectory(prefix="evosteer-mbpp-") as temp:
        path = Path(temp) / "candidate.py"
        path.write_text(limits + program + f"\nprint('\\n' + {sentinel!r})\n", encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONNOUSERSITE": "1", **_ONE_THREAD}
        result = subprocess.run(
            [sys.executable, "-I", str(path)],
            cwd=temp,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=cpu_seconds + 10,
            check=False,
        )
    return result.returncode == 0 and result.stdout.decode("utf-8", "replace").splitlines()[-1:] == [sentinel]


TASK_FAMILIES = frozenset({"hotpotqa", "nq_open", "medqa", "aime_2026", "mbpp_plus", "alfworld"})
EXTERNAL_HARNESS_KINDS = frozenset({"swe_bench_verified", "webshop"})


def task_family(row: dict[str, Any]) -> str:
    task_type = row.get("task_type")
    if task_type in TASK_FAMILIES and row["kind"] != "alfworld":
        return task_type
    return row["kind"]


def _score(kind: str, output: str, row: dict[str, Any]) -> float:
    if kind in QA_KINDS:
        return _qa_em(output, _qa_answers(kind, row))
    if kind == "medqa":
        return float(_choice_letter(output, row["options"]) == row["answer"])
    if kind == "aime_2026":
        if row.get("kind", kind) == "aime_2026":
            return _integer_score(output, row["answer"])
        return _final_value_score(output, row["answer"])
    if kind == "mbpp_plus":
        try:
            return _code_score(output, row)
        except (OSError, subprocess.SubprocessError, ValueError, MemoryError):
            return 0.0
    raise ValueError(f"unsupported curve benchmark: {kind}")


def _load_rows(root: Path) -> list[dict[str, Any]]:
    path = Path(os.environ.get("EVOSTEER_CURVE_DATA", root / "curve_benchmarks.jsonl"))
    if not path.is_file():
        raise FileNotFoundError(f"curve data not found: {path}; run build_curve_benchmarks.py")
    requested_split = os.environ.get("EVOSTEER_CURVE_SPLIT", "train")
    if requested_split not in {"train", "validation", "test", "all"}:
        raise ValueError("EVOSTEER_CURVE_SPLIT must be train, validation, test, or all")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if requested_split != "all":
        rows = [row for row in rows if row.get("split") == requested_split]
    if not rows:
        raise ValueError("curve benchmark manifest is empty")
    return rows


def _executor_model(policy: Any) -> Any:
    url = os.environ.get("EVOSTEER_EXECUTOR_URL")
    if not url:
        return policy
    from skillev.policy.sglang_text import SGLangFrozenText

    return SGLangFrozenText(
        url,
        policy.tokenizer,
        reference_id=f"{policy.reference_id}/sglang-executor",
        context_window=int(os.environ.get("EVOSTEER_EXECUTOR_CONTEXT", "32768")),
        thinking=os.environ.get("EVOSTEER_EXECUTOR_THINKING", "") == "1",
    )


def _environment_config_id(kind: str) -> str:
    return f"curve-{kind}@{3 if kind in QA_KINDS else 2}"


def task_factory(
    *,
    policy: Any,
    config: Any,
    executor_temperature: float = 0.3,
    executor_stop_regex: tuple[str, ...] = (),
) -> tuple[TaskBinding, ...]:
    root = Path(os.environ.get("EVOSTEER_CURVE_DATA_ROOT", "data/curve_benchmarks")).resolve()
    rows = _load_rows(root)
    external = sorted({row["kind"] for row in rows if row["kind"] in EXTERNAL_HARNESS_KINDS})
    if external:
        raise ValueError(f"{', '.join(external)} rows are scored by their own harness (see README.md), not by curve_benchmarks")
    alfworld_rows = [row for row in rows if row["kind"] == "alfworld"]
    rows = [row for row in rows if row["kind"] != "alfworld"]
    executor_model = _executor_model(policy)
    node_options = {"temperature": executor_temperature, "stop_regex": executor_stop_regex}
    executor_id = FrozenTextExecutor(executor_model, **node_options).frozen_identity
    bindings: list[TaskBinding] = []
    for row in rows:
        kind, task_id = task_family(row), row["task_id"]
        prompt = row["prompt"]
        task = EvoTask(task_id, kind, prompt, reset_id="initial", environment_config_id=_environment_config_id(kind))

        def session(request: SessionRequest, *, _row=row, _kind=kind, _task=task) -> TaskSession:
            def evaluate(output: str) -> float:
                return _score(_kind, output, _row)

            return TaskSession(
                FrozenTextExecutor(executor_model, **node_options),
                evaluate,
                reset_receipt=ResetReceipt(_task.identity, _task.reset_id, _task.environment_config_id, _task.identity, f"{_task.task_id}:{request.seed}:{uuid.uuid4().hex}", request.seed),
                risk_assessor=ExecutionRiskPolicy(executor_id, _task.environment_config_id, scope="text_only", capability_id="curve-static-verifier@1"),
            )

        bindings.append(TaskBinding(task, executor_id, session, replay_safe=True))
    if alfworld_rows:
        from skillev.experiments.curve_alfworld import make_bindings as alfworld_bindings

        for mode in sorted({row["alfworld_split"] for row in alfworld_rows}):
            picked = [row for row in alfworld_rows if row["alfworld_split"] == mode]
            made = alfworld_bindings(
                policy=executor_model,
                config=config,
                mode=mode,
                games={int(row["index"]): row["source_id"] for row in picked},
                temperature=executor_temperature,
            )
            if {b.task.task_id for b in made} != {row["task_id"] for row in picked}:
                raise ValueError("ALFWorld manifest task_id must be alfworld/<split>/<index:06d>")
            bindings.extend(made)
    return tuple(bindings)


__all__ = ["QA_KINDS", "TASK_FAMILIES", "qa_metrics", "task_factory", "task_family"]
