"""One-shot private SpreadsheetBench OJ worker.

The caller supplies a model-edited workbook path and an owner-only route.  The
worker recalculates a private copy with LibreOffice and invokes the pinned
SpreadsheetBench ``compare_workbooks`` implementation.  Oracle workbooks and
answer ranges never enter the rollout process.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, cast

_CELL_OR_RANGE = re.compile(r"([A-Za-z]+)([0-9]+)(?::([A-Za-z]+)([0-9]+))?")
_COLUMN_RANGE = re.compile(r"([A-Za-z]+):([A-Za-z]+)")
_ROW_RANGE = re.compile(r"([0-9]+):([0-9]+)")
_SAME_COLUMN_RANGE = re.compile(r"([A-Za-z]+)([0-9]+):([0-9]+)")


def _absolute_file(value: str, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{label} must be an absolute file")
    return path.resolve()


def _absolute_directory(value: str, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir():
        raise ValueError(f"{label} must be an absolute directory")
    return path.resolve()


def _relative_route(value: object, *, label: str) -> PurePosixPath:
    if type(value) is not str:
        raise TypeError(f"{label} must be text")
    route = PurePosixPath(value)
    if not value.strip() or route.is_absolute() or ".." in route.parts or route.as_posix() != value:
        raise ValueError(f"{label} must be normalized")
    return route


def _load_compare(
    source_root: Path,
) -> Callable[[str, str, str, str], tuple[bool, str]]:
    path = source_root / "evaluation" / "evaluation.py"
    if not path.is_file():
        raise ValueError("pinned SpreadsheetBench evaluator is absent")
    spec = importlib.util.spec_from_file_location("skillev_spreadsheetbench_evaluation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("pinned SpreadsheetBench evaluator cannot be loaded")
    module = ModuleType(spec.name)
    spec.loader.exec_module(module)
    compare = getattr(module, "compare_workbooks", None)
    if not callable(compare):
        raise RuntimeError("pinned SpreadsheetBench compare_workbooks is unavailable")
    return cast(Callable[[str, str, str, str], tuple[bool, str]], compare)


def _load_cell_compare(source_root: Path) -> Callable[[Any, Any, str, str], tuple[bool, str]]:
    path = source_root / "evaluation" / "evaluation.py"
    spec = importlib.util.spec_from_file_location("skillev_spreadsheetbench_cell_evaluation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("pinned SpreadsheetBench evaluator cannot be loaded")
    module = ModuleType(spec.name)
    spec.loader.exec_module(module)
    compare = getattr(module, "cell_level_compare", None)
    if not callable(compare):
        raise RuntimeError("pinned SpreadsheetBench cell comparator is unavailable")
    return cast(Callable[[Any, Any, str, str], tuple[bool, str]], compare)


def _split_answer_positions(value: str) -> tuple[str, ...]:
    """Split upstream ranges without treating commas inside quoted sheet names as separators."""

    parts: list[str] = []
    current: list[str] = []
    quoted = False
    index = 0
    while index < len(value):
        character = value[index]
        if character == "'":
            if quoted and index + 1 < len(value) and value[index + 1] == "'":
                current.extend(("'", "'"))
                index += 2
                continue
            if quoted:
                quoted = False
            elif not "".join(current).strip():
                quoted = True
        if character == "," and not quoted:
            part = "".join(current).strip()
            if not part:
                raise ValueError("SpreadsheetBench answer position contains an empty range")
            parts.append(part)
            current = []
        else:
            current.append(character)
        index += 1
    if quoted:
        raise ValueError("SpreadsheetBench answer position has an unterminated sheet quote")
    part = "".join(current).strip()
    if not part:
        raise ValueError("SpreadsheetBench answer position contains an empty range")
    parts.append(part)
    return tuple(parts)


def _parse_answer_position_part(value: str, *, default_sheet: str) -> tuple[str, str]:
    if "!" in value:
        sheet_name, cell_range = value.split("!", 1)
        sheet_name = sheet_name.strip().strip("'").replace("''", "'")
    else:
        sheet_name = default_sheet
        cell_range = value
    cell_range = cell_range.strip().strip("'")
    if not sheet_name or not cell_range:
        raise ValueError("SpreadsheetBench answer position is incomplete")
    return sheet_name, cell_range


def _column_name(number: int) -> str:
    if number < 1:
        raise ValueError("SpreadsheetBench worksheet has no columns")
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def _normalize_cell_range(value: str, *, max_row: int, max_column: int) -> str:
    if match := _CELL_OR_RANGE.fullmatch(value):
        start_column, start_row, end_column, end_row = match.groups()
        start = f"{start_column.upper()}{start_row}"
        return start if end_column is None else f"{start}:{end_column.upper()}{end_row}"
    if match := _COLUMN_RANGE.fullmatch(value):
        start_column, end_column = match.groups()
        return f"{start_column.upper()}1:{end_column.upper()}{max(1, max_row)}"
    if match := _ROW_RANGE.fullmatch(value):
        start_row, end_row = match.groups()
        return f"A{start_row}:{_column_name(max(1, max_column))}{end_row}"
    if match := _SAME_COLUMN_RANGE.fullmatch(value):
        column, start_row, end_row = match.groups()
        return f"{column.upper()}{start_row}:{column.upper()}{end_row}"
    raise ValueError("SpreadsheetBench answer position has an unsupported cell range")


def _upstream_answer_position_is_safe(value: str) -> bool:
    """The pinned parser accepts only comma-separated cell or rectangular cell ranges."""

    for part in value.split(","):
        cell_range = part.split("!", 1)[-1].strip("'")
        if _CELL_OR_RANGE.fullmatch(cell_range) is None:
            return False
    return True


def _canonical_answer_ranges(oracle: Path, value: str) -> tuple[tuple[str, str], ...]:
    load_workbook = importlib.import_module("openpyxl").load_workbook
    workbook = load_workbook(filename=oracle, data_only=True, read_only=True)
    try:
        if not workbook.sheetnames:
            raise ValueError("SpreadsheetBench oracle has no worksheets")
        default_sheet = workbook.sheetnames[0]
        result: list[tuple[str, str]] = []
        for part in _split_answer_positions(value):
            sheet_name, cell_range = _parse_answer_position_part(part, default_sheet=default_sheet)
            if sheet_name not in workbook.sheetnames:
                raise ValueError("SpreadsheetBench answer position names an absent worksheet")
            worksheet = workbook[sheet_name]
            result.append(
                (
                    sheet_name,
                    _normalize_cell_range(
                        cell_range,
                        max_row=worksheet.max_row,
                        max_column=worksheet.max_column,
                    ),
                )
            )
        return tuple(result)
    finally:
        workbook.close()


def _compare_canonical_ranges(
    source_root: Path,
    oracle: Path,
    recalculated: Path,
    ranges: tuple[tuple[str, str], ...],
) -> bool:
    load_workbook = importlib.import_module("openpyxl").load_workbook
    compare = _load_cell_compare(source_root)
    expected = load_workbook(filename=oracle, data_only=True, read_only=True)
    submitted = load_workbook(filename=recalculated, data_only=True, read_only=True)
    try:
        return all(
            compare(expected, submitted, sheet_name, cell_range)[0]
            for sheet_name, cell_range in ranges
        )
    finally:
        expected.close()
        submitted.close()


def _recalculate(
    submitted: Path,
    *,
    libreoffice: Path,
    temporary_root: Path,
    timeout_seconds: float,
) -> Path:
    directory = Path(tempfile.mkdtemp(prefix="spreadsheet-oj-", dir=temporary_root)).resolve()
    incoming = directory / "incoming"
    output = directory / "output"
    profile = directory / "profile"
    incoming.mkdir()
    output.mkdir()
    profile.mkdir()
    source = incoming / "submitted.xlsx"
    shutil.copy2(submitted, source)
    completed = subprocess.run(  # noqa: S603 - deployment-pinned LibreOffice binary
        (
            str(libreoffice),
            "--headless",
            f"-env:UserInstallation=file://{profile}",
            "--convert-to",
            "xlsx",
            "--outdir",
            str(output),
            str(source),
        ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout_seconds,
        check=False,
    )
    recalculated = output / "submitted.xlsx"
    if completed.returncode != 0 or not recalculated.is_file():
        shutil.rmtree(directory)
        raise RuntimeError("LibreOffice workbook recalculation failed")
    return recalculated


def _training_oracle(
    archive: Path,
    route: PurePosixPath,
    *,
    temporary_root: Path,
) -> Path:
    directory = Path(tempfile.mkdtemp(prefix="spreadsheet-oracle-", dir=temporary_root)).resolve()
    target = directory / "target.xlsx"
    with zipfile.ZipFile(archive) as source:
        try:
            target.write_bytes(source.read(f"{route.as_posix()}/target.xlsx"))
        except KeyError as error:
            shutil.rmtree(directory)
            raise ValueError("spreadsheet training oracle is absent") from error
    return target


def _verified_oracle(root: Path, route: PurePosixPath) -> Path:
    directory = (root / Path(*route.parts)).resolve()
    if not directory.is_relative_to(root) or not directory.is_dir():
        raise ValueError("SpreadsheetBench verified route is absent")
    candidates = tuple(directory.glob("*_golden.xlsx"))
    unprefixed = directory / "golden.xlsx"
    if unprefixed.is_file():
        candidates += (unprefixed,)
    if len(candidates) != 1:
        raise ValueError("SpreadsheetBench route has no unique golden workbook")
    return candidates[0]


def _object(value: object, *, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} has incompatible fields")
    return value


def main(arguments: argparse.Namespace) -> None:
    source_root = _absolute_directory(arguments.official_source_root, label="official source root")
    training_archive = _absolute_file(arguments.training_archive, label="training archive")
    verified_root = _absolute_directory(arguments.verified_root, label="verified root")
    libreoffice = _absolute_file(arguments.libreoffice, label="LibreOffice")
    workspace_root = _absolute_directory(arguments.workspace_root, label="workspace root")
    temporary_root = _absolute_directory(arguments.temporary_root, label="temporary root")
    compare = _load_compare(source_root)
    request = _object(
        json.loads(sys.stdin.buffer.readline()),
        fields={"operation", "submitted_workbook", "task_id"},
        label="SpreadsheetBench worker request",
    )
    if request["operation"] != "grade" or type(request["task_id"]) is not str:
        raise ValueError("SpreadsheetBench worker operation is unsupported")
    submission = _object(
        request["submitted_workbook"],
        fields={"private_payload", "submitted_workbook_path", "workspace_id"},
        label="SpreadsheetBench submission",
    )
    if submission["workspace_id"] != request["task_id"]:
        raise ValueError("SpreadsheetBench workspace identity differs from its task")
    submitted = _absolute_file(
        submission["submitted_workbook_path"],
        label="submitted workbook",
    )
    if not submitted.is_relative_to(workspace_root):
        raise ValueError("submitted workbook is outside the private workspace root")
    payload = submission["private_payload"]
    if not isinstance(payload, dict):
        raise TypeError("SpreadsheetBench private payload must be an object")
    answer_position = payload.get("answer_position")
    if type(answer_position) is not str or not answer_position.strip():
        raise ValueError("SpreadsheetBench answer position is unavailable")
    training_route = payload.get("spreadsheet_task_route")
    verified_route = payload.get("spreadsheet_relative_path")
    if type(training_route) is str and verified_route is None:
        oracle = _training_oracle(
            training_archive,
            _relative_route(training_route, label="training route"),
            temporary_root=temporary_root,
        )
        oracle_temporary = oracle.parent
    elif type(verified_route) is str and training_route is None:
        oracle = _verified_oracle(
            verified_root,
            _relative_route(verified_route, label="verified route"),
        )
        oracle_temporary = None
    else:
        raise ValueError("SpreadsheetBench private payload has no unique oracle route")
    recalculated = _recalculate(
        submitted,
        libreoffice=libreoffice,
        temporary_root=temporary_root,
        timeout_seconds=arguments.timeout_seconds,
    )
    recalculated_temporary = recalculated.parent.parent
    try:
        # The upstream evaluator prints a success sentence.  Worker stdout is
        # reserved for the single JSON response, so contain that diagnostic.
        with contextlib.redirect_stdout(io.StringIO()):
            if _upstream_answer_position_is_safe(answer_position):
                passed, _ = compare(str(oracle), str(recalculated), "", answer_position)
            else:
                ranges = _canonical_answer_ranges(oracle, answer_position)
                if all("," not in sheet_name and "'" not in sheet_name for sheet_name, _ in ranges):
                    canonical = ",".join(
                        f"'{sheet_name}'!{cell_range}" for sheet_name, cell_range in ranges
                    )
                    passed, _ = compare(str(oracle), str(recalculated), "", canonical)
                else:
                    passed = _compare_canonical_ranges(source_root, oracle, recalculated, ranges)
    finally:
        shutil.rmtree(recalculated_temporary)
        if oracle_temporary is not None:
            shutil.rmtree(oracle_temporary)
    json.dump(
        {"passed_case_count": int(bool(passed)), "total_case_count": 1},
        sys.stdout,
        separators=(",", ":"),
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source-root", required=True)
    parser.add_argument("--training-archive", required=True)
    parser.add_argument("--verified-root", required=True)
    parser.add_argument("--libreoffice", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--temporary-root", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    main(parser.parse_args())
