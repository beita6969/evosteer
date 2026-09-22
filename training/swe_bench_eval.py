

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

CONDA = os.environ.get("CONDA_EXE") or shutil.which("conda") or "conda"
SWE_ENVS = Path(os.environ.get("SWE_BENCH_ENVS", "swe_bench_envs"))
VERIFIED_DS_PATH = os.environ.get("SWE_BENCH_VERIFIED_PATH", "data/swebench_verified")


_verified_cache: Dict[str, dict] = {}
_swe_bench_specs: Dict = {}
_official_eval_lock = threading.Lock()


@dataclass(frozen=True, slots=True)
class SWEHarnessDiagnostic:
    classification: str
    phase: str
    retryable: bool
    run_id: str
    instance_hash: str
    log_relative_path: str | None
    log_blake2b: str | None
    test_output_present: bool
    report_present: bool
    container_exit_code: int | None
    container_oom_killed: bool | None
    image_key_hash: str | None


class SWEHarnessInfrastructureError(RuntimeError):
    """Fail-closed official-harness error with payload-safe diagnostics."""

    def __init__(self, diagnostic: SWEHarnessDiagnostic):
        self.diagnostic = diagnostic
        super().__init__(
            "SWE-bench infrastructure failure: "
            f"classification={diagnostic.classification} "
            f"phase={diagnostic.phase} "
            f"instance_hash={diagnostic.instance_hash}"
        )


def _exec_run_with_tolerant_decode(container, cmd, timeout=60):
    """Run the pinned harness command while preserving non-UTF-8 test output."""

    exec_result = bytearray()
    exec_id = None
    exception = None
    timed_out = False

    def run_command():
        nonlocal exec_id, exception
        try:
            exec_id = container.client.api.exec_create(container.id, cmd)["Id"]
            for chunk in container.client.api.exec_start(exec_id, stream=True):
                exec_result.extend(chunk)
        except Exception as error:
            exception = error

    thread = threading.Thread(target=run_command)
    start_time = time.time()
    thread.start()
    thread.join(timeout)
    if exception:
        raise exception
    if thread.is_alive():
        if exec_id is not None:
            exec_pid = container.client.api.exec_inspect(exec_id)["Pid"]
            container.exec_run(f"kill -TERM {exec_pid}", detach=True)
        timed_out = True
    elapsed = time.time() - start_time
    return exec_result.decode("utf-8", errors="replace"), timed_out, elapsed


def _make_test_spec_with_retry(make_test_spec, verified, namespace, *, attempts=3):
    """Build an official test spec with bounded retries for transient metadata fetches."""

    import requests

    for attempt in range(1, attempts + 1):
        try:
            return make_test_spec(verified, namespace=namespace)
        except requests.exceptions.RequestException as error:
            if attempt == attempts:
                raise
            delay = float(2 ** (attempt - 1))
            logger.warning(
                "[SWE-eval] transient test-spec metadata failure; "
                "attempt=%s/%s error_class=%s retry_in_seconds=%.1f",
                attempt,
                attempts,
                type(error).__name__,
                delay,
            )
            time.sleep(delay)


def _cpu_subprocess_environment() -> dict:
    """Keep repository builds and tests from acquiring a training GPU context."""

    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _load_verified_dataset():

    global _verified_cache
    if _verified_cache:
        return
    try:
        import datasets
        ds = datasets.load_from_disk(VERIFIED_DS_PATH)
        for row in ds:
            _verified_cache[row["instance_id"]] = {
                "instance_id": row["instance_id"],
                "repo": row["repo"],
                "version": row.get("version", ""),
                "base_commit": row["base_commit"],
                "environment_setup_commit": row["environment_setup_commit"],
                "patch": row["patch"],
                "test_patch": row["test_patch"],
                "FAIL_TO_PASS": json.loads(row["FAIL_TO_PASS"]) if isinstance(row["FAIL_TO_PASS"], str) else row["FAIL_TO_PASS"],
                "PASS_TO_PASS": json.loads(row["PASS_TO_PASS"]) if isinstance(row["PASS_TO_PASS"], str) else row["PASS_TO_PASS"],
            }
        logger.info(f"[SWE-eval] Loaded {len(_verified_cache)} verified instances")
    except Exception as e:
        if os.environ.get("SKILLEV_FORMAL_RUNTIME") == "1":
            raise RuntimeError("official SWE-bench Verified dataset is unavailable") from e
        logger.warning(f"[SWE-eval] Failed to load verified dataset: {e}")


def _load_specs():

    global _swe_bench_specs
    if _swe_bench_specs:
        return
    try:
        import sys
        swebench_repo = os.environ.get("SWEBENCH_HARNESS_PATH", "")
        if swebench_repo and swebench_repo not in sys.path:
            sys.path.insert(0, swebench_repo)
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
        _swe_bench_specs = MAP_REPO_VERSION_TO_SPECS
    except Exception as e:
        if os.environ.get("SKILLEV_FORMAL_RUNTIME") == "1":
            raise RuntimeError("official SWE-bench harness is unavailable") from e
        logger.warning(f"[SWE-eval] Failed to load specs: {e}")


def _repo_dir(repo: str) -> Path:

    return SWE_ENVS / repo.replace("/", "__")


def _env_name(repo: str, version: str) -> str:

    return f"swe_{repo.replace('/', '_')}_{version.replace('.', '')}"


def _env_python(repo: str, version: str) -> Optional[str]:

    name = _env_name(repo, version)
    envs_dir = os.environ.get("CONDA_ENVS_DIR")
    if not envs_dir and os.path.isabs(CONDA):
        envs_dir = str(Path(CONDA).resolve().parents[1] / "envs")
    if not envs_dir:
        return None
    py = str(Path(envs_dir) / name / "bin" / "python")
    if os.path.exists(py):
        return py
    return None


def setup_repo_env(repo: str, version: str, base_commit: str) -> bool:

    repo_path = _repo_dir(repo)
    env_name = _env_name(repo, version)

    _load_specs()
    spec = _swe_bench_specs.get(repo, {}).get(version, {})
    py_version = spec.get("python", "3.9")
    install_cmd = spec.get("install", "python -m pip install -e .")
    formal_runtime = os.environ.get("SKILLEV_FORMAL_RUNTIME") == "1"


    if not repo_path.exists():
        if formal_runtime:
            return False
        logger.info(f"[SWE-eval] Cloning {repo}...")
        url = f"https://github.com/{repo}.git"
        result = subprocess.run(
            ["git", "clone", "--quiet", url, str(repo_path)],
            capture_output=True, text=True, timeout=300,
            env=_cpu_subprocess_environment(),
        )
        if result.returncode != 0:
            logger.error(f"[SWE-eval] Clone failed: returncode={result.returncode}")
            return False


    env_py = _env_python(repo, version)
    if not env_py:
        if formal_runtime:
            return False
        logger.info(f"[SWE-eval] Creating env {env_name} (python={py_version})...")
        result = subprocess.run(
            [CONDA, "create", "-n", env_name, f"python={py_version}", "-y", "-q"],
            capture_output=True, text=True, timeout=300,
            env=_cpu_subprocess_environment(),
        )
        if result.returncode != 0:
            logger.error(f"[SWE-eval] Env creation failed: returncode={result.returncode}")
            return False
        env_py = _env_python(repo, version)


    if not formal_runtime:
        subprocess.run(
            ["git", "checkout", base_commit, "-q"],
            cwd=repo_path,
            capture_output=True,
            env=_cpu_subprocess_environment(),
        )
        subprocess.run(
            ["git", "checkout", ".", "-q"],
            cwd=repo_path,
            capture_output=True,
            env=_cpu_subprocess_environment(),
        )
        commands = install_cmd if isinstance(install_cmd, list) else [install_cmd]
        command_env = _cpu_subprocess_environment()
        command_env["PATH"] = f"{Path(env_py).parent}:{command_env.get('PATH', '')}"
        for command in commands:
            result = subprocess.run(
                ["bash", "-lc", str(command)],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=1200,
                env=command_env,
            )
            if result.returncode != 0:
                logger.warning(
                    f"[SWE-eval] Install warning: returncode={result.returncode}"
                )
                return False

    return True


def evaluate_patch(
    instance_id: str,
    model_patch: str,
    extra: Optional[Dict] = None,
    timeout: int = 60,
) -> Tuple[bool, float, str]:

    if not model_patch.strip():
        return False, 0.0, "empty_patch"

    _load_verified_dataset()
    verified = _verified_cache.get(instance_id)
    if not verified:
        raise _configuration_error("instance_not_in_verified", instance_id)

    repo = verified["repo"]
    version = verified["version"]
    base_commit = verified["base_commit"]
    test_patch = verified["test_patch"]
    fail_to_pass = verified["FAIL_TO_PASS"]

    if not fail_to_pass:
        raise _configuration_error("no_fail_to_pass_tests", instance_id)

    if os.environ.get("SKILLEV_FORMAL_RUNTIME") == "1":
        return _evaluate_patch_with_official_harness(
            instance_id=instance_id,
            model_patch=model_patch,
            verified=verified,
            timeout=timeout,
        )


    env_py = _env_python(repo, version)
    repo_path = _repo_dir(repo)
    if not env_py or not repo_path.exists():

        if not setup_repo_env(repo, version, base_commit):
            return False, 0.0, "env_not_ready"
        env_py = _env_python(repo, version)

    _load_specs()
    spec = _swe_bench_specs.get(repo, {}).get(version, {})
    test_cmd_template = spec.get("test_cmd", "pytest -rA")
    if isinstance(test_cmd_template, list):
        test_cmd_template = test_cmd_template[-1]


    import tempfile, shutil
    eval_dir = Path(tempfile.mkdtemp(prefix="swe_eval_"))
    worktree_path = eval_dir / "repo"

    try:

        worktree_result = subprocess.run(
            ["git", "worktree", "add", str(worktree_path), base_commit, "-q", "--detach"],
            cwd=repo_path, capture_output=True, timeout=30,
            env=_cpu_subprocess_environment(),
        )
        if worktree_result.returncode != 0:
            return False, 0.0, "worktree_create_failed"


        for so_file in repo_path.glob("**/*.so"):
            rel = so_file.relative_to(repo_path)
            dest = worktree_path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(str(so_file), str(dest))
            except Exception:
                pass


        patch_file = str(worktree_path / "_model.patch")
        with open(patch_file, "w") as f:
            f.write(model_patch)
        result = subprocess.run(
            ["git", "apply", "--ignore-whitespace", patch_file],
            cwd=worktree_path, capture_output=True, text=True, timeout=10,
            env=_cpu_subprocess_environment(),
        )
        if result.returncode != 0:
            result = subprocess.run(
                ["git", "apply", "--ignore-whitespace", "--3way", patch_file],
                cwd=worktree_path, capture_output=True, text=True, timeout=10,
                env=_cpu_subprocess_environment(),
            )
        if result.returncode != 0:
            return False, 0.0, f"patch_apply_failed({result.stderr[:100]})"


        test_file = str(worktree_path / "_test.patch")
        with open(test_file, "w") as f:
            f.write(test_patch)
        test_patch_result = subprocess.run(
            ["git", "apply", "--ignore-whitespace", test_file],
            cwd=worktree_path, capture_output=True, timeout=10,
            env=_cpu_subprocess_environment(),
        )
        if test_patch_result.returncode != 0:
            return False, 0.0, "test_patch_apply_failed"


        _needs_build = {"scikit-learn/scikit-learn"}
        if repo in _needs_build:
            subprocess.run(
                [str(env_py), "setup.py", "build_ext", "--inplace"],
                cwd=worktree_path, capture_output=True, timeout=600,
                env=_cpu_subprocess_environment(),
            )


        passed = 0
        total = len(fail_to_pass)

        for test_id in fail_to_pass:
            test_result = _run_single_test(env_py, worktree_path, repo, test_cmd_template, test_id, timeout=timeout)
            if test_result:
                passed += 1

        resolved = passed == total
        score = passed / total if total > 0 else 0.0
        details = f"resolved={'Y' if resolved else 'N'}, pass={passed}/{total}"
        return resolved, score, details

    except subprocess.TimeoutExpired:
        return False, 0.0, "timeout"
    except Exception as e:
        return False, 0.0, f"error({str(e)[:80]})"
    finally:

        try:
            subprocess.run(["git", "worktree", "remove", str(worktree_path), "--force"],
                          cwd=repo_path, capture_output=True, timeout=10,
                          env=_cpu_subprocess_environment())
            shutil.rmtree(str(eval_dir), ignore_errors=True)
        except Exception:
            pass


def _run_single_test(env_py: str, repo_path: Path, repo: str, test_cmd_template: str, test_id: str, timeout: int = 60) -> bool:


    if "runtests.py" in test_cmd_template:


        import re
        m = re.match(r'(\w+)\s+\(([^)]+)\)', test_id)
        if m:
            method, class_path = m.group(1), m.group(2)

            test_arg = f"{class_path}.{method}"
        else:
            test_arg = test_id

        cmd = [env_py, "./tests/runtests.py", "--settings=test_sqlite", "--parallel", "1", test_arg]
    else:

        cmd = [env_py, "-m", "pytest", "-xvs", test_id]

    try:
        result = subprocess.run(
            cmd, cwd=str(repo_path),
            capture_output=True, text=True, timeout=timeout,
            env=_cpu_subprocess_environment(),
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def _short_blake2b(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _configuration_error(classification: str, instance_id: str) -> SWEHarnessInfrastructureError:
    return SWEHarnessInfrastructureError(
        SWEHarnessDiagnostic(
            classification=classification,
            phase="configuration",
            retryable=False,
            run_id="",
            instance_hash=_short_blake2b(instance_id),
            log_relative_path=None,
            log_blake2b=None,
            test_output_present=False,
            report_present=False,
            container_exit_code=None,
            container_oom_killed=None,
            image_key_hash=None,
        )
    )


def _tail_bytes(path: Path, limit: int = 256 * 1024) -> bytes:
    if not path.is_file():
        return b""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - limit))
        return handle.read(limit)


def _classify_missing_report(log_tail: bytes, test_output_tail: bytes) -> tuple[str, str, bool]:
    normalized = (log_tail + b"\n" + test_output_tail).decode(
        "utf-8", errors="replace"
    ).lower()
    if "apply_patch_fail" in normalized or "patch apply failed" in normalized:
        return "patch_apply_failed", "patch_apply", False
    if "test timed out after" in normalized or "timeout error:" in normalized:
        return "test_timeout", "test", False
    if any(
        marker in normalized
        for marker in ("manifest unknown", "pull access denied", "no such image")
    ):
        return "image_missing_or_pull_failed", "image", False
    if "buildimageerror" in normalized:
        return "image_build_failed", "image", False
    if "permission denied" in normalized or "operation not permitted" in normalized:
        return "docker_permission_denied", "docker", False
    transport_markers = (
        "dockerexception",
        "connection aborted",
        "connection refused",
        "connection reset",
        "read timed out",
        "protocolerror",
        "broken pipe",
    )
    if any(marker in normalized for marker in transport_markers):
        return "docker_transport_transient", "docker", True
    if "containererror" in normalized and "create" in normalized:
        return "container_create_failed", "container", False
    if "containererror" in normalized and "start" in normalized:
        return "container_start_failed", "container", False
    if test_output_tail:
        return "grading_failed", "grading", False
    return "report_missing_unknown", "report", False


def _read_report(report_path: Path, instance_id: str) -> dict | None:
    if not report_path.is_file():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise _configuration_error("report_malformed", instance_id) from error
    return payload if isinstance(payload, dict) else None


def _official_instance_log(log_dir: Path) -> Path:
    """Return the log name written by the pinned harness.

    The pinned SWE-bench revision writes ``run_instance.log``.  Retaining the
    older fallback keeps diagnostics readable for already-created runs.
    """

    for name in ("run_instance.log", "instance.log"):
        candidate = log_dir / name
        if candidate.is_file():
            return candidate
    return log_dir / "run_instance.log"


def _evaluate_patch_with_official_harness(
    *,
    instance_id: str,
    model_patch: str,
    verified: dict,
    timeout: int,
) -> Tuple[bool, float, str]:
    """Evaluate one patch in the official immutable SWE-bench Docker image."""

    evaluation_root = Path(os.environ["SWE_BENCH_EVALUATION_ROOT"])
    evaluation_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    harness_root = os.environ["SWEBENCH_HARNESS_PATH"]
    if harness_root not in sys.path:
        sys.path.insert(0, harness_root)
    try:
        import docker
        from swebench.harness import run_evaluation
        from swebench.harness.test_spec.test_spec import make_test_spec
    except Exception as error:
        raise RuntimeError("official SWE-bench evaluator is unavailable") from error
    run_evaluation.RUN_EVALUATION_LOG_DIR = evaluation_root / "logs"
    prediction = {
        "instance_id": instance_id,
        "model_name_or_path": "skillflow-baseline",
        "model_patch": model_patch,
    }
    evaluation_id = hashlib.blake2b(
        f"{instance_id}\0{model_patch}".encode(), digest_size=12
    ).hexdigest()
    run_id = f"{evaluation_id}-{uuid.uuid4().hex[:12]}"
    log_dir = evaluation_root / "logs" / run_id / "skillflow-baseline" / instance_id
    report_path = log_dir / "report.json"
    test_output_path = log_dir / "test_output.txt"
    captured: dict[str, object] = {}
    client = docker.from_env(timeout=max(timeout, 60))
    try:
        with _official_eval_lock:
            test_spec = _make_test_spec_with_retry(
                make_test_spec,
                verified,
                os.environ["SWE_BENCH_DOCKER_NAMESPACE"],
            )
            image_key = str(getattr(test_spec, "instance_image_key", ""))
            original_exec = run_evaluation.exec_run_with_timeout
            original_cleanup = run_evaluation.cleanup_container

            def cleanup_with_capture(client_arg, container, logger_arg):
                if container is not None:
                    try:
                        container.reload()
                        state = container.attrs.get("State", {})
                        captured["exit_code"] = state.get("ExitCode")
                        captured["oom_killed"] = state.get("OOMKilled")
                    except Exception:
                        pass
                return original_cleanup(client_arg, container, logger_arg)

            run_evaluation.exec_run_with_timeout = _exec_run_with_tolerant_decode
            run_evaluation.cleanup_container = cleanup_with_capture
            try:
                outcome = run_evaluation.run_instance(
                    test_spec,
                    prediction,
                    False,
                    False,
                    client,
                    run_id,
                    timeout=timeout,
                )
            finally:
                run_evaluation.exec_run_with_timeout = original_exec
                run_evaluation.cleanup_container = original_cleanup
    finally:
        client.close()
    report = outcome[1] if isinstance(outcome, tuple) and len(outcome) == 2 else None
    if report is None:
        report = _read_report(report_path, instance_id)
    if report is None:
        instance_log = _official_instance_log(log_dir)
        log_tail = _tail_bytes(instance_log)
        test_output_tail = _tail_bytes(test_output_path)
        classification, phase, retryable = _classify_missing_report(
            log_tail, test_output_tail
        )
        if classification in {"patch_apply_failed", "test_timeout"}:
            return False, 0.0, classification
        relative_log = None
        if instance_log.is_file():
            relative_log = str(instance_log.relative_to(evaluation_root))
        raise SWEHarnessInfrastructureError(
            SWEHarnessDiagnostic(
                classification=classification,
                phase=phase,
                retryable=retryable,
                run_id=run_id,
                instance_hash=_short_blake2b(instance_id),
                log_relative_path=relative_log,
                log_blake2b=_short_blake2b(instance_log.read_bytes())
                if instance_log.is_file()
                else None,
                test_output_present=test_output_path.is_file(),
                report_present=report_path.is_file(),
                container_exit_code=captured.get("exit_code")
                if isinstance(captured.get("exit_code"), int)
                else None,
                container_oom_killed=captured.get("oom_killed")
                if isinstance(captured.get("oom_killed"), bool)
                else None,
                image_key_hash=_short_blake2b(image_key) if image_key else None,
            )
        )
    if not isinstance(report, dict) or not isinstance(report.get(instance_id), dict):
        raise _configuration_error("report_malformed", instance_id)
    resolved = report[instance_id].get("resolved")
    if not isinstance(resolved, bool):
        raise _configuration_error("report_malformed", instance_id)
    return resolved, float(resolved), "resolved" if resolved else "unresolved"
