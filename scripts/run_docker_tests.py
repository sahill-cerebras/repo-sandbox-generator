# #!/usr/bin/env python3
# """
# Parallelized build-and-test runner for multiple Docker work instances.

# Features:
#  - Parallel build + test per instance (ProcessPoolExecutor).
#  - Per-instance logs & artifacts (test_results.json, test_output.txt, logs/test_run.log).
#  - Global progress logger (docker_test_runs.log + stdout).
#  - Immediate image removal after each test run (no pruning phase needed).
#  - Respects --force to re-run instances with existing results.
#  - Concurrency control via --jobs.
#  - Robust error handling; failures isolated.
#  - Aggregated summary JSON (docker_test_summary.json).

# Usage:
#     python scripts/run_docker_tests.py \
#             --instances-dir work_instances \
#             --limit 50 \
#             --jobs 4
# """
# from __future__ import annotations

# import argparse, json, logging, re, shutil, subprocess, sys, traceback, multiprocessing
# from dataclasses import dataclass
# from pathlib import Path
# from typing import List, Dict, Optional, Any, Tuple
# from concurrent.futures import ProcessPoolExecutor, as_completed

# # ---------------- Logging helpers -----------------

# def setup_global_logger(log_path: Path) -> logging.Logger:
#     """
#     Global logger (main process). Per-instance logs are created in worker processes.
#     """
#     logger = logging.getLogger("docker_tests_global")
#     logger.setLevel(logging.INFO)
#     if not logger.handlers:
#         fh = logging.FileHandler(log_path)
#         fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
#         ch = logging.StreamHandler(sys.stdout)
#         ch.setFormatter(logging.Formatter('%(message)s'))
#         logger.addHandler(fh)
#         logger.addHandler(ch)
#     return logger

# def setup_instance_logger(instance_dir: Path) -> logging.Logger:
#     """
#     Per-instance logger. Safe to call in worker processes.
#     """
#     logs_dir = instance_dir / 'logs'
#     logs_dir.mkdir(parents=True, exist_ok=True)
#     logger = logging.getLogger(f"docker_tests.{instance_dir.name}")
#     logger.setLevel(logging.INFO)
#     # Reset handlers for idempotency (important for repeated runs within same process)
#     logger.handlers = []  # type: ignore
#     fh = logging.FileHandler(logs_dir / 'test_run.log')
#     fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
#     logger.addHandler(fh)
#     # Do not add stream handler here (workers shouldn't spam main stdout)
#     return logger

# # ---------------- Subprocess helper -----------------

# def run(cmd: List[str], cwd: Optional[Path] = None, capture: bool = False, text: bool = True, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
#     """
#     Wrapper around subprocess.run. Returns CompletedProcess. Uses str(cwd) if provided.
#     """
#     try:
#         return subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=capture, text=text, env=env, check=False)
#     except Exception as e:
#         # Build a fake CompletedProcess-like object to carry the error info
#         cp = subprocess.CompletedProcess(args=cmd, returncode=255, stdout='', stderr=str(e))
#         return cp

# def docker_available() -> bool:
#     return shutil.which("docker") is not None

# def sanitize_tag(name: str) -> str:
#     return re.sub(r'[^a-zA-Z0-9_.-]', '_', name.lower())

# # ---------------- Pytest output parsing -----------------

# COUNT_PATTERN = re.compile(r"(\d+)\s+(passed|failed|skipped|xfailed|xpassed|errors?|error)s?\b", re.IGNORECASE)

# def parse_pytest_output(output: str) -> Dict[str, int]:
#     """
#     Parse pytest summary counts from output. Returns map of counts.
#     Scans from the end to find the last summary-like line containing 'passed' or 'failed'.
#     """
#     counts: Dict[str, int] = {}
#     for line in reversed(output.splitlines()):
#         if ('passed' in line.lower() or 'failed' in line.lower() or 'skipped' in line.lower() or 'error' in line.lower()):
#             for m in COUNT_PATTERN.finditer(line):
#                 n = int(m.group(1))
#                 key = m.group(2).lower()
#                 # normalize variations
#                 if key.startswith('error'):
#                     key = 'errors'
#                 if key.endswith('s') and key not in ('xpassed','xfailed'):
#                     key = key.rstrip('s')
#                 counts[key] = counts.get(key, 0) + n
#             if counts:
#                 break
#     return counts

# # ---------------- Data structures -----------------

# @dataclass
# class TestResult:
#     instance_id: str
#     image_tag: str
#     build_ok: bool
#     test_invoked: bool
#     counts: Dict[str, int]
#     return_code: int
#     error: Optional[str]

#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             "instance_id": self.instance_id,
#             "image_tag": self.image_tag,
#             "build_ok": self.build_ok,
#             "test_invoked": self.test_invoked,
#             "counts": self.counts,
#             "return_code": self.return_code,
#             "error": self.error,
#         }

# # ---------------- Core operations (usable in workers) -----------------

# def build_image(instance_dir: Path, logger: logging.Logger) -> Tuple[bool, str, Optional[str]]:
#     """
#     Build docker image for given instance dir. Returns (success, tag, stderr_or_error).
#     """
#     dockerfile = instance_dir / 'Dockerfile'
#     instance_id = instance_dir.name
#     if not dockerfile.exists():
#         return False, '', 'Dockerfile missing'
#     tag = f"sandbox_{sanitize_tag(instance_id)}:latest"
#     cmd = ["docker", "build", "-t", tag, str(instance_dir)]
#     logger.info("Building image %s from %s", tag, instance_dir)
#     proc = run(cmd, capture=True)
#     if proc.returncode != 0:
#         # include some stderr/ stdout context
#         stderr_sample = (proc.stderr or '')[-2000:]
#         logger.error("Build failed for %s (rc=%s). stderr_sample: %s", instance_id, proc.returncode, stderr_sample)
#         return False, tag, proc.stderr or f"returncode={proc.returncode}"
#     logger.info("Built image %s", tag)
#     return True, tag, None

# def run_tests_individually(tag: str, tests: List[str], logger: logging.Logger) -> Tuple[Dict[str, int], str]:
#     """
#     Run each pytest node individually inside the container to isolate invalid node ids.
#     Returns (counts, concatenated_output).
#     """
#     counts = {"passed": 0, "failed": 0, "errors": 0, "invalid": 0}
#     outputs: List[str] = []
#     for t in tests:
#         cmd = ["docker", "run", "--rm", tag, "pytest", "-q", t]
#         proc = run(cmd, capture=True)
#         single_out = (proc.stdout or '') + '\n' + (proc.stderr or '')
#         outputs.append(f"===== {t} (rc={proc.returncode}) =====\n{single_out}\n")
#         rc = proc.returncode
#         if rc == 0:
#             counts["passed"] += 1
#         elif rc == 1:
#             counts["failed"] += 1
#         elif rc in (4, 5):  # usage error or no tests collected
#             counts["invalid"] += 1
#         else:
#             counts["errors"] += 1
#     logger.info("Per-test execution summary: %s", counts)
#     return counts, '\n'.join(outputs)

# def run_tests_in_container(tag: str, instance_dir: Path, logger: logging.Logger) -> Tuple[bool, Dict[str, int], int, Optional[str], str]:
#     """
#     Run tests inside a container created from tag, reading test_cases_pass_to_pass.json from the instance directory (which is baked into the image).
#     Returns (ok_flag, counts, return_code, error_msg_or_none, raw_output).
#     """
#     tests_file = instance_dir / 'test_cases_pass_to_pass.json'
#     if not tests_file.exists():
#         return False, {}, 0, 'test_cases_pass_to_pass.json missing', ''
#     try:
#         tests = json.loads(tests_file.read_text())
#     except Exception as e:
#         return False, {}, 0, f'cannot read tests list: {e}', ''
#     if not isinstance(tests, list):
#         return False, {}, 0, 'tests file not a list', ''
#     if not tests:
#         return True, {"passed": 0}, 0, None, 'NO_TESTS'

#     # Inline python harness to ensure consistent summary formatting (same as original)
#     harness = ("import json,pytest,sys;"
#                "tests=json.load(open('test_cases_pass_to_pass.json'));"
#                "rc=pytest.main(tests);sys.exit(rc)")
#     cmd = ["docker", "run", "--rm", tag, "python", "-c", harness]
#     proc = run(cmd, capture=True)
#     output = (proc.stdout or '') + '\n' + (proc.stderr or '')
#     counts = parse_pytest_output(output)

#     # If pytest returns usage error, fall back to running individually
#     if proc.returncode == 4:
#         logger.warning("Batch test invocation returned usage error (rc=4). Falling back to per-test execution.")
#         indiv_counts, indiv_output = run_tests_individually(tag, tests, logger)
#         output += '\n' + indiv_output
#         return True, indiv_counts, proc.returncode, None, output

#     if proc.returncode != 0 and not counts:
#         logger.error("Tests failed without parsable summary (rc=%s). Raw output saved.", proc.returncode)
#         return False, counts, proc.returncode, 'no summary parsed', output

#     logger.info("Test summary %s (rc=%s)", counts, proc.returncode)
#     return True, counts, proc.returncode, None, output

# # ---------------- Image removal helpers (modified) -----------------

# def remove_image(tag: str, logger: logging.Logger):
#     if not tag:
#         return
#     proc = run(["docker", "rmi", "-f", tag], capture=True)
#     if proc.returncode != 0:
#         logger.warning("Could not remove image %s (rc=%s): %s", tag, proc.returncode, (proc.stderr or '')[-300:])
#     else:
#         logger.info("Removed image %s", tag)

# def prune_all_docker(logger: logging.Logger):
#     cmds = [
#         ["docker", "container", "prune", "-f"],
#         ["docker", "image", "prune", "-a", "-f"],
#         ["docker", "builder", "prune", "-a", "-f"],
#         ["docker", "volume", "prune", "-f"],
#         ["docker", "network", "prune", "-f"],
#         ["docker", "system", "prune", "-a", "-f"],
#     ]
#     for cmd in cmds:
#         proc = run(cmd, capture=True)
#         if proc.returncode != 0:
#             logger.warning("Prune command failed (%s): %s", " ".join(cmd), proc.stderr)
#         else:
#             logger.info("Ran: %s", " ".join(cmd))

# # ---------------- Worker function (patched) -----------------

# def process_instance_worker(instance_dir_str: str, force: bool) -> Dict[str, Any]:
#     instance_dir = Path(instance_dir_str)
#     inst_logger = setup_instance_logger(instance_dir)
#     instance_id = instance_dir.name
#     result_path = instance_dir / 'test_results.json'
#     try:
#         if result_path.exists() and not force:
#             inst_logger.info("Skipping %s because results exist and --force not set.", instance_id)
#             return {"skipped": True, "instance_id": instance_id}
#         inst_logger.info("Starting work for %s", instance_id)
#         build_ok, tag, build_err = build_image(instance_dir, inst_logger)
#         if not build_ok:
#             tr = TestResult(instance_id=instance_id, image_tag=tag or "", build_ok=False, test_invoked=False, counts={}, return_code=0, error=build_err)
#             result_path.write_text(json.dumps(tr.to_dict(), indent=2))
#             inst_logger.error("Build failed for %s: %s", instance_id, build_err)
#             # remove_image(tag, inst_logger)
#             prune_all_docker(inst_logger)

#             return {"result": tr.to_dict(), "image_tag": tag or ""}
#         test_ok, counts, rc, test_err, raw_output = run_tests_in_container(tag, instance_dir, inst_logger)
#         tr = TestResult(instance_id=instance_id, image_tag=tag, build_ok=True, test_invoked=True, counts=counts, return_code=rc, error=test_err)
#         result_path.write_text(json.dumps(tr.to_dict(), indent=2))
#         (instance_dir / 'test_output.txt').write_text(raw_output)
#         inst_logger.info("Completed tests for %s: %s", instance_id, counts)
#         # Always remove image after tests
#         # remove_image(tag, inst_logger)
#         prune_all_docker(inst_logger)

#         return {"result": tr.to_dict(), "image_tag": tag}
#     except Exception as exc:
#         tb = traceback.format_exc()
#         inst_logger.exception("Unhandled exception in worker for %s: %s", instance_id, tb)
#         tr = TestResult(instance_id=instance_id, image_tag="", build_ok=False, test_invoked=False, counts={}, return_code=255, error=f"exception: {exc}")
#         try:
#             result_path.write_text(json.dumps(tr.to_dict(), indent=2))
#         except Exception:
#             inst_logger.error("Failed to write test_results.json for %s", instance_id)
#         return {"result": tr.to_dict(), "image_tag": ""}

# # ---------------- Main flow (patched prune-every) -----------------

# def main():
#     parser = argparse.ArgumentParser(description="Build and test each work instance inside Docker (parallelized, immediate image removal).")
#     parser.add_argument('--instances-dir', default='work_instances')
#     parser.add_argument('--limit', type=int)
#     parser.add_argument('--force', action='store_true')
#     parser.add_argument('--summary-file', default='docker_test_summary.json')
#     parser.add_argument('--jobs', type=int, default=1)
#     args = parser.parse_args()
#     instances_root = Path(args.instances_dir)
#     if not instances_root.exists():
#         print(f"Instances directory {instances_root} does not exist", file=sys.stderr)
#         sys.exit(1)
#     global_logger = setup_global_logger(Path('docker_test_runs.log'))
#     if not docker_available():
#         global_logger.error("Docker CLI not found in PATH")
#         sys.exit(2)
#     instance_dirs = [d for d in sorted(instances_root.iterdir()) if d.is_dir() and (d / 'Dockerfile').exists()]
#     if args.limit:
#         instance_dirs = instance_dirs[:args.limit]
#     global_logger.info("Found %d instance directories to process (jobs=%d).", len(instance_dirs), args.jobs)
#     results_list: List[Dict[str, Any]] = []
#     with ProcessPoolExecutor(max_workers=args.jobs) as exe:
#         futures = {exe.submit(process_instance_worker, str(d), args.force): d for d in instance_dirs}
#         for fut in as_completed(futures):
#             inst_dir = futures[fut]
#             try:
#                 res = fut.result()
#             except Exception as exc:
#                 global_logger.exception("Worker for %s raised exception: %s", inst_dir.name, exc)
#                 err_tr = TestResult(instance_id=inst_dir.name, image_tag="", build_ok=False, test_invoked=False, counts={}, return_code=255, error=str(exc))
#                 res = {"result": err_tr.to_dict(), "image_tag": ""}
#             results_list.append(res)
#             rid = (res.get("result") or {}).get("instance_id") or res.get("instance_id") or inst_dir.name
#             global_logger.info("Finished instance %s", rid)
#     summary: Dict[str, Any] = {"results": [], "total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
#     for item in results_list:
#         if item.get("skipped"):
#             summary["skipped"] += 1
#             continue
#         res = item.get("result") or item
#         if isinstance(res, dict) and "instance_id" in res:
#             summary["results"].append(res)
#             counts = res.get("counts", {}) or {}
#             passed = counts.get("passed", 0)
#             failed = counts.get("failed", 0)
#             errors = counts.get("errors", 0)
#             summary["total"] += passed + failed + errors
#             summary["passed"] += passed
#             summary["failed"] += failed
#             summary["errors"] += errors
#         else:
#             summary["results"].append({"raw": res})
#     Path(args.summary_file).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
#     global_logger.info("Done. Summary saved to %s. total=%d passed=%d failed=%d errors=%d skipped=%d", args.summary_file, summary["total"], summary["passed"], summary["failed"], summary["errors"], summary["skipped"])    

# if __name__ == '__main__':
#     main()



#!/usr/bin/env python3
"""
Parallelized build-and-test runner for multiple Docker work instances.

Features:
 - Parallel build + test per instance (ProcessPoolExecutor).
 - Per-instance logs & artifacts (test_results.json, test_output.txt, logs/test_run.log).
 - Global progress logger (docker_test_runs.log + stdout).
 - Immediate image removal after each test run (no pruning phase needed).
 - Respects --force to re-run instances with existing results.
 - Concurrency control via --jobs.
 - Robust error handling; failures isolated.
 - Aggregated summary JSON (docker_test_summary.json).
 - Build timeout support: if a docker build takes too long it will be aborted and the
   runner will move on to the next instance.
Usage:
    python scripts/run_docker_tests.py \
            --instances-dir work_instances \
            --limit 50 \
            --jobs 4 \
            --build-timeout 300
"""
from __future__ import annotations

import argparse, json, logging, re, shutil, subprocess, sys, traceback, multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------- Logging helpers -----------------

def setup_global_logger(log_path: Path) -> logging.Logger:
    """
    Global logger (main process). Per-instance logs are created in worker processes.
    """
    logger = logging.getLogger("docker_tests_global")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

def setup_instance_logger(instance_dir: Path) -> logging.Logger:
    """
    Per-instance logger. Safe to call in worker processes.
    """
    logs_dir = instance_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"docker_tests.{instance_dir.name}")
    logger.setLevel(logging.INFO)
    # Reset handlers for idempotency (important for repeated runs within same process)
    logger.handlers = []  # type: ignore
    fh = logging.FileHandler(logs_dir / 'test_run.log')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    # Do not add stream handler here (workers shouldn't spam main stdout)
    return logger

# ---------------- Subprocess helper -----------------

def run(cmd: List[str], cwd: Optional[Path] = None, capture: bool = False, text: bool = True, env: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    """
    Wrapper around subprocess.run. Returns CompletedProcess. Uses str(cwd) if provided.
    Accepts optional timeout (seconds). On timeout returns a CompletedProcess with returncode=124.
    """
    try:
        return subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=capture, text=text, env=env, check=False, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        # Return a CompletedProcess-like object indicating timeout (rc 124)
        stdout = e.stdout if hasattr(e, 'stdout') and e.stdout is not None else ''
        stderr = e.stderr if hasattr(e, 'stderr') and e.stderr is not None else ''
        # include a short description of timeout
        stderr = (stderr + "\n" if stderr else "") + f"TimeoutExpired: command exceeded {timeout} seconds"
        return subprocess.CompletedProcess(args=cmd, returncode=124, stdout=stdout, stderr=stderr)
    except Exception as e:
        # Build a fake CompletedProcess-like object to carry the error info
        cp = subprocess.CompletedProcess(args=cmd, returncode=255, stdout='', stderr=str(e))
        return cp

def docker_available() -> bool:
    return shutil.which("docker") is not None

def sanitize_tag(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', name.lower())

# ---------------- Pytest output parsing -----------------

COUNT_PATTERN = re.compile(r"(\d+)\s+(passed|failed|skipped|xfailed|xpassed|errors?|error)s?\b", re.IGNORECASE)

def parse_pytest_output(output: str) -> Dict[str, int]:
    """
    Parse pytest summary counts from output. Returns map of counts.
    Scans from the end to find the last summary-like line containing 'passed' or 'failed'.
    """
    counts: Dict[str, int] = {}
    for line in reversed(output.splitlines()):
        if ('passed' in line.lower() or 'failed' in line.lower() or 'skipped' in line.lower() or 'error' in line.lower()):
            for m in COUNT_PATTERN.finditer(line):
                n = int(m.group(1))
                key = m.group(2).lower()
                # normalize variations
                if key.startswith('error'):
                    key = 'errors'
                if key.endswith('s') and key not in ('xpassed','xfailed'):
                    key = key.rstrip('s')
                counts[key] = counts.get(key, 0) + n
            if counts:
                break
    return counts

# ---------------- Data structures -----------------

@dataclass
class TestResult:
    instance_id: str
    image_tag: str
    build_ok: bool
    test_invoked: bool
    counts: Dict[str, int]
    return_code: int
    error: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "image_tag": self.image_tag,
            "build_ok": self.build_ok,
            "test_invoked": self.test_invoked,
            "counts": self.counts,
            "return_code": self.return_code,
            "error": self.error,
        }

# ---------------- Core operations (usable in workers) -----------------

def build_image(instance_dir: Path, logger: logging.Logger, timeout: Optional[int] = None) -> Tuple[bool, str, Optional[str]]:
    """
    Build docker image for given instance dir. Returns (success, tag, stderr_or_error).
    If `timeout` is provided (seconds), the docker build will be aborted after that time.
    """
    dockerfile = instance_dir / 'Dockerfile'
    instance_id = instance_dir.name
    if not dockerfile.exists():
        return False, '', 'Dockerfile missing'
    tag = f"sandbox_{sanitize_tag(instance_id)}:latest"
    cmd = ["docker", "build", "-t", tag, str(instance_dir)]
    logger.info("Building image %s from %s (timeout=%s)", tag, instance_dir, timeout)
    proc = run(cmd, capture=True, timeout=timeout)
    if proc.returncode != 0:
        # Detect timeout case
        if proc.returncode == 124:
            # TimeoutExpired
            logger.error("Build timed out for %s after %s seconds. stderr_sample: %s", instance_id, timeout, (proc.stderr or '')[-1000:])
            return False, tag, f"build timeout after {timeout}s"
        # include some stderr/ stdout context
        stderr_sample = (proc.stderr or '')[-2000:]
        logger.error("Build failed for %s (rc=%s). stderr_sample: %s", instance_id, proc.returncode, stderr_sample)
        return False, tag, proc.stderr or f"returncode={proc.returncode}"
    logger.info("Built image %s", tag)
    return True, tag, None

def run_tests_individually(tag: str, tests: List[str], logger: logging.Logger) -> Tuple[Dict[str, int], str]:
    """
    Run each pytest node individually inside the container to isolate invalid node ids.
    Returns (counts, concatenated_output).
    """
    counts = {"passed": 0, "failed": 0, "errors": 0, "invalid": 0}
    outputs: List[str] = []
    for t in tests:
        cmd = ["docker", "run", "--rm", tag, "pytest", "-q", t]
        proc = run(cmd, capture=True)
        single_out = (proc.stdout or '') + '\n' + (proc.stderr or '')
        outputs.append(f"===== {t} (rc={proc.returncode}) =====\n{single_out}\n")
        rc = proc.returncode
        if rc == 0:
            counts["passed"] += 1
        elif rc == 1:
            counts["failed"] += 1
        elif rc in (4, 5):  # usage error or no tests collected
            counts["invalid"] += 1
        else:
            counts["errors"] += 1
    logger.info("Per-test execution summary: %s", counts)
    return counts, '\n'.join(outputs)

def run_tests_in_container(tag: str, instance_dir: Path, logger: logging.Logger) -> Tuple[bool, Dict[str, int], int, Optional[str], str]:
    """
    Run tests inside a container created from tag, reading test_cases_pass_to_pass.json from the instance directory (which is baked into the image).
    Returns (ok_flag, counts, return_code, error_msg_or_none, raw_output).
    """
    tests_file = instance_dir / 'test_cases_pass_to_pass.json'
    if not tests_file.exists():
        return False, {}, 0, 'test_cases_pass_to_pass.json missing', ''
    try:
        tests = json.loads(tests_file.read_text())
    except Exception as e:
        return False, {}, 0, f'cannot read tests list: {e}', ''
    if not isinstance(tests, list):
        return False, {}, 0, 'tests file not a list', ''
    if not tests:
        return True, {"passed": 0}, 0, None, 'NO_TESTS'

    # Inline python harness to ensure consistent summary formatting (same as original)
    harness = ("import json,pytest,sys;"
               "tests=json.load(open('test_cases_pass_to_pass.json'));"
               "rc=pytest.main(tests);sys.exit(rc)")
    cmd = ["docker", "run", "--rm", tag, "python", "-c", harness]
    proc = run(cmd, capture=True)
    output = (proc.stdout or '') + '\n' + (proc.stderr or '')
    counts = parse_pytest_output(output)

    # If pytest returns usage error, fall back to running individually
    if proc.returncode == 4:
        logger.warning("Batch test invocation returned usage error (rc=4). Falling back to per-test execution.")
        indiv_counts, indiv_output = run_tests_individually(tag, tests, logger)
        output += '\n' + indiv_output
        return True, indiv_counts, proc.returncode, None, output

    if proc.returncode != 0 and not counts:
        logger.error("Tests failed without parsable summary (rc=%s). Raw output saved.", proc.returncode)
        return False, counts, proc.returncode, 'no summary parsed', output

    logger.info("Test summary %s (rc=%s)", counts, proc.returncode)
    return True, counts, proc.returncode, None, output

# ---------------- Image removal helpers (modified) -----------------

def remove_image(tag: str, logger: logging.Logger):
    if not tag:
        return
    proc = run(["docker", "rmi", "-f", tag], capture=True)
    if proc.returncode != 0:
        logger.warning("Could not remove image %s (rc=%s): %s", tag, proc.returncode, (proc.stderr or '')[-300:])
    else:
        logger.info("Removed image %s", tag)

def prune_all_docker(logger: logging.Logger):
    cmds = [
        ["docker", "container", "prune", "-f"],
        ["docker", "image", "prune", "-a", "-f"],
        ["docker", "builder", "prune", "-a", "-f"],
        ["docker", "volume", "prune", "-f"],
        ["docker", "network", "prune", "-f"],
        ["docker", "system", "prune", "-a", "-f"],
    ]
    for cmd in cmds:
        proc = run(cmd, capture=True)
        if proc.returncode != 0:
            logger.warning("Prune command failed (%s): %s", " ".join(cmd), proc.stderr)
        else:
            logger.info("Ran: %s", " ".join(cmd))

# ---------------- Worker function (patched) -----------------

def process_instance_worker(instance_dir_str: str, force: bool, build_timeout: Optional[int]) -> Dict[str, Any]:
    instance_dir = Path(instance_dir_str)
    inst_logger = setup_instance_logger(instance_dir)
    instance_id = instance_dir.name
    result_path = instance_dir / 'test_results.json'
    try:
        if result_path.exists() and not force:
            inst_logger.info("Skipping %s because results exist and --force not set.", instance_id)
            return {"skipped": True, "instance_id": instance_id}
        inst_logger.info("Starting work for %s", instance_id)
        build_ok, tag, build_err = build_image(instance_dir, inst_logger, timeout=build_timeout)
        if not build_ok:
            tr = TestResult(instance_id=instance_id, image_tag=tag or "", build_ok=False, test_invoked=False, counts={}, return_code=0, error=build_err)
            result_path.write_text(json.dumps(tr.to_dict(), indent=2))
            inst_logger.error("Build failed for %s: %s", instance_id, build_err)
            # Clean up any dangling state after a failed/timeout build
            prune_all_docker(inst_logger)

            return {"result": tr.to_dict(), "image_tag": tag or ""}
        test_ok, counts, rc, test_err, raw_output = run_tests_in_container(tag, instance_dir, inst_logger)
        tr = TestResult(instance_id=instance_id, image_tag=tag, build_ok=True, test_invoked=True, counts=counts, return_code=rc, error=test_err)
        result_path.write_text(json.dumps(tr.to_dict(), indent=2))
        (instance_dir / 'test_output.txt').write_text(raw_output)
        inst_logger.info("Completed tests for %s: %s", instance_id, counts)
        # Always try to clean up after tests
        prune_all_docker(inst_logger)

        return {"result": tr.to_dict(), "image_tag": tag}
    except Exception as exc:
        tb = traceback.format_exc()
        inst_logger.exception("Unhandled exception in worker for %s: %s", instance_id, tb)
        tr = TestResult(instance_id=instance_id, image_tag="", build_ok=False, test_invoked=False, counts={}, return_code=255, error=f"exception: {exc}")
        try:
            result_path.write_text(json.dumps(tr.to_dict(), indent=2))
        except Exception:
            inst_logger.error("Failed to write test_results.json for %s", instance_id)
        return {"result": tr.to_dict(), "image_tag": ""}

# ---------------- Main flow (patched prune-every) -----------------

def main():
    parser = argparse.ArgumentParser(description="Build and test each work instance inside Docker (parallelized, immediate image removal).")
    parser.add_argument('--instances-dir', default='work_instances')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--summary-file', default='docker_test_summary.json')
    parser.add_argument('--jobs', type=int, default=1)
    # New: build timeout in seconds (default 300 = 5 minutes)
    parser.add_argument('--build-timeout', type=int, default=300, help='Timeout in seconds for docker build (default 300)')
    args = parser.parse_args()
    instances_root = Path(args.instances_dir)
    if not instances_root.exists():
        print(f"Instances directory {instances_root} does not exist", file=sys.stderr)
        sys.exit(1)
    global_logger = setup_global_logger(Path('docker_test_runs.log'))
    if not docker_available():
        global_logger.error("Docker CLI not found in PATH")
        sys.exit(2)
    instance_dirs = [d for d in sorted(instances_root.iterdir()) if d.is_dir() and (d / 'Dockerfile').exists()]
    if args.limit:
        instance_dirs = instance_dirs[:args.limit]
    global_logger.info("Found %d instance directories to process (jobs=%d).", len(instance_dirs), args.jobs)
    results_list: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as exe:
        # pass build_timeout into each worker
        futures = {exe.submit(process_instance_worker, str(d), args.force, args.build_timeout): d for d in instance_dirs}
        for fut in as_completed(futures):
            inst_dir = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                global_logger.exception("Worker for %s raised exception: %s", inst_dir.name, exc)
                err_tr = TestResult(instance_id=inst_dir.name, image_tag="", build_ok=False, test_invoked=False, counts={}, return_code=255, error=str(exc))
                res = {"result": err_tr.to_dict(), "image_tag": ""}
            results_list.append(res)
            rid = (res.get("result") or {}).get("instance_id") or res.get("instance_id") or inst_dir.name
            global_logger.info("Finished instance %s", rid)
    summary: Dict[str, Any] = {"results": [], "total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for item in results_list:
        if item.get("skipped"):
            summary["skipped"] += 1
            continue
        res = item.get("result") or item
        if isinstance(res, dict) and "instance_id" in res:
            summary["results"].append(res)
            counts = res.get("counts", {}) or {}
            passed = counts.get("passed", 0)
            failed = counts.get("failed", 0)
            errors = counts.get("errors", 0)
            summary["total"] += passed + failed + errors
            summary["passed"] += passed
            summary["failed"] += failed
            summary["errors"] += errors
        else:
            summary["results"].append({"raw": res})
    Path(args.summary_file).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    global_logger.info("Done. Summary saved to %s. total=%d passed=%d failed=%d errors=%d skipped=%d", args.summary_file, summary["total"], summary["passed"], summary["failed"], summary["errors"], summary["skipped"])

if __name__ == '__main__':
    main()
