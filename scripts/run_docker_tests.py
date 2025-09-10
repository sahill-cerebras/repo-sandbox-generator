# #!/usr/bin/env python3
# """Build Docker images for each prepared work instance and run pass_to_pass tests.

# Features:
#  - Scans an instances directory (default: work_instances) for per-instance repos (directories that contain a Dockerfile).
#  - Builds a Docker image per instance (tag pattern: sandbox_<instance_id>:latest).
#  - Runs tests listed in test_cases_pass_to_pass.json inside the container using an inline Python harness.
#  - Parses pytest summary to capture counts (passed, failed, skipped, xfailed, xpassed, errors).
#  - Logs detailed output to per-instance log file logs/test_run.log and a global log docker_test_runs.log.
#  - Writes per-instance test_results.json with structured results.
#  - Maintains a cumulative summary written to docker_test_summary.json at end.
#  - Frees disk space: after N successful (or attempted) builds (configurable --prune-every), removes built images so far and resets the build counter window.

# Assumptions:
#  - Each instance directory already contains the source and Dockerfile generated previously.
#  - Dockerfile sets a WORKDIR (we simply rely on the final WORKDIR; no bind mount needed).
#  - The image includes python + pytest (guaranteed by earlier Dockerfile generation logic).
#  - test_cases_pass_to_pass.json contains a JSON list of pytest node ids or file paths.

# Usage:
#   python scripts/run_docker_tests.py \
#       --instances-dir work_instances \
#       --prune-every 10 \
#       --limit 50
# """
# from __future__ import annotations

# import argparse
# import json
# import logging
# import os
# import re
# import shutil
# import subprocess
# import sys
# from dataclasses import dataclass
# from pathlib import Path
# from typing import List, Dict, Optional, Any

# # ---------------- Logging -----------------

# def setup_global_logger(log_path: Path) -> logging.Logger:
#     logger = logging.getLogger("docker_tests")
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
#     logs_dir = instance_dir / 'logs'
#     logs_dir.mkdir(exist_ok=True, parents=True)
#     logger = logging.getLogger(f"docker_tests.{instance_dir.name}")
#     logger.setLevel(logging.INFO)
#     # Reset handlers for idempotency
#     logger.handlers = []  # type: ignore
#     fh = logging.FileHandler(logs_dir / 'test_run.log')
#     fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
#     logger.addHandler(fh)
#     return logger

# # ---------------- Utilities -----------------

# def run(cmd: List[str], cwd: Optional[Path] = None, capture: bool = False, text: bool = True, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
#     return subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=capture, text=text, env=env, check=False)


# def docker_available() -> bool:
#     return shutil.which("docker") is not None


# def sanitize_tag(name: str) -> str:
#     return re.sub(r'[^a-zA-Z0-9_.-]', '_', name.lower())

# SUMMARY_PATTERN = re.compile(r"=+.*?\b(\d+)\s+passed\b.*?$")  # quick presence check
# COUNT_PATTERN = re.compile(r"(\d+)\s+(passed|failed|skipped|xfailed|xpassed|errors?)")

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

# # ---------------- Parsing logic -----------------

# def parse_pytest_output(output: str) -> Dict[str, int]:
#     counts: Dict[str, int] = {}
#     # Look for last summary line(s)
#     for line in reversed(output.splitlines()):
#         if '===' in line and ('passed' in line or 'failed' in line or 'skipped' in line):
#             for m in COUNT_PATTERN.finditer(line):
#                 n = int(m.group(1))
#                 key = m.group(2)
#                 # normalize 'errors' or 'error'
#                 if key.startswith('error'):
#                     key = 'errors'
#                 counts[key] = n
#             if counts:
#                 break
#     return counts

# # ---------------- Core operations -----------------

# from typing import Tuple  # added for explicit tuple annotations

# def build_image(instance_dir: Path, logger: logging.Logger) -> Tuple[bool, str, Optional[str]]:
#     dockerfile = instance_dir / 'Dockerfile'
#     if not dockerfile.exists():
#         return False, '', 'Dockerfile missing'
#     instance_id = instance_dir.name
#     tag = f"sandbox_{sanitize_tag(instance_id)}:latest"
#     cmd = ["docker", "build", "-t", tag, str(instance_dir)]
#     proc = run(cmd, capture=True)
#     if proc.returncode != 0:
#         logger.error("Build failed for %s: %s", instance_id, proc.stderr[-500:])
#         return False, tag, proc.stderr
#     logger.info("Built image %s", tag)
#     return True, tag, None


# def run_tests_in_container(tag: str, instance_dir: Path, logger: logging.Logger) -> Tuple[bool, Dict[str, int], int, Optional[str], str]:
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

#     # Inline python harness to ensure consistent summary formatting
#     harness = ("import json,pytest,sys;"\
#                "tests=json.load(open('test_cases_pass_to_pass.json'));"\
#                "rc=pytest.main(tests);sys.exit(rc)")
#     cmd = ["docker", "run", "--rm", tag, "python", "-c", harness]
#     proc = run(cmd, capture=True)
#     output = (proc.stdout or '') + '\n' + (proc.stderr or '')
#     counts = parse_pytest_output(output)
#     if proc.returncode == 4:  # pytest usage error (likely invalid test node ids). Fallback to per-test execution.
#         logger.warning("Batch test invocation returned usage error (rc=4). Falling back to per-test execution.")
#         indiv_counts, indiv_output = run_tests_individually(tag, tests, logger)
#         # Compose synthetic output
#         output += '\n' + indiv_output
#         return True, indiv_counts, proc.returncode, None, output
#     if proc.returncode != 0 and not counts:
#         logger.error("Tests failed without parsable summary (rc=%s)", proc.returncode)
#         return False, counts, proc.returncode, 'no summary parsed', output
#     logger.info("Test summary %s", counts)
#     return True, counts, proc.returncode, None, output


# def run_tests_individually(tag: str, tests: List[str], logger: logging.Logger) -> Tuple[Dict[str, int], str]:
#     """Run each test id separately to isolate invalid node ids.

#     Classification:
#       exit 0 -> passed
#       exit 1 -> failed
#       exit 4/5 -> invalid (usage error / no tests collected)
#       other non-zero -> errors
#     Returns counts dict and concatenated output.
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


# def prune_images(tags: List[str], global_logger: logging.Logger):
#     if not tags:
#         return
#     cmd = ["docker", "rmi", "-f", *tags]
#     proc = run(cmd, capture=True)
#     if proc.returncode != 0:
#         global_logger.warning("Failed to remove some images: %s", proc.stderr[-300:])
#     else:
#         global_logger.info("Removed images: %s", ','.join(tags))
#     # Optional builder prune (best effort)
#     run(["docker", "builder", "prune", "-f"], capture=True)

# # ---------------- Main flow -----------------

# def main():
#     parser = argparse.ArgumentParser(description="Build and test each work instance inside Docker")
#     parser.add_argument('--instances-dir', default='work_instances', help='Directory containing instance worktrees')
#     parser.add_argument('--limit', type=int, help='Process only first N instances')
#     parser.add_argument('--prune-every', type=int, default=10, help='Prune built images after this many builds to save space')
#     parser.add_argument('--force', action='store_true', help='Rebuild / re-test even if test_results.json exists')
#     parser.add_argument('--summary-file', default='docker_test_summary.json', help='Output summary JSON path')
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

#     global_logger.info("Found %d instance directories to process", len(instance_dirs))

#     built_tags: List[str] = []
#     summary: Dict[str, Any] = {"results": [], "total": 0, "passed": 0, "failed": 0, "errors": 0}

#     for idx, inst_dir in enumerate(instance_dirs, start=1):
#         instance_id = inst_dir.name
#         inst_logger = setup_instance_logger(inst_dir)
#         result_path = inst_dir / 'test_results.json'
#         if result_path.exists() and not args.force:
#             global_logger.info("[%d/%d] Skipping %s (results exist)", idx, len(instance_dirs), instance_id)
#             continue
#         global_logger.info("[%d/%d] Processing %s", idx, len(instance_dirs), instance_id)
#         inst_logger.info("Starting test run for %s", instance_id)

#         build_ok, tag, build_err = build_image(inst_dir, inst_logger)
#         if not build_ok:
#             tr = TestResult(instance_id, tag, False, False, {}, 0, build_err)
#             result_path.write_text(json.dumps(tr.to_dict(), indent=2))
#             summary["results"].append(tr.to_dict())
#             summary["errors"] += 1
#             continue
#         built_tags.append(tag)

#         test_ok, counts, rc, test_err, raw_output = run_tests_in_container(tag, inst_dir, inst_logger)
#         tr = TestResult(instance_id, tag, True, True, counts, rc, test_err)
#         # Update aggregate counters
#         passed = counts.get('passed', 0)
#         failed = counts.get('failed', 0) + counts.get('errors', 0)
#         summary['total'] += passed + failed
#         summary['passed'] += passed
#         summary['failed'] += counts.get('failed', 0)
#         summary['errors'] += counts.get('errors', 0)

#         # Persist per-instance artifacts
#         result_path.write_text(json.dumps(tr.to_dict(), indent=2))
#         (inst_dir / 'test_output.txt').write_text(raw_output)

#         summary['results'].append(tr.to_dict())
#         global_logger.info("Completed %s: %s", instance_id, counts)

#         # Prune images if threshold reached
#         if args.prune_every > 0 and len(built_tags) >= args.prune_every:
#             global_logger.info("Pruning images after %d builds", len(built_tags))
#             prune_images(built_tags, global_logger)
#             built_tags.clear()

#     # Final prune leftover tags
#     if built_tags:
#         global_logger.info("Final pruning of remaining %d images", len(built_tags))
#         prune_images(built_tags, global_logger)

#     # Write summary
#     Path(args.summary_file).write_text(json.dumps(summary, indent=2))
#     global_logger.info("Done. Total tests=%d passed=%d failed=%d errors=%d -> %s", summary['total'], summary['passed'], summary['failed'], summary['errors'], args.summary_file)


# if __name__ == '__main__':  # pragma: no cover
#     main()


#!/usr/bin/env python3
"""
Parallelized build-and-test runner for multiple Docker work instances.

Features (compared to your original sequential script):
 - Runs independent instance jobs (build + tests) in parallel using ProcessPoolExecutor.
 - Each worker creates its own per-instance logger and writes per-instance artifacts (test_results.json, test_output.txt, logs/test_run.log).
 - Global logger records high-level progress only (to docker_test_runs.log and stdout).
 - Pruning is performed only after all parallel jobs complete, avoiding races where an image is pruned while another job needs it.
 - Respects --force to re-run instances with existing test_results.json.
 - Limits concurrency with --jobs.
 - Robust error handling so a failing instance does not crash the manager.
 - Produces docker_test_summary.json with aggregated results.

Usage:
  python scripts/run_docker_tests_parallel.py \
      --instances-dir work_instances \
      --prune-every 10 \
      --limit 50 \
      --jobs 4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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

def run(cmd: List[str], cwd: Optional[Path] = None, capture: bool = False, text: bool = True, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    """
    Wrapper around subprocess.run. Returns CompletedProcess. Uses str(cwd) if provided.
    """
    try:
        return subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=capture, text=text, env=env, check=False)
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

def build_image(instance_dir: Path, logger: logging.Logger) -> Tuple[bool, str, Optional[str]]:
    """
    Build docker image for given instance dir. Returns (success, tag, stderr_or_error).
    """
    dockerfile = instance_dir / 'Dockerfile'
    instance_id = instance_dir.name
    if not dockerfile.exists():
        return False, '', 'Dockerfile missing'
    tag = f"sandbox_{sanitize_tag(instance_id)}:latest"
    cmd = ["docker", "build", "-t", tag, str(instance_dir)]
    logger.info("Building image %s from %s", tag, instance_dir)
    proc = run(cmd, capture=True)
    if proc.returncode != 0:
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

# ---------------- Worker function -----------------

def process_instance_worker(instance_dir_str: str, force: bool) -> Dict[str, Any]:
    """
    Worker entrypoint executed in a separate process.
    Returns a dict serializable to JSON with test result info and image tag (if any).
    Writes per-instance files (test_results.json, test_output.txt, logs/test_run.log) itself.
    """
    instance_dir = Path(instance_dir_str)
    inst_logger = setup_instance_logger(instance_dir)
    instance_id = instance_dir.name
    result_path = instance_dir / 'test_results.json'

    try:
        if result_path.exists() and not force:
            inst_logger.info("Skipping %s because results exist and --force not set.", instance_id)
            return {"skipped": True, "instance_id": instance_id}

        inst_logger.info("Starting work for %s", instance_id)

        build_ok, tag, build_err = build_image(instance_dir, inst_logger)
        if not build_ok:
            tr = TestResult(instance_id=instance_id, image_tag=tag or "", build_ok=False, test_invoked=False, counts={}, return_code=0, error=build_err)
            result_path.write_text(json.dumps(tr.to_dict(), indent=2))
            inst_logger.error("Build failed for %s: %s", instance_id, build_err)
            return {"result": tr.to_dict(), "image_tag": tag or ""}

        # Run tests
        test_ok, counts, rc, test_err, raw_output = run_tests_in_container(tag, instance_dir, inst_logger)
        tr = TestResult(instance_id=instance_id, image_tag=tag, build_ok=True, test_invoked=True, counts=counts, return_code=rc, error=test_err)
        # Persist results
        result_path.write_text(json.dumps(tr.to_dict(), indent=2))
        (instance_dir / 'test_output.txt').write_text(raw_output)
        inst_logger.info("Completed tests for %s: %s", instance_id, counts)
        return {"result": tr.to_dict(), "image_tag": tag}
    except Exception as exc:
        # Catch-all to ensure worker doesn't crash without reporting
        tb = traceback.format_exc()
        inst_logger.exception("Unhandled exception in worker for %s: %s", instance_id, tb)
        tr = TestResult(instance_id=instance_id, image_tag="", build_ok=False, test_invoked=False, counts={}, return_code=255, error=f"exception: {exc}")
        try:
            result_path.write_text(json.dumps(tr.to_dict(), indent=2))
        except Exception:
            inst_logger.error("Failed to write test_results.json for %s", instance_id)
        return {"result": tr.to_dict(), "image_tag": ""}

# ---------------- Prune helper -----------------

def prune_images(tags: List[str], global_logger: logging.Logger):
    if not tags:
        return
    cmd = ["docker", "rmi", "-f", *tags]
    proc = run(cmd, capture=True)
    if proc.returncode != 0:
        global_logger.warning("Failed to remove some images. stderr sample: %s", (proc.stderr or '')[-1000:])
    else:
        global_logger.info("Removed images: %s", ','.join(tags))
    # best-effort builder prune
    run(["docker", "builder", "prune", "-f"], capture=True)

# ---------------- Main flow -----------------

def main():
    parser = argparse.ArgumentParser(description="Build and test each work instance inside Docker (parallelized).")
    parser.add_argument('--instances-dir', default='work_instances', help='Directory containing instance worktrees')
    parser.add_argument('--limit', type=int, help='Process only first N instances')
    parser.add_argument('--prune-every', type=int, default=10, help='Prune built images after this many builds to save space (prune happens after all jobs)')
    parser.add_argument('--force', action='store_true', help='Rebuild / re-test even if test_results.json exists')
    parser.add_argument('--summary-file', default='docker_test_summary.json', help='Output summary JSON path')
    parser.add_argument('--jobs', type=int, default=max(1, (multiprocessing.cpu_count() or 4)//2), help='Number of parallel worker processes to run')
    args = parser.parse_args()

    instances_root = Path(args.instances_dir)
    if not instances_root.exists():
        print(f"Instances directory {instances_root} does not exist", file=sys.stderr)
        sys.exit(1)

    global_logger = setup_global_logger(Path('docker_test_runs.log'))

    if not docker_available():
        global_logger.error("Docker CLI not found in PATH")
        sys.exit(2)

    # Discover instance directories (must contain Dockerfile)
    instance_dirs = [d for d in sorted(instances_root.iterdir()) if d.is_dir() and (d / 'Dockerfile').exists()]
    if args.limit:
        instance_dirs = instance_dirs[:args.limit]

    global_logger.info("Found %d instance directories to process (jobs=%d).", len(instance_dirs), args.jobs)

    # Prepare to run workers
    results_list: List[Dict[str, Any]] = []
    built_tags_all: List[str] = []

    # Submit work to process pool
    with ProcessPoolExecutor(max_workers=args.jobs) as exe:
        futures = {exe.submit(process_instance_worker, str(d), args.force): d for d in instance_dirs}
        for fut in as_completed(futures):
            inst_dir = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                # Shouldn't happen because worker catches exceptions, but be defensive
                global_logger.exception("Worker for %s raised exception: %s", inst_dir.name, exc)
                # create an error result
                err_tr = TestResult(instance_id=inst_dir.name, image_tag="", build_ok=False, test_invoked=False, counts={}, return_code=255, error=str(exc))
                res = {"result": err_tr.to_dict(), "image_tag": ""}
            # Aggregate
            results_list.append(res)
            # collect tag if present
            tag = res.get("image_tag") or (res.get("result") or {}).get("image_tag", "")
            if tag:
                built_tags_all.append(tag)
            # Log high-level progress
            rid = (res.get("result") or {}).get("instance_id") or res.get("instance_id") or inst_dir.name
            global_logger.info("Finished instance %s (image=%s)", rid, tag or "<none>")

    # # After all workers complete, optionally prune images in batches of prune_every
    # if args.prune_every and args.prune_every > 0:
    #     # prune in windows to limit command length
    #     global_logger.info("Pruning built images in windows of %d (total tags=%d)", args.prune_every, len(built_tags_all))
    #     for i in range(0, len(built_tags_all), args.prune_every):
    #         batch = built_tags_all[i:i+args.prune_every]
    #         prune_images(batch, global_logger)

    # Build final summary
    summary: Dict[str, Any] = {"results": [], "total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for item in results_list:
        if item.get("skipped"):
            summary["skipped"] += 1
            continue
        res = item.get("result") or item
        # if res is a TestResult-like dict already, use it; else, keep item
        if isinstance(res, dict) and "instance_id" in res:
            summary["results"].append(res)
            counts = res.get("counts", {}) or {}
            passed = counts.get("passed", 0)
            failed = counts.get("failed", 0)
            errors = counts.get("errors", 0)
            # treat errors as failures in 'total' tallies (you can adjust)
            summary["total"] += passed + failed + errors
            summary["passed"] += passed
            summary["failed"] += failed
            summary["errors"] += errors
        else:
            # unexpected shape
            summary["results"].append({"raw": res})

    # Persist summary
    Path(args.summary_file).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    global_logger.info("Done. Summary saved to %s. total=%d passed=%d failed=%d errors=%d skipped=%d",
                       args.summary_file, summary["total"], summary["passed"], summary["failed"], summary["errors"], summary["skipped"])


if __name__ == '__main__':
    main()
