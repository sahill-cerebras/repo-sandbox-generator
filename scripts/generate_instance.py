#!/usr/bin/env python3
"""Generate Docker configuration and test case files for a SWE-bench dataset instance.

Workflow (single instance focus):
 1. Read parquet dataset (fastparquet or pyarrow) and select a row by index or instance_id.
 2. Clone the GitHub repo once into a cache directory (if not already cloned).
 3. Create a git worktree (preferred) or fallback copy at the target commit (environment_setup_commit or base_commit).
 4. Run internal RepositoryAnalyzer + DockerfileGenerator to produce Dockerfile, .dockerignore, analysis.json in the checked out worktree.
 5. Write test_cases_pass_to_pass.json and test_cases_fail_to_pass.json with lists from dataset (PASS_TO_PASS / FAIL_TO_PASS columns).
 6. Log all actions to logs/<instance_id>/process.log.

Assumptions:
 - This script is executed from repository root (so that 'src' is importable).
 - User installed project editable: `pip install -e .` and has pandas + fastparquet.

Example:
  python scripts/generate_instance.py \
      --dataset SWE-bench/data/dev-00000-of-00001.parquet \
      --index 0 \
      --commit-column environment_setup_commit \
      --cache-dir .cache/repos \
      --output-dir work_instances \
      --logs-dir logs
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# Ensure we can import the package modules
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

try:
    import pandas as pd  # type: ignore
except ImportError as e:  # pragma: no cover
    print("pandas is required. Install with: pip install pandas fastparquet", file=sys.stderr)
    raise

from repo_sandbox.analyzer.repo_analyzer import RepositoryAnalyzer  # type: ignore
from repo_sandbox.generators.dockerfile_generator import DockerfileGenerator  # type: ignore


@dataclass
class InstanceData:
    repo_full: str
    instance_id: str
    commit: str
    pass_to_pass: List[str]
    fail_to_pass: List[str]
    raw_row: Dict[str, Any]


def load_instance(df, index: Optional[int], instance_id: Optional[str], commit_column: str) -> InstanceData:
    if index is None and instance_id is None:
        raise ValueError("Provide either --index or --instance-id")
    if instance_id is not None:
        mask = df['instance_id'] == instance_id
        if not mask.any():
            raise ValueError(f"instance_id {instance_id} not found in dataset")
        row = df[mask].iloc[0]
    else:
        row = df.iloc[index]
    repo_full = row['repo']  # e.g. 'sqlfluff/sqlfluff'
    commit = row[commit_column]
    p2p = _parse_list(row.get('PASS_TO_PASS'))
    f2p = _parse_list(row.get('FAIL_TO_PASS'))
    return construct_instance_from_row(row, commit_column)


def construct_instance_from_row(row, commit_column: str) -> InstanceData:
    # Support both pandas Series and plain dict rows
    if hasattr(row, 'to_dict'):
        raw = row.to_dict()
    else:
        # Already a dict (from itertuples()._asdict())
        raw = dict(row)
    repo_full = raw['repo']
    commit = raw[commit_column]
    p2p = _parse_list(raw.get('PASS_TO_PASS'))
    f2p = _parse_list(raw.get('FAIL_TO_PASS'))
    return InstanceData(
        repo_full=repo_full,
        instance_id=raw['instance_id'],
        commit=commit,
        pass_to_pass=p2p,
        fail_to_pass=f2p,
        raw_row=raw,
    )


def _parse_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            data = json.loads(value)
            if isinstance(data, list):
                return [str(v) for v in data]
        except Exception:
            # Fallback: comma separated
            return [v.strip() for v in value.split(',') if v.strip()]
    return []


_CLONE_LOCKS: Dict[str, threading.Lock] = {}
_CLONE_LOCKS_GLOBAL = threading.Lock()
_FETCH_SEMAPHORE: threading.Semaphore = threading.Semaphore(8)  # resized after parsing args


def _get_clone_lock(repo_full: str) -> threading.Lock:
    with _CLONE_LOCKS_GLOBAL:
        lock = _CLONE_LOCKS.get(repo_full)
        if lock is None:
            lock = threading.Lock()
            _CLONE_LOCKS[repo_full] = lock
        return lock


def ensure_repo_clone(repo_full: str, cache_dir: Path, logger: logging.Logger) -> Path:
    """Clone repo if absent. Returns path to cached working clone.

    Uses absolute paths to avoid duplication like .cache/repos/.cache/repos/...
    """
    cache_dir = cache_dir.resolve()
    owner, name = repo_full.split('/', 1)
    target = (cache_dir / f"{owner}__{name}").resolve()
    # Thread-safe: avoid simultaneous clone of same repo
    lock = _get_clone_lock(repo_full)
    with lock:
        if target.exists():
            logger.info("Using existing clone: %s", target)
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/{repo_full}.git"
        logger.info("Cloning %s -> %s", url, target)
        # Clone without changing cwd so destination path is interpreted correctly
        run(["git", "clone", "--quiet", url, str(target)], logger, cwd=None)
        return target


def create_worktree_or_copy(repo_dir: Path, commit: str, dest_dir: Path, logger: logging.Logger):
    """Create a git worktree at absolute dest_dir (or fallback copy).

    Fixes earlier issue where a relative dest path caused worktree creation
    inside the cached clone (repo_dir/relative/...), leaving expected path empty.
    """
    dest_dir_abs = dest_dir.resolve()
    if dest_dir_abs.exists():
        logger.info("Destination %s already exists; removing for fresh checkout", dest_dir_abs)
        shutil.rmtree(dest_dir_abs)
    dest_dir_abs.parent.mkdir(parents=True, exist_ok=True)
    # Ensure commit is present locally (retry fetch if not)
    if not git_has_commit(repo_dir, commit):
        fetched = attempt_fetch_commit(repo_dir, commit, logger)
        if not fetched and not git_has_commit(repo_dir, commit):
            raise RuntimeError(f"Unable to fetch commit {commit}")
    # Try worktree first with absolute path
    rc = run_allow_fail(["git", "worktree", "add", "--detach", str(dest_dir_abs), commit], cwd=repo_dir)
    if rc != 0:
        logger.warning("git worktree failed (code %s). Falling back to copy & checkout", rc)
        shutil.copytree(repo_dir, dest_dir_abs, ignore=shutil.ignore_patterns('.git', '.gitworktrees'))
        run(["git", "clone", "--quiet", str(repo_dir), str(dest_dir_abs / '.git-temp')], logger)
        tmp_git = dest_dir_abs / '.git-temp' / '.git'
        if tmp_git.exists():
            shutil.move(str(tmp_git), str(dest_dir_abs / '.git'))
            shutil.rmtree(dest_dir_abs / '.git-temp', ignore_errors=True)
        run(["git", "checkout", commit], logger, cwd=dest_dir_abs)
    else:
        # Fallback: if absolute directory still missing, maybe worktree created relative to repo_dir
        if not dest_dir_abs.exists():
            possible = (repo_dir / dest_dir)
            if possible.exists():
                logger.warning("Worktree materialized at %s; moving to %s", possible, dest_dir_abs)
                dest_dir_abs.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(possible), str(dest_dir_abs))
        logger.info("Created worktree at %s for commit %s", dest_dir_abs, commit)
    # Return final path (even though function signature doesn't use return internally)
    return dest_dir_abs


def generate_docker_assets(repo_path: Path, logger: logging.Logger, include_tests: bool = True):
    analyzer = RepositoryAnalyzer()
    analysis = analyzer.analyze_repository(repo_path)
    if include_tests:
        # ensure include flag semantics; analysis already includes test_config
        pass
    gen = DockerfileGenerator()
    dockerfile_content = gen.generate_dockerfile(analysis, template='auto', include_tests=include_tests)
    dockerignore_content = gen.generate_dockerignore(analysis)
    # Write files at repo root
    (repo_path / 'Dockerfile').write_text(dockerfile_content)
    (repo_path / '.dockerignore').write_text(dockerignore_content)
    (repo_path / 'analysis.json').write_text(json.dumps(analysis, indent=2))
    logger.info("Wrote Dockerfile, .dockerignore, analysis.json")


def write_test_case_files(repo_path: Path, instance: InstanceData, logger: logging.Logger):
    (repo_path / 'test_cases_pass_to_pass.json').write_text(json.dumps(instance.pass_to_pass, indent=2))
    (repo_path / 'test_cases_fail_to_pass.json').write_text(json.dumps(instance.fail_to_pass, indent=2))
    logger.info("Wrote test case files: pass_to_pass=%d fail_to_pass=%d", len(instance.pass_to_pass), len(instance.fail_to_pass))


def run(cmd: List[str], logger: logging.Logger, cwd: Optional[Path] = None):
    logger.debug("RUN: %s (cwd=%s)", ' '.join(cmd), cwd or os.getcwd())
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def run_allow_fail(cmd: List[str], cwd: Optional[Path] = None) -> int:
    try:
        subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)
        return 0
    except subprocess.CalledProcessError as e:  # pragma: no cover
        return e.returncode


def setup_logger(log_dir: Path, instance_id: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"instance.{instance_id}")
    logger.setLevel(logging.INFO)
    # Clear handlers if re-run
    logger.handlers = []  # type: ignore
    fh = logging.FileHandler(log_dir / 'process.log')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _process_instance_batch(row_dict: Dict[str, Any], ordinal: int, total: int, args, cache_dir: Path) -> Tuple[str, bool, Optional[str]]:
    """Process one instance in batch mode. Returns (instance_id, success, error)."""
    try:
        instance = construct_instance_from_row(row_dict, args.commit_column)
    except Exception as e:  # malformed row
        return (row_dict.get('instance_id', f'row{ordinal}'), False, f'row-parse: {e}')
    inst_dir = Path(args.output_dir) / instance.instance_id
    if inst_dir.exists() and not args.force:
        if args.skip_existing:
            # Treat skip as success (not a failure scenario)
            return (instance.instance_id, True, None)
        else:
            shutil.rmtree(inst_dir)
    logs_base = Path(args.logs_dir) / instance.instance_id
    logger = setup_logger(logs_base, instance.instance_id)
    logger.info("(%d/%d) Instance %s repo=%s commit=%s", ordinal, total, instance.instance_id, instance.repo_full, instance.commit)
    try:
        repo_cache = ensure_repo_clone(instance.repo_full, cache_dir, logger)
        inst_dir_final = create_worktree_or_copy(repo_cache, instance.commit, inst_dir, logger)
        generate_docker_assets(inst_dir_final, logger, include_tests=not args.no_tests)
        write_test_case_files(inst_dir_final, instance, logger)
        (inst_dir_final / 'dataset_row.json').write_text(json.dumps(instance.raw_row, indent=2))
        logger.info("Done instance %s", instance.instance_id)
        return (instance.instance_id, True, None)
    except Exception as e:  # pragma: no cover
        logger.exception("Failed instance %s: %s", instance.instance_id, e)
        return (instance.instance_id, False, str(e))


def _accumulate_summary(summary: Dict[str, Any], result: Tuple[str, bool, Optional[str]]):
    instance_id, success, error = result
    summary["processed"] += 1
    if success:
        summary["succeeded"] += 1
    else:
        summary["failed"] += 1
        summary["failures"].append({"instance_id": instance_id, "error": error})


# --------------------- Git fetch robustness utilities ---------------------

def git_has_commit(repo_dir: Path, commit: str) -> bool:
    return run_allow_fail(["git", "cat-file", "-t", commit], cwd=repo_dir) == 0


def attempt_fetch_commit(repo_dir: Path, commit: str, logger: logging.Logger, attempts: int = 3) -> bool:
    """Fetch a specific commit with retries, limiting concurrent fetches.

    Attempts:
      1: generic --all
      2: targeted shallow fetch of commit
      3+: broaden shallow depth
    """
    backoff_base = 2
    for attempt in range(1, attempts + 1):
        if git_has_commit(repo_dir, commit):
            return True
        with _FETCH_SEMAPHORE:
            if attempt == 1:
                cmd = ["git", "fetch", "--quiet", "--all"]
            elif attempt == 2:
                cmd = ["git", "fetch", "--quiet", "--depth", "1", "origin", commit]
            else:
                depth = 50 * attempt
                cmd = ["git", "fetch", "--quiet", f"--depth={depth}", "origin"]
            rc = run_allow_fail(cmd, cwd=repo_dir)
        if rc == 0 and git_has_commit(repo_dir, commit):
            logger.info("Fetched commit %s (attempt %d)", commit, attempt)
            return True
        logger.warning("Fetch attempt %d failed for commit %s (rc=%s)", attempt, commit, rc)
        if attempt < attempts:
            sleep_for = backoff_base ** attempt + (attempt * 0.1)
            time.sleep(sleep_for)
    return git_has_commit(repo_dir, commit)

def main():
    parser = argparse.ArgumentParser(description="Generate Docker assets for a SWE-bench dataset instance")
    parser.add_argument('--dataset', required=True, help='Path to parquet dataset file')
    parser.add_argument('--index', type=int, help='Row index to process')
    parser.add_argument('--instance-id', help='Explicit instance_id to process (overrides --index)')
    parser.add_argument('--commit-column', default='base_commit', choices=['environment_setup_commit', 'base_commit'], help='Which commit column to checkout')
    parser.add_argument('--cache-dir', default='.cache/repos', help='Directory to cache cloned repos')
    parser.add_argument('--output-dir', default='work_instances', help='Parent directory for per-instance worktrees')
    parser.add_argument('--logs-dir', default='logs', help='Directory for per-instance log folders')
    parser.add_argument('--no-tests', action='store_true', help='Do not include test setup in Dockerfile generation')
    parser.add_argument('--all', action='store_true', help='Process all rows in the dataset')
    parser.add_argument('--limit', type=int, help='With --all, process only first N rows')
    parser.add_argument('--force', action='store_true', help='Force regeneration even if instance output exists')
    parser.add_argument('--skip-existing', action='store_true', help='Skip instances whose output dir already exists (ignored if --force)')
    parser.add_argument('--workers', type=int, default=1, help='Parallel workers for --all (use 0 or negative for auto = CPU count)')
    parser.add_argument('--fetch-parallelism', type=int, default=8, help='Max concurrent git fetch ops (reduce if hitting network limits)')
    args = parser.parse_args()

    # Adjust global fetch semaphore
    global _FETCH_SEMAPHORE
    _FETCH_SEMAPHORE = threading.Semaphore(args.fetch_parallelism if args.fetch_parallelism > 0 else 4)

    df = pd.read_parquet(args.dataset)
    if args.all:
        total_rows = len(df)
        if args.limit:
            total_rows = min(total_rows, args.limit)
        rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(df.itertuples(index=False), start=0):
            if args.limit and idx >= args.limit:
                break
            rows.append(row._asdict())
        cache_dir = Path(args.cache_dir).resolve()
        output_parent = Path(args.output_dir)
        output_parent.mkdir(parents=True, exist_ok=True)
        workers = args.workers
        if workers is None or workers <= 0:
            workers = os.cpu_count() or 1
        summary = {"processed": 0, "succeeded": 0, "failed": 0, "failures": []}

        if workers == 1:
            # Fallback to sequential (reuse prior logic pattern)
            for i, row_dict in enumerate(rows, start=1):
                result = _process_instance_batch(row_dict, i, len(rows), args, cache_dir)
                _accumulate_summary(summary, result)
        else:
            print(f"Running batch with {workers} workers over {len(rows)} instances")
            futures = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for i, row_dict in enumerate(rows, start=1):
                    futures.append(executor.submit(_process_instance_batch, row_dict, i, len(rows), args, cache_dir))
                for fut in as_completed(futures):
                    result = fut.result()
                    _accumulate_summary(summary, result)
                    if summary["processed"] % 10 == 0:
                        print(f"Progress: processed={summary['processed']} succeeded={summary['succeeded']} failed={summary['failed']}")
        summary_path = Path(args.output_dir) / 'batch_summary.json'
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Batch complete: processed={summary['processed']} succeeded={summary['succeeded']} failed={summary['failed']} -> {summary_path}")
    else:
        instance = load_instance(df, args.index, args.instance_id, args.commit_column)
        logs_base = Path(args.logs_dir) / instance.instance_id
        logger = setup_logger(logs_base, instance.instance_id)
        logger.info("Processing instance %s (repo=%s commit=%s)", instance.instance_id, instance.repo_full, instance.commit)
        cache_dir = Path(args.cache_dir).resolve()
        repo_cache = ensure_repo_clone(instance.repo_full, cache_dir, logger)
        output_parent = Path(args.output_dir)
        worktree_dir = output_parent / instance.instance_id
        if worktree_dir.exists() and args.force:
            shutil.rmtree(worktree_dir)
        elif worktree_dir.exists() and args.skip_existing:
            logger.info("Skipping existing instance directory %s", worktree_dir)
            return
        worktree_dir_final = create_worktree_or_copy(repo_cache, instance.commit, worktree_dir, logger)
        generate_docker_assets(worktree_dir_final, logger, include_tests=not args.no_tests)
        write_test_case_files(worktree_dir_final, instance, logger)
        (worktree_dir_final / 'dataset_row.json').write_text(json.dumps(instance.raw_row, indent=2))
        logger.info("Done instance %s", instance.instance_id)


if __name__ == '__main__':  # pragma: no cover
    main()
