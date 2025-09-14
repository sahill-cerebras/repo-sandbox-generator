#!/usr/bin/env python3
"""
Parallelized build-and-test runner for multiple Docker work instances with intelligent space management.

Features:
 - Parallel build + test per instance (ProcessPoolExecutor)
 - Per-instance logs & artifacts (test_results.json, test_output.txt, logs/test_run.log)
 - Global progress logger (docker_test_runs.log + stdout)
 - Intelligent Docker cleanup: immediate image removal + periodic cache cleanup
 - Build timeout support with proper error handling
 - Respects --force to re-run instances with existing results
 - Concurrency control via --jobs
 - Robust error handling; failures isolated
 - Aggregated summary JSON (docker_test_summary.json)

Usage:
    python scripts/run_docker_tests.py \
            --instances-dir work_instances \
            --limit 50 \
            --jobs 4 \
            --build-timeout 300
"""
from __future__ import annotations

import argparse, json, logging, os, re, shutil, subprocess, sys, time, traceback, threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Global lock for Docker cleanup operations (thread-safe)
DOCKER_CLEANUP_LOCK = threading.Lock()

# ---------------- Logging helpers -----------------

def setup_global_logger(log_path: Path) -> logging.Logger:
    """Global logger (main process). Per-instance logs are created in worker processes."""
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
    """Per-instance logger. Safe to call in worker processes."""
    logs_dir = instance_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"docker_tests.{instance_dir.name}")
    logger.setLevel(logging.INFO)
    # Reset handlers for idempotency
    logger.handlers = []
    fh = logging.FileHandler(logs_dir / 'test_run.log')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    return logger

# ---------------- Subprocess helper -----------------

def run(cmd: List[str], cwd: Optional[Path] = None, capture: bool = False, text: bool = True, 
        env: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    """Enhanced subprocess.run wrapper with timeout handling."""
    try:
        # Merge environment variables
        final_env = os.environ.copy()
        if env:
            final_env.update(env)
        
        return subprocess.run(
            cmd, 
            cwd=str(cwd) if cwd else None, 
            capture_output=capture, 
            text=text, 
            env=final_env, 
            check=False, 
            timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        stdout = getattr(e, 'stdout', '') or ''
        stderr = getattr(e, 'stderr', '') or ''
        stderr += f"\nTimeoutExpired: command exceeded {timeout} seconds"
        return subprocess.CompletedProcess(args=cmd, returncode=124, stdout=stdout, stderr=stderr)
    except Exception as e:
        return subprocess.CompletedProcess(args=cmd, returncode=255, stdout='', stderr=str(e))

def docker_available() -> bool:
    """Check if Docker CLI is available."""
    return shutil.which("docker") is not None

def sanitize_tag(name: str) -> str:
    """Sanitize Docker tag name."""
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', name.lower())

# ---------------- Pytest output parsing -----------------

COUNT_PATTERN = re.compile(r"(\d+)\s+(passed|failed|skipped|xfailed|xpassed|errors?|error)s?\b", re.IGNORECASE)

def parse_pytest_output(output: str) -> Dict[str, int]:
    """Parse pytest summary counts from output."""
    counts: Dict[str, int] = {}
    for line in reversed(output.splitlines()):
        if any(keyword in line.lower() for keyword in ['passed', 'failed', 'skipped', 'error']):
            for m in COUNT_PATTERN.finditer(line):
                n = int(m.group(1))
                key = m.group(2).lower()
                # Normalize variations
                if key.startswith('error'):
                    key = 'errors'
                if key.endswith('s') and key not in ('xpassed', 'xfailed'):
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

# ---------------- Docker operations -----------------

def build_image(instance_dir: Path, logger: logging.Logger, timeout: Optional[int] = None) -> Tuple[bool, str, Optional[str]]:
    """Build Docker image with proper timeout handling."""
    dockerfile = instance_dir / 'Dockerfile'
    instance_id = instance_dir.name
    
    if not dockerfile.exists():
        return False, '', 'Dockerfile missing'
    
    tag = f"sandbox_{sanitize_tag(instance_id)}:latest"
    cmd = ["docker", "build", "-t", tag, str(instance_dir)]
    
    logger.info("Building image %s (timeout=%ss)", tag, timeout)
    
    proc = run(cmd, capture=True, timeout=timeout)
    
    if proc.returncode == 124:  # Timeout
        logger.error("Build timeout for %s after %s seconds", instance_id, timeout)
        return False, tag, f"build timeout after {timeout}s"
    elif proc.returncode != 0:
        stderr_sample = (proc.stderr or '')[-2000:]
        logger.error("Build failed for %s (rc=%s): %s", 
                    instance_id, proc.returncode, stderr_sample)
        return False, tag, proc.stderr or f"returncode={proc.returncode}"
    
    logger.info("Built image %s", tag)
    return True, tag, None

def run_tests_individually(tag: str, tests: List[str], logger: logging.Logger) -> Tuple[Dict[str, int], str]:
    """Run each test individually to isolate invalid test node IDs."""
    counts = {"passed": 0, "failed": 0, "errors": 0, "invalid": 0}
    outputs: List[str] = []
    
    for test in tests:
        cmd = ["docker", "run", "--rm", tag, "pytest", "-q", test]
        proc = run(cmd, capture=True)
        
        output = (proc.stdout or '') + '\n' + (proc.stderr or '')
        outputs.append(f"===== {test} (rc={proc.returncode}) =====\n{output}\n")
        
        if proc.returncode == 0:
            counts["passed"] += 1
        elif proc.returncode == 1:
            counts["failed"] += 1
        elif proc.returncode in (4, 5):  # Usage error or no tests collected
            counts["invalid"] += 1
        else:
            counts["errors"] += 1
    
    logger.info("Individual test execution: %s", counts)
    return counts, '\n'.join(outputs)

def run_tests_in_container(tag: str, instance_dir: Path, logger: logging.Logger) -> Tuple[bool, Dict[str, int], int, Optional[str], str]:
    """Run tests inside container with fallback to individual execution."""
    tests_file = instance_dir / 'test_cases_pass_to_pass.json'
    
    if not tests_file.exists():
        return False, {}, 0, 'test_cases_pass_to_pass.json missing', ''
    
    try:
        tests = json.loads(tests_file.read_text())
        if not isinstance(tests, list):
            return False, {}, 0, 'tests file not a list', ''
        if not tests:
            return True, {"passed": 0}, 0, None, 'NO_TESTS'
    except Exception as e:
        return False, {}, 0, f'cannot read tests list: {e}', ''

    # Use inline Python harness for consistent output
    harness = (
        "import json,pytest,sys;"
        "tests=json.load(open('test_cases_pass_to_pass.json'));"
        "rc=pytest.main(tests);"
        "sys.exit(rc)"
    )
    
    cmd = ["docker", "run", "--rm", tag, "python", "-c", harness]
    proc = run(cmd, capture=True)
    
    output = (proc.stdout or '') + '\n' + (proc.stderr or '')
    counts = parse_pytest_output(output)
    
    # Fallback to individual execution on usage error
    if proc.returncode == 4:
        logger.warning("Batch execution failed (rc=4), falling back to individual tests")
        indiv_counts, indiv_output = run_tests_individually(tag, tests, logger)
        output += '\n--- INDIVIDUAL EXECUTION FALLBACK ---\n' + indiv_output
        return True, indiv_counts, proc.returncode, None, output
    
    if proc.returncode != 0 and not counts:
        logger.error("Tests failed without summary (rc=%s)", proc.returncode)
        return False, counts, proc.returncode, 'no summary parsed', output
    
    logger.info("Test execution complete: %s (rc=%s)", counts, proc.returncode)
    return True, counts, proc.returncode, None, output

# ---------------- Cleanup operations -----------------

def remove_image_safe(tag: str, logger: logging.Logger) -> bool:
    """Safely remove a specific Docker image."""
    if not tag:
        return True
    
    try:
        proc = run(["docker", "rmi", "-f", tag], capture=True, timeout=30)
        if proc.returncode == 0:
            logger.info("Removed image: %s", tag)
            return True
        else:
            logger.warning("Failed to remove image %s", tag)
            return False
    except Exception as e:
        logger.error("Exception removing image %s: %s", tag, e)
        return False

def cleanup_docker_system(logger: logging.Logger) -> bool:
    """Perform Docker system cleanup that's safe for parallel builds."""
    with DOCKER_CLEANUP_LOCK:
        # SAFER cleanup - avoid removing layers that builds might need
        cleanup_commands = [
            (["docker", "container", "prune", "-f"], "stopped containers"),
            (["docker", "image", "prune", "-f"], "dangling images only"),  # No -a flag
            (["docker", "volume", "prune", "-f"], "unused volumes"),
            (["docker", "network", "prune", "-f"], "unused networks"),
            # Skip aggressive builder/system prune during parallel execution
        ]
        
        success_count = 0
        for cmd, desc in cleanup_commands:
            try:
                proc = run(cmd, capture=True, timeout=60)
                if proc.returncode == 0:
                    success_count += 1
                    logger.info("Cleaned %s", desc)
                else:
                    logger.warning("Failed to clean %s", desc)
            except Exception as e:
                logger.error("Exception cleaning %s: %s", desc, e)
        
        return success_count > 0

def cleanup_docker_system_aggressive(logger: logging.Logger) -> bool:
    """Aggressive cleanup - only safe when no builds are running."""
    with DOCKER_CLEANUP_LOCK:
        logger.info("Performing aggressive cleanup")
        cleanup_commands = [
            (["docker", "system", "prune", "-a", "-f"], "everything"),
            (["docker", "builder", "prune", "-a", "-f"], "all build cache"),
        ]
        
        success_count = 0
        for cmd, desc in cleanup_commands:
            try:
                proc = run(cmd, capture=True, timeout=120)
                if proc.returncode == 0:
                    success_count += 1
                    logger.info("Aggressively cleaned %s", desc)
            except Exception as e:
                logger.error("Exception in aggressive cleanup %s: %s", desc, e)
        
        return success_count > 0


def process_batch(batch_dirs: List[Path], args, global_logger) -> List[Dict[str, Any]]:
    """Process a batch of instances in parallel, then clean up."""
    batch_results = []
    batch_size = len(batch_dirs)
    
    global_logger.info("Starting batch of %d instances", batch_size)
    
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        # Submit all jobs in the batch
        future_to_dir = {
            executor.submit(process_instance_worker, str(d), args.force, args.build_timeout): d 
            for d in batch_dirs
        }
        
        # Wait for all jobs in batch to complete
        for future in as_completed(future_to_dir):
            inst_dir = future_to_dir[future]
            
            try:
                result = future.result()
            except Exception as exc:
                global_logger.exception("Worker exception for %s: %s", inst_dir.name, exc)
                error_result = TestResult(
                    instance_id=inst_dir.name,
                    image_tag="",
                    build_ok=False,
                    test_invoked=False,
                    counts={},
                    return_code=255,
                    error=f"worker process exception: {exc}"
                )
                result = {"result": error_result.to_dict(), "image_tag": ""}
            
            batch_results.append(result)
            
            # Log individual completion
            rid = result.get("result", {}).get("instance_id") or result.get("instance_id") or inst_dir.name
            global_logger.info("Completed %s", rid)
    
    global_logger.info("Batch complete - all %d instances finished", batch_size)
    return batch_results

def cleanup_everything(logger: logging.Logger) -> bool:
    """Clean up everything after batch completion."""
    with DOCKER_CLEANUP_LOCK:
        logger.info("Performing complete cleanup after batch")
        
        cleanup_commands = [
            (["docker", "container", "prune", "-f"], "all stopped containers"),
            (["docker", "image", "prune", "-a", "-f"], "all unused images"),
            (["docker", "builder", "prune", "-a", "-f"], "all build cache"),
            (["docker", "volume", "prune", "-f"], "unused volumes"),
            (["docker", "network", "prune", "-f"], "unused networks"),
            (["docker", "system", "prune", "-a", "-f"], "everything else"),
        ]
        
        success_count = 0
        for cmd, desc in cleanup_commands:
            try:
                proc = run(cmd, capture=True, timeout=120)
                if proc.returncode == 0:
                    success_count += 1
                    logger.info("Cleaned %s", desc)
                else:
                    logger.warning("Failed to clean %s", desc)
            except Exception as e:
                logger.error("Exception cleaning %s: %s", desc, e)
        
        return success_count > 0
    
# ---------------- Worker function -----------------

def process_instance_worker(instance_dir_str: str, force: bool, build_timeout: Optional[int]) -> Dict[str, Any]:
    """Process a single instance with comprehensive error handling."""
    instance_dir = Path(instance_dir_str)
    inst_logger = setup_instance_logger(instance_dir)
    instance_id = instance_dir.name
    result_path = instance_dir / 'test_results.json'
    
    tag = ""
    
    try:
        # Check if we should skip
        if result_path.exists() and not force:
            inst_logger.info("Skipping %s (results exist, --force not set)", instance_id)
            return {"skipped": True, "instance_id": instance_id}
        
        inst_logger.info("Processing instance: %s", instance_id)
        
        # Build phase
        build_ok, tag, build_err = build_image(instance_dir, inst_logger, timeout=build_timeout)
        
        if not build_ok:
            tr = TestResult(
                instance_id=instance_id, 
                image_tag=tag, 
                build_ok=False, 
                test_invoked=False, 
                counts={}, 
                return_code=1, 
                error=build_err
            )
            result_path.write_text(json.dumps(tr.to_dict(), indent=2))
            inst_logger.error("Build failed: %s", build_err)
            
            # Clean up any partial build artifacts
            if tag:
                remove_image_safe(tag, inst_logger)
            
            return {"result": tr.to_dict(), "image_tag": tag}
        
        # Test phase
        test_ok, counts, rc, test_err, raw_output = run_tests_in_container(tag, instance_dir, inst_logger)
        
        tr = TestResult(
            instance_id=instance_id,
            image_tag=tag,
            build_ok=True,
            test_invoked=True,
            counts=counts,
            return_code=rc,
            error=test_err
        )
        
        # Save results
        result_path.write_text(json.dumps(tr.to_dict(), indent=2))
        (instance_dir / 'test_output.txt').write_text(raw_output)
        
        inst_logger.info("Completed %s: %s", instance_id, counts)
        
        return {"result": tr.to_dict(), "image_tag": tag}
        
    except Exception as exc:
        tb = traceback.format_exc()
        inst_logger.exception("Unhandled exception: %s", tb)
        
        tr = TestResult(
            instance_id=instance_id,
            image_tag=tag,
            build_ok=False,
            test_invoked=False,
            counts={},
            return_code=255,
            error=f"worker exception: {exc}"
        )
        
        try:
            result_path.write_text(json.dumps(tr.to_dict(), indent=2))
        except Exception:
            inst_logger.error("Failed to write error results")
        
        return {"result": tr.to_dict(), "image_tag": tag}
    
    finally:
        # Always try to remove the specific image we created
        if tag:
            remove_image_safe(tag, inst_logger)

# ---------------- Main execution -----------------
def main():
    parser = argparse.ArgumentParser(
        description="Parallel Docker build/test runner with batch processing"
    )
    parser.add_argument('--instances-dir', default='work_instances_parr', 
                       help='Directory containing work instances')
    parser.add_argument('--limit', type=int, 
                       help='Limit number of instances to process')
    parser.add_argument('--force', action='store_true',
                       help='Re-run instances even if results exist')
    parser.add_argument('--summary-file', default='docker_test_summary_parr.json',
                       help='Output file for aggregated results')
    parser.add_argument('--jobs', type=int, default=8,
                       help='Number of parallel workers (batch size will match this)')
    parser.add_argument('--build-timeout', type=int, default=3600,
                       help='Build timeout in seconds (default: 3600)')
    
    args = parser.parse_args()
    
    # Batch size is always same as jobs
    batch_size = args.jobs
    
    # Validate arguments
    instances_root = Path(args.instances_dir)
    if not instances_root.exists():
        print(f"ERROR: Instances directory {instances_root} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Setup logging
    global_logger = setup_global_logger(Path('docker_test_runs.log'))
    
    if not docker_available():
        global_logger.error("Docker CLI not found in PATH")
        sys.exit(2)
    
    # Find instances
    instance_dirs = [
        d for d in sorted(instances_root.iterdir()) 
        if d.is_dir() and (d / 'Dockerfile').exists()
    ]
    
    if args.limit:
        instance_dirs = instance_dirs[:args.limit]
    
    total_instances = len(instance_dirs)
    total_batches = (total_instances + batch_size - 1) // batch_size
    
    global_logger.info("Found %d instances to process in %d batches (batch_size=%d, jobs=%d)", 
                      total_instances, total_batches, batch_size, args.jobs)
    
    # Process instances in batches
    all_results: List[Dict[str, Any]] = []
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_instances)
        batch_dirs = instance_dirs[start_idx:end_idx]
        
        global_logger.info("=== BATCH %d/%d: Processing instances %d-%d ===", 
                          batch_num + 1, total_batches, start_idx + 1, end_idx)
        
        # Process batch
        batch_results = process_batch(batch_dirs, args, global_logger)
        all_results.extend(batch_results)
        
        # Clean up after batch (safe because no parallel builds running)
        global_logger.info("=== BATCH %d/%d COMPLETE: Cleaning up ===", batch_num + 1, total_batches)
        cleanup_everything(global_logger)
        
        # Progress summary
        completed_so_far = len(all_results)
        global_logger.info("=== PROGRESS: %d/%d instances completed ===", completed_so_far, total_instances)
    
    global_logger.info("=== ALL BATCHES COMPLETE ===")
    
    # Generate summary
    summary = {
        "results": [],
        "totals": {"instances": 0, "passed": 0, "failed": 0, "errors": 0, "invalid": 0, "skipped": 0}
    }
    
    for item in all_results:
        if item.get("skipped"):
            summary["totals"]["skipped"] += 1
            continue
        
        result_data = item.get("result", {})
        if "instance_id" in result_data:
            summary["results"].append(result_data)
            counts = result_data.get("counts", {})
            
            for key in ["passed", "failed", "errors", "invalid"]:
                summary["totals"][key] += counts.get(key, 0)
            
            summary["totals"]["instances"] += 1
    
    # Save summary
    Path(args.summary_file).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    
    # Final report
    totals = summary["totals"]
    global_logger.info(
        "FINAL SUMMARY: %d instances, %d passed, %d failed, %d errors, %d invalid, %d skipped",
        totals["instances"], totals["passed"], totals["failed"], 
        totals["errors"], totals["invalid"], totals["skipped"]
    )

if __name__ == '__main__':
    main()