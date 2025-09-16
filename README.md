# Repository Sandbox Generator

Generate ready-to-build Docker configurations for arbitrary Python repositories. Point it at a local path or a remote Git URL and it will automatically:

1. **Analyze Repository Structure**: Deep inspection of project layout and configuration
2. **Detect Python Version**: Intelligent heuristics across 10+ sources (.python-version, pyproject.toml, setup.py, etc.)  
3. **Aggregate Dependencies**: Comprehensive dependency resolution from multiple sources (requirements*, setup.py, pyproject.toml, Pipfile, environment.yml, setup.cfg)
4. **Infer System Packages**: Automatic detection of system dependencies (e.g., libpq-dev for psycopg2, libjpeg-dev)
5. **Detect Test Framework**: Support for pytest, unittest, Django tests, tox, and generic test setups
6. **Generate Optimized Dockerfile**: Single-stage Dockerfile with layered dependency caching for optimal build performance
7. **Create Supporting Files**: Generates analysis.json metadata and .dockerignore tuned for Python projects

## Installation

You can install from source (editable):

```bash
python -m pip install -e .[dev]
```

Or just runtime deps:

```bash
python -m pip install -e .
```


## Quick Start

### Basic Usage (Single Repository)

1. **Install the tool**:
```bash
cd /cb/data-platform/sahil/test/repo-sandbox-generator
pip install -e .
```

2. **Analyze a repository without generating Docker files**:
```bash
repo-sandbox analyze /path/to/your/repo
repo-sandbox analyze /path/to/your/repo --format json > analysis.json
```

3. **Generate Docker configuration for a local repository**:
```bash
# Basic generation
repo-sandbox generate /path/to/your/repo -o ./docker-output

# With tests and verbose output
repo-sandbox generate /path/to/your/repo -o ./docker-output --include-tests -v

# Override Python version
repo-sandbox generate /path/to/your/repo -o ./docker-output --python-version 3.11
```

4. **Generate from a remote Git repository**:
```bash
repo-sandbox from-git https://github.com/owner/project.git -o ./docker-output -v
```

### Advanced CLI Options

```bash
# Complete command reference
repo-sandbox --help
repo-sandbox generate --help
repo-sandbox from-git --help

# Generate with specific template
repo-sandbox generate /path/to/repo -o ./output --template scientific

# Skip copying source files (for CI/CD scenarios)
repo-sandbox generate /path/to/repo -o ./output --no-copy-source

# Different output formats for analysis
repo-sandbox analyze /path/to/repo --format yaml
repo-sandbox analyze /path/to/repo --format text
```

## Dataset Processing & Batch Operations

### Processing SWE-bench Datasets

The tool includes powerful scripts for processing entire datasets like SWE-bench:

#### Single Instance Processing
```bash
# Process a single instance by index
python scripts/generate_instance.py \
    --dataset /path/to/dataset.parquet \
    --index 0 \
    --commit-column base_commit \
    --cache-dir .cache/repos \
    --output-dir work_instances \
    --logs-dir logs

# Process a specific instance ID
python scripts/generate_instance.py \
    --dataset /path/to/dataset.parquet \
    --instance-id "some-instance-id" \
    --commit-column base_commit \
    --cache-dir .cache/repos \
    --output-dir work_instances
```

#### Batch Processing (All Instances)
```bash
# Process all instances in dataset with parallel workers
python scripts/generate_instance.py \
    --dataset /path/to/dataset.parquet \
    --all \
    --workers 8 \
    --limit 100 \
    --commit-column base_commit \
    --cache-dir .cache/repos \
    --output-dir work_instances \
    --logs-dir logs \
    --force

# Generate with full Git history (for cases needing complete clone)
python scripts/generate_instance.py \
    --dataset /path/to/dataset.parquet \
    --all \
    --workers 4 \
    --full-git \
    --fetch-parallelism 4 \
    --cache-dir .cache/repos \
    --output-dir work_instances
```

### Docker Build & Test Automation

#### Serial Testing (Basic)
```bash
# Test all generated instances with Docker
python scripts/run_docker_tests.py \
    --instances-dir work_instances \
    --jobs 1 \
    --build-timeout 300 \
    --summary-file results_summary.json

# Test with custom timeout and limited instances
python scripts/run_docker_tests.py \
    --instances-dir work_instances \
    --limit 50 \
    --jobs 1 \
    --build-timeout 600 \
    --force
```

#### Parallel Batch Testing (Recommended)
```bash
# High-throughput parallel testing with cleanup
python scripts/run_docker_tests_parr.py \
    --instances-dir work_instances \
    --jobs 8 \
    --build-timeout 3600 \
    --limit 200 \
    --summary-file docker_test_summary_parr.json

# Production-scale batch processing
python scripts/run_docker_tests_parr.py \
    --instances-dir work_instances_parr \
    --jobs 16 \
    --build-timeout 1800 \
    --force \
    --summary-file final_results.json
```

### Script Parameters Reference

#### `generate_instance.py` Options:
- `--dataset`: Path to parquet dataset file (required)
- `--index`: Row index to process (for single instance)
- `--instance-id`: Specific instance ID to process
- `--commit-column`: Which commit column to use (`environment_setup_commit` or `base_commit`)
- `--all`: Process all instances in dataset
- `--workers`: Number of parallel workers (0 = auto-detect CPU count)
- `--limit`: Maximum number of instances to process
- `--cache-dir`: Directory for caching cloned repositories
- `--output-dir`: Output directory for generated instances
- `--logs-dir`: Directory for per-instance logs
- `--force`: Overwrite existing instance directories
- `--skip-existing`: Skip instances that already exist
- `--full-git`: Create full clone with Git history (vs lightweight worktree)
- `--fetch-parallelism`: Max concurrent Git fetch operations
- `--no-tests`: Skip test framework detection and setup

#### `run_docker_tests*.py` Options:
- `--instances-dir`: Directory containing work instances
- `--jobs`: Number of parallel build/test workers
- `--limit`: Maximum instances to process
- `--build-timeout`: Docker build timeout in seconds
- `--force`: Re-run instances with existing results
- `--summary-file`: Output file for aggregated results JSON

### Output Structure

After processing, you'll have:
```
work_instances/
├── instance-1/
│   ├── Dockerfile                    # Generated Dockerfile
│   ├── .dockerignore                # Docker ignore patterns
│   ├── analysis.json                # Repository analysis results
│   ├── test_cases_pass_to_pass.json # Test cases from dataset
│   ├── test_cases_fail_to_pass.json # Failing test cases
│   ├── dataset_row.json             # Original dataset row
│   ├── test_results.json            # Docker test results
│   ├── test_output.txt              # Raw test output
│   ├── logs/
│   │   ├── process.log              # Instance generation log
│   │   └── test_run.log             # Docker test run log
│   └── [source files]              # Repository source code
├── instance-2/
├── ...
└── batch_summary.json              # Overall batch processing summary
```

## Generated Output & Docker Usage

### Single Repository Output
Inside the chosen output directory (default `./docker-config`):
```
docker-config/
├── Dockerfile                 # Optimized single-stage build
├── .dockerignore             # Python-optimized ignore patterns  
├── analysis.json             # Complete repository analysis metadata
├── requirements.txt          # Copied dependency files (as available)
├── pyproject.toml           # 
├── setup.py                 # 
└── [source files/]          # Full source tree (if --copy-source enabled)
```

### Docker Build & Run
```bash
cd docker-config

# Build image (tag automatically determined from analysis)
docker build -t <repo_name> .

# Run with port mapping
docker run -p 8000:8000 <repo_name>

# Run interactively for debugging
docker run -it <repo_name> bash

# Build with custom tag
docker build -t my-custom-name .
```

### Analysis Output Format
The `analysis.json` contains comprehensive metadata:
```json
{
  "repo_name": "example-project",
  "repo_path": "/path/to/repo",
  "python_version": "3.11",
  "python_packages": ["requests", "numpy", "pytest"],
  "system_packages": ["build-essential", "libjpeg-dev"],
  "environment_vars": {
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1"
  },
  "test_config": {
    "framework": "pytest",
    "test_paths": ["tests/"],
    "test_commands": ["python -m pytest"]
  },
  "build_system": "pyproject",
  "dependency_files": ["pyproject.toml", "requirements-dev.txt"],
  "errors": []
}
```