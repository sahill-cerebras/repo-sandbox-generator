# Repository Sandbox Generator

Generate ready-to-build Docker configurations for arbitrary Python repositories. Point it at a local path or a remote Git URL and it will automatically:

1. **Analyze Repository Structure**: Deep inspection of project layout and configuration
2. **Detect Python Version**: Intelligent heuristics across 10+ sources (.python-version, pyproject.toml, setup.py, etc.)  
3. **Aggregate Dependencies**: Comprehensive dependency resolution from multiple sources (requirements*, setup.py, pyproject.toml, Pipfile, environment.yml, setup.cfg)
4. **Infer System Packages**: Automatic detection of system dependencies (e.g., libpq-dev for psycopg2, libjpeg-dev for Pillow)
5. **Detect Test Framework**: Support for pytest, unittest, Django tests, tox, and generic test setups
6. **Generate Optimized Dockerfile**: Single-stage Dockerfile with layered dependency caching for optimal build performance
7. **Create Supporting Files**: Generates analysis.json metadata and .dockerignore tuned for Python projects

## Key Features
- **Smart Base Image Selection**: Automatically chooses between slim/scientific base images based on detected packages
- **Intelligent Caching**: Optimized layer ordering for maximum Docker build cache efficiency  
- **Test Integration**: Optional embedding of test execution into Docker build process
- **Comprehensive Dependency Support**: Handles requirements files (flat/nested), setup.py, pyproject.toml (PEP 621/Poetry), Pipfile, Conda environments
- **System Package Inference**: Automatically maps Python packages to required system dependencies
- **Parallel Processing**: Built-in scripts for batch processing datasets and parallel testing

## Installation

You can install from source (editable):

```bash
python -m pip install -e .[dev]
```

Or just runtime deps:

```bash
python -m pip install -e .
```

(When published to PyPI it will be installable via `pip install repo-sandbox-generator`.)

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
    --commit-column environment_setup_commit \
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
    --commit-column environment_setup_commit \
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
# High-throughput parallel testing with intelligent cleanup
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

## Technical Details

### Python Version Detection
The analyzer uses intelligent heuristics across 10+ sources with priority ordering:
1. `.python-version` file (highest priority)
2. `runtime.txt` (Heroku format)
3. `pyproject.toml` (`requires-python`, Poetry, PDM)
4. `setup.py` (`python_requires`)
5. `setup.cfg` (`python_requires`)
6. `Dockerfile` (FROM python:X.Y)
7. GitHub Actions workflows (setup-python versions)
8. `tox.ini` (basepython, envlist)
9. `environment.yml` (conda python version)
10. README files (version mentions)
11. Requirements file comments

Default fallback: **Python 3.11**. Override with `--python-version` flag.

### Test Framework Integration
When `--include-tests` is enabled:
- **pytest**: Detected via config files or dependencies → `python -m pytest`
- **Django**: Detected via `manage.py` + Django dependency → `python manage.py test`
- **unittest**: Detected via import analysis → `python -m unittest discover`
- **tox**: Detected via `tox.ini` → `tox`
- **generic**: Fallback when test directories exist → `python -m unittest discover`

Test commands are embedded in Dockerfile for early validation during build.

### System Package Inference
Automatic mapping of Python packages to system dependencies:
- `psycopg2` → `postgresql-dev`, `libpq-dev`
- `Pillow` → `libjpeg-dev`, `libpng-dev`, `libtiff-dev`
- `lxml` → `libxml2-dev`, `libxslt-dev`
- `numpy/scipy` → `build-essential`, `gfortran`, `libatlas-base-dev`
- `matplotlib` → `pkg-config`, `libfreetype6-dev`
- `cryptography` → `libssl-dev`, `libffi-dev`

### Base Image Selection Logic
- **Scientific**: Triggered by packages like numpy, scipy, matplotlib, pandas, tensorflow
- **Slim**: Default for most Python projects (smaller footprint)
- **Full**: Used when scientific packages detected or specified via `--template`

### Docker Optimization Features
- **Layer Caching**: Dependencies copied and installed before source code
- **Build Caching**: Multi-stage approach with pip cache optimization
- **Size Optimization**: APT cache cleanup, pip no-cache flags
- **Python Optimizations**: PYTHONDONTWRITEBYTECODE=1, PYTHONUNBUFFERED=1

## Real-World Examples

### Example 1: Analyzing a Flask Web Application
```bash
# Analyze a Flask project
repo-sandbox analyze /path/to/flask-app --format json

# Expected output shows:
# - Python version detection from runtime.txt or .python-version  
# - Dependencies: flask, gunicorn, psycopg2-binary
# - System packages: postgresql-dev, libpq-dev
# - Test framework: pytest (if configured)
# - Base image: slim (web apps don't need scientific packages)
```

### Example 2: Processing a Machine Learning Repository  
```bash
# Generate Docker config for ML project
repo-sandbox generate /path/to/ml-project -o ./ml-docker --include-tests

# Expected behavior:
# - Scientific base image (detects numpy, pandas, scikit-learn)
# - Extended system packages for compilation
# - Jupyter/notebook support if detected
# - Test integration with pytest/unittest
```

### Example 3: SWE-bench Dataset Processing Pipeline
```bash
# Full pipeline for processing SWE-bench instances
cd /cb/data-platform/sahil/test/repo-sandbox-generator

# 1. Generate instances from dataset
python scripts/generate_instance.py \
    --dataset /path/to/swe-bench-dataset.parquet \
    --all \
    --workers 8 \
    --cache-dir .cache/repos \
    --output-dir work_instances \
    --logs-dir logs \
    --commit-column environment_setup_commit

# 2. Build and test all instances in parallel
python scripts/run_docker_tests_parr.py \
    --instances-dir work_instances \
    --jobs 8 \
    --build-timeout 1800 \
    --summary-file results.json

# 3. Analyze results
python -c "
import json
with open('results.json') as f: 
    data = json.load(f)
    totals = data['totals']
    print(f'Processed: {totals[\"instances\"]}')  
    print(f'Passed: {totals[\"passed\"]}')
    print(f'Failed: {totals[\"failed\"]}')
"
```

## Installation

### From Source (Recommended)
```bash
cd /cb/data-platform/sahil/test/repo-sandbox-generator

# Install with development dependencies
pip install -e .[dev]

# Or runtime dependencies only  
pip install -e .

# Verify installation
repo-sandbox --help
```

### Dependencies
Core requirements:
- `click>=8.0.0` (CLI framework)
- `toml>=0.10.0` (TOML parsing) 
- `pyyaml>=6.0` (YAML parsing)
- `jinja2>=3.0.0` (Template engine)
- `requests>=2.25.0` (HTTP requests)
- `gitpython>=3.1.0` (Git operations)
- `packaging>=21.0` (Version parsing)

Development dependencies:
- `pytest>=7.0`, `pytest-cov>=4.0`
- `black>=22.0`, `flake8>=5.0`, `mypy>=1.0`

### Dataset Processing Requirements
For SWE-bench dataset processing:
```bash
pip install pandas fastparquet pyarrow
```

## Project Architecture

```
src/repo_sandbox/
├── cli.py                    # Main CLI interface (Click-based)
├── analyzer/
│   ├── repo_analyzer.py      # Orchestrates complete repository analysis  
│   ├── dependency_resolver.py # Multi-source dependency aggregation
│   ├── test_discovery.py     # Test framework detection & configuration
│   └── version_detector.py   # Python version detection heuristics
├── generators/
│   └── dockerfile_generator.py # Optimized Dockerfile generation
└── utils/
    ├── file_utils.py         # File system utilities
    └── git_utils.py          # Git repository operations

scripts/
├── generate_instance.py     # SWE-bench dataset processing
├── run_docker_tests.py      # Serial Docker build & test runner
└── run_docker_tests_parr.py # Parallel batch processing with cleanup
```

## Extensibility Points

### Adding New Dependency Sources
Extend `DependencyResolver.resolve_all_dependencies()`:
```python
# Add to DEP_SOURCES list in __init__
self.DEP_SOURCES.append(('custom.toml', self._parse_custom_format))

def _parse_custom_format(self, file_path: Path) -> ParseResult:
    # Implementation for new format
    return {'python_packages': [...]}
```

### Adding New Test Framework Detection
Update `TestDiscovery` class:
```python
def _check_custom_test_framework(self, repo_path: Path) -> Optional[Dict[str, Any]]:
    # Detection logic
    if self._is_dependency_present(repo_path, r'^custom-test-framework'):
        return {
            'framework': 'custom',
            'test_commands': ['custom-test-runner'],
            'test_paths': test_paths
        }
```

### Adding New Python Version Sources
Extend `PythonVersionDetector`:
```python
# Add to detection_methods list in detect_python_version()
(VersionSource.CUSTOM_FILE, 'custom-version.txt', self._parse_custom_version),
```

## Development Workflow

```bash
# Format code
black src/ scripts/

# Lint  
flake8 src/ scripts/

# Type checking
mypy src/

# Run any existing tests
pytest -v

# Test CLI commands
repo-sandbox analyze tests/fixtures/sample-repo
```

## License
MIT

## Troubleshooting

### Common Issues

**"Docker CLI not found"**
```bash
# Install Docker and ensure it's in PATH
which docker
docker --version
```

**"Build timeout" errors**  
```bash
# Increase timeout for complex builds
python scripts/run_docker_tests.py --build-timeout 3600
```

**"Permission denied" accessing Git repositories**
```bash
# For private repositories, ensure SSH keys are configured
ssh -T git@github.com

# Or use HTTPS with tokens
git config --global credential.helper store
```

**High memory usage during batch processing**
```bash
# Reduce parallel workers and batch size
python scripts/generate_instance.py --workers 4
python scripts/run_docker_tests_parr.py --jobs 4
```

**Python version not detected correctly**
```bash
# Override version detection
repo-sandbox generate /path/to/repo --python-version 3.11

# Check what was detected
repo-sandbox analyze /path/to/repo --format json | jq '.python_version'
```

### Performance Tips

1. **Use caching for repeated operations**:
   ```bash
   # Reuse cache directory for multiple dataset runs
   --cache-dir .cache/repos
   ```

2. **Optimize parallel processing**:
   ```bash
   # Match workers to CPU cores, consider memory constraints
   --workers $(nproc)
   --jobs $(nproc)
   ```

3. **Use appropriate timeouts**:
   ```bash
   # Adjust based on project complexity
   --build-timeout 1800  # 30 minutes for complex scientific packages
   --build-timeout 300   # 5 minutes for simple web apps
   ```

4. **Enable Docker BuildKit for faster builds**:
   ```bash
   export DOCKER_BUILDKIT=1
   ```

## Contributing

We welcome contributions! Areas for improvement:

- **New dependency sources**: Support for additional package managers
- **Enhanced system package detection**: More Python package → system dependency mappings  
- **Framework-specific optimizations**: Django, FastAPI, Flask-specific Dockerfile patterns
- **Cloud platform support**: AWS/GCP/Azure deployment configurations
- **Performance improvements**: Faster analysis, better caching strategies

## Related Projects

- **SWE-bench**: Software engineering evaluation benchmark
- **Docker**: Container platform for packaging applications
- **Poetry**: Modern Python dependency management  
- **pipenv**: Python virtual environment management
