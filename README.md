# Repository Sandbox Generator

Generate ready-to-build Docker configurations for arbitrary Python repositories. Point it at a local path or a remote Git URL and it will:

1. Analyze the repository structure
2. Detect Python version (heuristic)
3. Aggregate dependencies from multiple sources (requirements*, setup.py, pyproject.toml, Pipfile, environment.yml, setup.cfg)
4. Infer system (APT / APK) packages sometimes required for popular Python wheels
5. Detect test setup (pytest, unittest, Django tests, tox, or generic) and optionally embed test execution into the Docker build
6. Produce a single‑stage Dockerfile optimized for build caching (dependency spec files copied before source)
7. Emit analysis metadata (analysis.json) and a .dockerignore tuned for Python projects

## Features
- Single-stage Dockerfile with layered dependency caching
- Automatic base image selection (slim vs scientific) based on detected packages
- Optional inclusion of test commands at build time
- Heuristic system package inference (e.g. libpq-dev for psycopg2)
- Support for: requirements files (flat or layered), setup.py, pyproject.toml (PEP 621 / Poetry), Pipfile, environment.yml, setup.cfg
<!-- docker-compose generation removed -->

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

## CLI Usage

After installation the entrypoint `repo-sandbox` is available.

### Analyze a local repository
```bash
repo-sandbox analyze /path/to/repo
```

### Generate Docker assets for a local repository
```bash
repo-sandbox generate /path/to/repo -o ./docker-config --include-tests -v
```

### Generate directly from a remote Git URL
```bash
repo-sandbox from-git https://github.com/owner/project.git -o ./docker-config -v
```

### Help
```bash
repo-sandbox --help
repo-sandbox generate --help
```

## Generated Output (default)
Inside the chosen output directory (default `./docker-config`):
- `Dockerfile` – single stage build
<!-- docker-compose.yml generation removed -->
- `.dockerignore`
- `analysis.json` – structured metadata of the repository
- Copied dependency definition files (e.g. `pyproject.toml`, `requirements.txt`, etc.)
- (Optional) full source tree copy if `--no-copy-source` not passed

## Docker Build / Run
```bash
cd docker-config
# Build (image tag equals lowercase repo name in analysis)
docker build -t <repo_name> .
# Run (example mapping port 8000)
docker run -p 8000:8000 <repo_name>
```
<!-- docker-compose instructions removed -->

## Python Version Detection
The analyzer currently uses heuristics (files & metadata) and falls back to a default (3.9) when uncertain. You can override with `--python-version` during `generate`.

## Tests Integration
If `--include-tests` is provided, test commands (pytest / unittest / Django / tox) are executed during the image build for early validation. Disable for faster iteration or CI layering. Dev requirements are installed if a `requirements-dev.txt` file is detected.

## System Package Inference
A small curated mapping converts certain Python packages into required Debian/APK packages (e.g. Pillow -> libjpeg-dev, libpng-dev). This is intentionally conservative; extend `DependencyResolver._infer_system_dependencies` for more coverage.

## Extensibility Points
- Add new dependency sources: extend `DependencyResolver.resolve_all_dependencies`
- Add new test detection heuristics: update `TestDiscovery`
- Add framework-specific behavior: (framework detection intentionally removed in this iteration)
- Enhance default command inference in `DockerfileGenerator._infer_default_cmd`

## Development
```bash
# Lint / format
flake8 src/
black src/

# Type check
mypy src/

# Run tests (if repository has tests for this project itself)
pytest -q
```

## Project Structure
```
src/repo_sandbox/
  cli.py                      # CLI entrypoint (click group)
  analyzer/
    repo_analyzer.py          # Orchestrates full analysis
    dependency_resolver.py    # Collects python/system deps & install commands
    test_discovery.py         # Detects test frameworks & commands
    version_detector.py       # Python version heuristics
  generators/
    dockerfile_generator.py   # Builds single-stage Dockerfile & .dockerignore
  <!-- docker_compose_generator.py removed -->
  utils/
    file_utils.py             # File helpers (if extended)
    git_utils.py              # Git helpers (for potential future use)
```

## Requirements File Regeneration
A base `requirements.txt` (runtime) derived from `pyproject.toml` core dependencies is provided below. Dev tools are in an extras section (`.[dev]`).

## License
MIT

## Disclaimer
This README was regenerated automatically after accidental deletion. Adjust wording, org links, or branding as needed.
