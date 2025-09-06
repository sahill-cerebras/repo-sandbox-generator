"""Dockerfile generation module."""

import logging
from typing import Dict, Any, List
from pathlib import Path
from jinja2 import Template

logger = logging.getLogger(__name__)


class DockerfileGenerator:
    """Generates a (now single-stage) Dockerfile based on repository analysis.

    Changes (single-stage simplification per user request):
    - Collapsed multi-stage builder/runtime into one stage for simplicity.
    - Still preserves ordered layering for dependency caching.
    - Retains apt/apk detection, optional tests, micromamba (if environment.yml).
    - Keeps start command inference.
    NOTE: Build context MUST be the repository root to include source code.
    """

    BASE_TEMPLATES = {
        'slim': 'python:{version}-slim-bullseye',
        'full': 'python:{version}-bullseye',
        'alpine': 'python:{version}-alpine',
        'scientific': 'python:{version}-slim-bullseye',  # scientific extras layered via pip
    }

    DOCKERFILE_TEMPLATE = """
# Generated Dockerfile for {{ repo_name }} (single-stage)
# Build with: docker build -t {{ repo_name }} -f /path/to/Dockerfile /absolute/path/to/repo
# Python {{ python_version }}

ARG PYTHON_VERSION={{ python_version }}
FROM {{ base_image }}

WORKDIR /app

# Detect package manager (apt or apk)
{% set uses_apk = 'alpine' in base_image %}
{% if system_packages %}
{% if uses_apk %}
RUN apk add --no-cache {% for pkg in system_packages %}{{ pkg }} {% endfor %}
{% else %}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends{% for pkg in system_packages %} {{ pkg }}{% endfor %} \
    && rm -rf /var/lib/apt/lists/*
{% endif %}
{% endif %}

# Environment variables
{% for key, value in environment_vars.items() %}ENV {{ key }}="{{ value }}"
{% endfor %}

RUN python -m pip install --upgrade pip

# Copy dependency spec files first for caching
{% if dependency_files %}
COPY {{ dependency_files | join(' ') }} ./
{% endif %}

{% if has_environment_yml %}
# Conda environment via micromamba (lightweight)
RUN wget -qO /usr/local/bin/micromamba https://micro.mamba.pm/api/micromamba/linux-64/latest \
    && chmod +x /usr/local/bin/micromamba \
    && micromamba create -y -n app -f environment.yml \
    && micromamba clean -a -y
ENV PATH="/usr/local/micromamba/envs/app/bin:${PATH}"
{% endif %}

# Install base dependencies (layer cached if specs unchanged)
RUN set -eux; \
    if [ -f requirements.txt ]; then python -m pip install -r requirements.txt; \
    elif [ -f Pipfile ]; then pip install pipenv && pipenv install --system --deploy; \
    elif [ -f pyproject.toml ]; then python -m pip install .; \
    elif [ -f setup.py ]; then python -m pip install -e .; \
    elif [ -f environment.yml ]; then echo "Conda environment handled above"; \
    else echo "No primary dependency file found"; fi

{% if include_tests and test_commands %}
# Optional test dependencies (re-run only if specs change)
{% if test_requirements_hint %}# Hint: {{ test_requirements_hint }}{% endif %}
RUN set -eux; if [ -f requirements-dev.txt ]; then python -m pip install -r requirements-dev.txt; fi
{% endif %}

# Copy full source AFTER deps for better caching (remember build context must be repo root)
COPY . .

{% if include_tests and test_commands %}
# Run tests at build time (can be removed if not desired)
RUN set -eux; {{ test_commands | join(' && ') }}
{% endif %}

{% if expose_ports %}
{% for port in expose_ports %}EXPOSE {{ port }}
{% endfor %}
{% endif %}

# Default start command (auto-inferred, override with docker run ...)
{{ default_cmd }}
""".strip()

    def generate_dockerfile(self, analysis: Dict[str, Any], template: str = 'auto', include_tests: bool = False) -> str:
        if template == 'auto':
            template = self._select_optimal_template(analysis)
        base_image = self._get_base_image(template, analysis['python_version'])
        dep_files = analysis.get('dependency_files', [])
        # Only include files that actually exist in build context
        dependency_files = [f for f in dep_files if f]
        test_cfg = analysis.get('test_config', {}) or {}
        test_commands = test_cfg.get('test_commands', []) if include_tests else []
        default_cmd = self._infer_default_cmd(analysis)
        template_vars = {
            'repo_name': analysis['repo_name'],
            'python_version': analysis['python_version'],
            'base_image': base_image,
            'system_packages': self._optimize_system_packages(analysis.get('system_packages', [])),
            'environment_vars': analysis.get('environment_vars', {}),
            'dependency_files': dependency_files,
            'has_environment_yml': 'environment.yml' in dependency_files,
            'expose_ports': analysis.get('expose_ports', []),
            'include_tests': include_tests,
            'test_commands': test_commands,
            'test_requirements_hint': 'requirements-dev.txt present' if 'requirements-dev.txt' in dependency_files else '',
            'default_cmd': default_cmd,
        }
        dockerfile_content = Template(self.DOCKERFILE_TEMPLATE).render(**template_vars)
        logger.info("Generated SINGLE-STAGE Dockerfile using %s template", template)
        return dockerfile_content

    def generate_dockerignore(self, analysis: Dict[str, Any]) -> str:
        ignore_patterns = [
            '.git', '.gitignore', '.gitattributes',
            '__pycache__', '*.pyc', '*.pyo', '*.pyd', '.Python', 'env', 'venv', '.venv', '.env',
            '.vscode', '.idea', '*.swp', '*.swo', '*~',
            '.DS_Store', 'Thumbs.db',
            'build', 'dist', '*.egg-info', '.eggs',
            '.pytest_cache', '.coverage', 'htmlcov', '.tox',
            'docs/_build', 'Dockerfile*', 'docker-compose*.yml', '.dockerignore', '*.log'
        ]
        return '\n'.join(ignore_patterns) + '\n'

    def _select_optimal_template(self, analysis: Dict[str, Any]) -> str:
        packages = analysis.get('python_packages', [])
        scientific = ['numpy', 'scipy', 'matplotlib', 'pandas', 'scikit-learn', 'tensorflow', 'torch']
        if any(any(pkg.lower().startswith(s) for s in scientific) for pkg in packages):
            return 'scientific'
        return 'slim'

    def _get_base_image(self, template: str, python_version: str) -> str:
        return self.BASE_TEMPLATES.get(template, self.BASE_TEMPLATES['slim']).format(version=python_version)

    def _optimize_system_packages(self, packages: List[str]) -> List[str]:
        optimized = sorted(set(packages))
        for essential in ['curl', 'wget', 'git']:
            if essential not in optimized:
                optimized.insert(0, essential)
        return optimized

    # Removed runtime system package pruning in single-stage mode; keep all.

    def _infer_default_cmd(self, analysis: Dict[str, Any]) -> str:
        """Infer a reasonable default CMD.

        Heuristics:
        - manage.py present -> runserver 0.0.0.0:8000
        - uvicorn in deps & any module ending with app/main/api -> uvicorn module:app
        - gunicorn in deps & wsgi.py present -> gunicorn module:wsgi.application
        - fallback simple python -m package or interactive notice.
        """
        repo_path = Path(analysis.get('repo_path', '.'))
        pkgs = [p.lower() for p in analysis.get('python_packages', [])]
        manage_py = (repo_path and Path(repo_path) / 'manage.py').exists()
        if manage_py:
            return 'CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]'
        if any(p.startswith('uvicorn') for p in pkgs):
            # naive search
            candidates = ['app', 'main', 'api']
            for c in candidates:
                if (Path(repo_path) / f'{c}.py').exists():
                    return f'CMD ["uvicorn", "{c}:app", "--host", "0.0.0.0", "--port", "8000"]'
        if any(p.startswith('gunicorn') for p in pkgs):
            if (Path(repo_path) / 'wsgi.py').exists():
                return 'CMD ["gunicorn", "wsgi:application", "--bind", "0.0.0.0:8000"]'
        # fallback
        return 'CMD ["python", "-c", "print(\'Container ready. Override CMD to run your application.\')"]'

    def _generate_default_command(self, analysis: Dict[str, Any]) -> str:  # compatibility stub
        return None
