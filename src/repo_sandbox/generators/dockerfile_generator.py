"""Optimized Dockerfile generation module with enhanced flow and accuracy."""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
from jinja2 import Template, Environment, BaseLoader
from enum import Enum
import json as _json
from urllib.request import urlopen, Request as _UrlRequest
import urllib.error as _urlerr
import re

logger = logging.getLogger(__name__)


class BaseImageType(Enum):
    FULL = 'full'
    SLIM = 'slim'
    SCIENTIFIC = 'scientific'


class PackageManager(Enum):
    APT = 'apt'


class DockerfileGenerator:
    """
    Optimized Dockerfile generator with improved flow control and accuracy.
    
    Key improvements:
    - Better template selection logic
    - Enhanced dependency detection
    - Improved caching strategies
    - More accurate command inference
    - Better error handling and validation
    - Support for more frameworks and environments
    """

    # Enhanced base image templates with more options
    BASE_TEMPLATES = {
        # BaseImageType.FULL: 'python:{version}-{codename}',
        # BaseImageType.SLIM: 'python:{version}-slim-{codename}',
        # BaseImageType.SCIENTIFIC: 'python:{version}-{codename}',
        BaseImageType.FULL: 'python:{version}',
        BaseImageType.SLIM: 'python:{version}-slim',
        BaseImageType.SCIENTIFIC: 'python:{version}',
    }


    # Scientific packages for detection
    SCIENTIFIC_PACKAGES = {
        'numpy', 'scipy', 'matplotlib', 'pandas', 'scikit-learn',
        'tensorflow', 'torch', 'pytorch', 'keras', 'xgboost',
        'lightgbm', 'catboost', 'jupyterlab', 'notebook'
    }

    # Web framework detection patterns
    WEB_FRAMEWORKS: dict[str, dict] = {}

    # Optimized Dockerfile template with better structure
    DOCKERFILE_TEMPLATE = """# Generated Dockerfile for {{ repo_name }}
# Build: docker build -t {{ repo_name|lower|replace(' ', '-') }} -f Dockerfile .
# Python {{ python_version }} on {{ base_image_type.value }}

ARG PYTHON_VERSION={{ python_version }}
FROM {{ base_image }} AS final

# Set build arguments for better caching
ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_NO_CACHE_DIR=1
ARG PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

{%- if needs_eol_debian_fix %}
# --- FIX for EOL Debian Stretch repos ---
RUN sed -i 's/deb.debian.org/archive.debian.org/g' /etc/apt/sources.list \
 && sed -i '/security.debian.org/d' /etc/apt/sources.list \
 && sed -i '/stretch-updates/d' /etc/apt/sources.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        wget ca-certificates curl build-essential git bzip2 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

{%- endif %}
# Install system dependencies (APT only)
{%- set base_pkgs = system_packages or [] %}
RUN apt-get update && \
    apt-get install -y --no-install-recommends {{ (base_pkgs + ['wget','curl','ca-certificates','build-essential','git','bzip2']) | join(' ') }} && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

{# Extra build tools for pyproject.toml/meson-based projects (matplotlib, scipy, etc.) #}
{%- if 'pyproject.toml' in dependency_files and package_manager == PackageManager.APT %}
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ gfortran make pkg-config \
        meson ninja-build \
        libffi-dev libssl-dev \
        libfreetype6-dev libpng-dev libjpeg-dev libtiff-dev \
        libopenblas-dev liblapack-dev libqhull-dev \
        libx11-dev libxext-dev libxrender-dev libsm-dev \
        && apt-get clean && rm -rf /var/lib/apt/lists/*
{%- endif %}

# Optional Node.js
{%- if add_nodejs %}
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g npm@latest
{%- endif %}

# Environment variables
{%- for key, value in environment_vars.items() %}
ENV {{ key }}="{{ value }}"
{%- endfor %}

# Upgrade pip base tools (done early to have modern wheel/build support)
RUN python -m pip install --upgrade pip setuptools wheel build cython

# -------------------------------
# Python dependencies (priority order)
# Prefers Conda environment.yml when present (more reproducible for complex stacks)
# -------------------------------
{%- if 'environment.yml' in dependency_files or 'environment.yaml' in dependency_files %}
# --- Conda environment installation (preferred when provided) ---
{%- set env_file = 'environment.yml' if 'environment.yml' in dependency_files else 'environment.yaml' %}
RUN apt-get update && apt-get install -y curl bash
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
RUN bash miniconda.sh -b -p /opt/conda
RUN rm miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"
RUN touch /opt/conda/.conda_installed


# Accept Anaconda TOS for default channels (best-effort; harmless if unavailable)
RUN conda init bash || true && \
    conda config --add channels defaults || true && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

COPY {{ env_file }} /tmp/environment.yml
RUN conda env update -n base -f /tmp/environment.yml
RUN conda clean -a -y 
RUN echo "Activated conda env 'base'" && \
    echo "dependencies captured from {{ env_file }}" 

{%- elif 'requirements.txt' in dependency_files %}
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

{%- elif 'Pipfile' in dependency_files %}
COPY Pipfile* /tmp/
RUN pip install pipenv && pipenv install --system --deploy

{%- elif 'poetry.lock' in dependency_files %}
COPY pyproject.toml poetry.lock /tmp/
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-dev

{%- elif 'pyproject.toml' in dependency_files %}
# Generic PEP 517/518 handling: try Poetry, Flit, or build wheel then install
COPY . /tmp/project
WORKDIR /tmp/project

RUN pip install --upgrade pip setuptools wheel build cython 

# Poetry branch
RUN if grep -q "\[tool.poetry\]" pyproject.toml 2>/dev/null; then \
      pip install poetry && \
      poetry config virtualenvs.create false && \
      poetry install --no-dev; \
    fi

# Flit branch
RUN if grep -q "\[tool.flit\]" pyproject.toml 2>/dev/null || grep -q "flit_core" pyproject.toml 2>/dev/null; then \
      pip install flit && \
      python -m build --wheel && \
      pip install --no-cache-dir dist/*.whl; \
    fi

# Fallback pip branch
RUN pip install  --no-cache-dir -e .[test] || \
    pip install  --no-cache-dir -e . || \
    pip install  --no-cache-dir . || \
    (python -m build --wheel && pip install  --no-cache-dir dist/*.whl)

WORKDIR /app

{%- elif 'setup.py' in dependency_files or 'setup.cfg' in dependency_files %}
# Classic setuptools-based project
COPY . /tmp/project
WORKDIR /tmp/project
RUN pip install --upgrade pip setuptools wheel
RUN pip install  --no-cache-dir  -e .{{ setuptools_extras }}
WORKDIR /app

{%- else %}
RUN echo "No dependency file found; skipping Python package installation"
{%- endif %}

# Copy application code (always copy after dependency installs to leverage Docker cache)
COPY . .

{%- if frontend_build_commands %}
RUN {{ frontend_build_commands|join(' && ') }}
{%- endif %}

# Always install pytest (make image test-ready) plus normalized test-only dependencies
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir pytest{% if installable_test_imports %} {{ installable_test_imports|join(' ') }}{% endif %}

# Final aggregate install of all declared packages (python_packages + extras + test imports)
{% if all_declared_packages %}
RUN echo "Installing aggregated declared packages ({{ all_declared_packages|length }})" && \
    pip install --no-cache-dir {% for p in all_declared_packages %}{{ p }} {% endfor %}
{% endif %}

"""


    def __init__(self):
        """Initialize the Dockerfile generator."""
        self.env = Environment(loader=BaseLoader())
        self.template = self.env.from_string(self.DOCKERFILE_TEMPLATE)

    def generate_dockerfile(
        self,
        analysis: Dict[str, Any],
        template: str = 'auto',
        include_tests: bool = False,
    ) -> str:
        """
        Generate an optimized Dockerfile based on repository analysis.
        
        Args:
            analysis: Repository analysis results
            template: Base image template to use
            include_tests: Whether to run tests during build
            include_dev_deps: (removed)
            optimize_size: (removed)
            
        Returns:
            Generated Dockerfile content
        """
        # Validate and enhance analysis
        analysis = self._validate_and_enhance_analysis(analysis)
        
        # Select optimal template
        base_image_type = self._select_optimal_template(analysis) if template == 'auto' else BaseImageType(template)
        
        # Get base image and package manager
        base_image = self._get_base_image(base_image_type, analysis['python_version'])
        package_manager = self._detect_package_manager(base_image)
        
        # Prepare template variables
        template_vars = self._prepare_template_variables(
            analysis, base_image, base_image_type, package_manager,
            include_tests
        )
        
        # Generate Dockerfile
        dockerfile_content = self.template.render(**template_vars)
        
    # size optimization removed
        
        logger.info(
            "Generated optimized Dockerfile using %s template with %s package manager",
            base_image_type.value, package_manager.value
        )
        
        return dockerfile_content

    def generate_dockerignore(self, analysis: Dict[str, Any], custom_patterns: List[str] = None) -> str:
        """
        Generate an optimized .dockerignore file.
        
        Args:
            analysis: Repository analysis results
            custom_patterns: Additional patterns to ignore
            
        Returns:
            .dockerignore content
        """
        # ignore_patterns = [
        #     # Version control
        #     '.git', '.gitignore', '.gitattributes', '.svn', '.hg',
            
        #     # Python
        #     '__pycache__', '*.py[cod]', '*$py.class', '*.so',
        #     '.Python', 'build/', 'develop-eggs/', 'dist/', 'downloads/',
        #     'eggs/', '.eggs/', 'lib/', 'lib64/', 'parts/', 'sdist/',
        #     'var/', 'wheels/', '*.egg-info/', '.installed.cfg', '*.egg',
        #     'MANIFEST', 'pip-log.txt', 'pip-delete-this-directory.txt',
            
        #     # Virtual environments
        #     'env/', 'venv/', 'ENV/', '.venv/', '.env',
            
        #     # Testing
        #     '.tox/', '.nox/', '.coverage', '.coverage.*', '.cache',
        #     'nosetests.xml', 'coverage.xml', '*.cover', '.hypothesis/',
        #     '.pytest_cache/', '.mypy_cache/', '.dmypy.json', 'dmypy.json',
            
        #     # IDEs
        #     '.vscode/', '.idea/', '*.swp', '*.swo', '*~', '.project',
        #     '.pydevproject', '.settings/', '.classpath',
            
        #     # OS
        #     '.DS_Store', 'Thumbs.db', 'ehthumbs.db', 'Desktop.ini',
            
        #     # Documentation
        #     'docs/', '*.md', 'LICENSE', 'README*',
            
        #     # Docker
        #     'Dockerfile*', 'docker-compose*.yml', '.dockerignore',
            
        #     # Logs
        #     '*.log', 'logs/', '*.pid',
            
        #     # Temporary files
        #     'tmp/', 'temp/', '*.tmp', '*.bak', '*.backup',
            
        #     # Node.js (if applicable)
        #     'node_modules/', 'npm-debug.log*', 'yarn-debug.log*',
        #     'yarn-error.log*', '.npm', '.yarn/',
        # ]
        ignore_patterns = []
        
        # Add custom patterns
        if custom_patterns:
            ignore_patterns.extend(custom_patterns)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_patterns = []
        for pattern in ignore_patterns:
            if pattern not in seen:
                seen.add(pattern)
                unique_patterns.append(pattern)
        
        return '\n'.join(unique_patterns) + '\n'

    def _validate_and_enhance_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enhance the analysis data."""
        # Set defaults
        analysis.setdefault('repo_name', 'app')
        analysis.setdefault('python_version', '3.11')
        analysis.setdefault('python_packages', [])
        analysis.setdefault('system_packages', [])
        analysis.setdefault('environment_vars', {})
        analysis.setdefault('dependency_files', [])
         
        # Clean repo name
        analysis['repo_name'] = re.sub(r'[^a-zA-Z0-9_-]', '-', analysis['repo_name'].lower())
        
        # Validate Python version
        if not re.match(r'^\d+\.\d+(\.\d+)?$', analysis['python_version']):
            logger.warning("Invalid Python version %s, using 3.11", analysis['python_version'])
            analysis['python_version'] = '3.11'
        
        # Detect framework
        analysis['framework'] = self._detect_framework(analysis)
        
        return analysis

    def _select_optimal_template(self, analysis: Dict[str, Any]) -> BaseImageType:
        """Select the optimal base image template based on analysis."""
        packages = set(pkg.lower().split('==')[0] for pkg in analysis.get('python_packages', []))
        if packages & self.SCIENTIFIC_PACKAGES:
            return BaseImageType.SCIENTIFIC
        # Default to full
        return BaseImageType.FULL

    def _get_base_image(self, image_type: BaseImageType, python_version: str) -> str:
        """Get the base image string for the given type and Python version."""
        template = self.BASE_TEMPLATES[image_type]
        
        # codename = self._get_debian_codename(python_version)
        codename = ""
        # print("Codename:",codename)
        # print(template.format(version=python_version, codename=codename))
        return template.format(version=python_version, codename=codename)


    def _detect_package_manager(self, base_image: str) -> PackageManager:
        """Detect the package manager based on the base image (always APT in simplified version)."""
        return PackageManager.APT

    def _detect_framework(self, analysis: Dict[str, Any]) -> Optional[str]:
        """Detect the web framework being used."""
        repo_path = Path(analysis.get('repo_path', '.'))
        packages = set(pkg.lower().split('==')[0] for pkg in analysis.get('python_packages', []))
        
        for framework, patterns in self.WEB_FRAMEWORKS.items():
            # Check packages
            if any(pkg in packages for pkg in patterns['packages']):
                return framework
            
            # Check files
            if repo_path.exists():
                for file_pattern in patterns['files']:
                    if (repo_path / file_pattern).exists():
                        return framework
        
        return None

    # Node.js and binary dependency helpers removed (always not used)

    def _optimize_system_packages(self, packages: List[str]) -> List[str]:
        """Optimize system package list (dedupe & ensure essentials)."""
        # Remove duplicates and sort
        packages = sorted(set(packages))
        
        # Add essential packages based on package manager
        essentials = ['curl', 'ca-certificates', 'wget']
        for essential in essentials:
            if essential not in packages:
                packages.insert(0, essential)
        
        return packages

    def _prepare_template_variables(
        self,
        analysis: Dict[str, Any],
        base_image: str,
        base_image_type: BaseImageType,
        package_manager: PackageManager,
        include_tests: bool,
    ) -> Dict[str, Any]:
        
        # Filter dependency files to only include those that exist
        dependency_files = [
            f for f in analysis.get('dependency_files', [])
            if f and (Path(analysis.get('repo_path', '.')) / f).exists()
        ]
        
        # Aggregate all declared packages (core python_packages + extras_require values + test import install names)
        all_declared_packages = self._collect_all_declared_packages(
            analysis,
            include_tests,
            dependency_files,
            installable_test_imports=None  # placeholder, will fill after mapping below
        )  # we will regenerate after mapping

        # Build mapping of test imports to installable names (existing behavior)
        installable_test_imports = self._map_install_names(
            analysis.get('test_config', {}).get('extra_test_imports', [])
        )

        # Recompute with mapped test imports now
        all_declared_packages = self._collect_all_declared_packages(
            analysis,
            include_tests,
            dependency_files,
            installable_test_imports=installable_test_imports
        )

        return {
            'repo_name': analysis['repo_name'],
            'python_version': analysis['python_version'],
            'base_image': base_image,
            'base_image_type': base_image_type,
            'package_manager': package_manager,
            'PackageManager': PackageManager,  # For template access
            'needs_eol_debian_fix': self._needs_eol_debian_fix(analysis['python_version']),
            'system_packages': self._optimize_system_packages(
                analysis.get('system_packages', [])
            ),
            'environment_vars': analysis.get('environment_vars', {}),
            'dependency_files': dependency_files,
            'has_conda_env': 'environment.yml' in dependency_files or 'environment.yaml' in dependency_files,
            'add_nodejs': False,
            'frontend_build_commands': self._get_frontend_build_commands(analysis),
            # include_dev_deps removed
            'run_tests': include_tests,
            'test_commands': self._get_test_commands(analysis) if include_tests else [],
            'setuptools_extras': self._get_setuptools_extras(analysis, include_tests),
            'analysis_test_imports': analysis.get('test_config', {}).get('extra_test_imports', []),
            'installable_test_imports': installable_test_imports,
            'all_declared_packages': all_declared_packages,
        }

    def _collect_all_declared_packages(
        self,
        analysis: Dict[str, Any],
        include_tests: bool,
        dependency_files: List[str],
        installable_test_imports: Optional[List[str]] = None,
    ) -> List[str]:
        """Aggregate every declared package name/version we want to force-install.

        Order preference:
          1. python_packages (as discovered)
          2. extras_require flattened (tests/docs/lint/dev etc.)
          3. test imports mapped to installable names

        Duplicates removed preserving first occurrence. Empty / None skipped.
        """
        seen: Set[str] = set()
        ordered: List[str] = []

        # def _normalize_requirement(raw: str) -> str:
        #     """Normalize spaces around version specifiers (==,>=,<=,~=,!=,===,<,>). Preserve environment markers.

        #     Examples:
        #       'docutils == 0.15.2' -> 'docutils==0.15.2'
        #       'cftime >= 1.1.1' -> 'cftime>=1.1.1'
        #       'pkg >=1.0 ; python_version<"3.11"' -> 'pkg>=1.0; python_version<"3.11"'
        #     """
        #     if not raw:
        #         return raw
        #     s = raw.strip()
        #     # Split off environment marker part if present
        #     marker_part = ''
        #     if ';' in s:
        #         parts = s.split(';', 1)
        #         s, marker_part = parts[0].strip(), ';' + parts[1].strip()
        #     # Regex for name[extras] operator version
        #     # Allow operators: ==,>=,<=,~=,!=,===,<,>
        #     import re as _re
        #     m = _re.match(r'^([A-Za-z0-9_.-]+(?:\[[^\]]+\])?)\s*([!~<>=]{1,3})\s*([^\s]+)$', s)
        #     if m:
        #         name, op, ver = m.groups()
        #         s = f"{name}{op}{ver}"
        #     return s + (marker_part if marker_part else '')

        def _extract_package_name(raw: str) -> str:
            """Return only the package name (with extras if present), dropping versions and markers.
            
            Examples:
            'docutils == 0.15.2' -> 'docutils'
            'cftime >= 1.1.1' -> 'cftime'
            'pkg >=1.0 ; python_version<"3.11"' -> 'pkg'
            'uvicorn[standard]>=0.15.0' -> 'uvicorn[standard]'
            """
            if not raw:
                return raw.strip()

            s = raw.strip()
            # remove environment marker if any
            if ";" in s:
                s = s.split(";", 1)[0].strip()

            # capture name[extras] before any version specifier
            m = re.match(r'^([A-Za-z0-9_.-]+(?:\[[^\]]+\])?)', s)
            if m:
                return m.group(1)

            return s


        # def add(pkg: str):
        #     if not pkg:
        #         return
        #     normalized = _normalize_requirement(pkg)
        #     key = normalized.lower()
        #     if key not in seen:
        #         seen.add(key)
        #         ordered.append(normalized)
        def add(pkg: str):
            if not pkg:
                return
            name_only = _extract_package_name(pkg)
            key = name_only.lower()
            if key not in seen:
                seen.add(key)
                ordered.append(name_only)


        # 1. core python packages
        # for p in analysis.get('python_packages', []) or []:
        #     add(p)

        # 2. extras_require values
        for _extra, pkgs in (analysis.get('extras_require') or {}).items():
            for p in pkgs or []:
                add(p)

        # 3. test imports (mapped) if include_tests
        if include_tests and installable_test_imports:
            for p in installable_test_imports:
                add(p)

        return ordered

    def _get_frontend_build_commands(self, analysis: Dict[str, Any]) -> List[str]:
        """Get frontend build commands if applicable."""
        repo_path = Path(analysis.get('repo_path', '.'))
        commands = []
        
        if (repo_path / 'package.json').exists():
            if (repo_path / 'yarn.lock').exists():
                commands.append('yarn install --frozen-lockfile')
                commands.append('yarn build')
            else:
                commands.append('npm ci')
                commands.append('npm run build')
        
        return commands

    def _get_test_commands(self, analysis: Dict[str, Any]) -> List[str]:
        """Get test commands from analysis results."""
        # Use test commands already discovered during analysis
        test_config = analysis.get('test_config', {})
        test_commands = test_config.get('test_commands', [])
        # print(test_commands)
        if test_commands:
            return test_commands
        
        # Fallback to basic discovery if analysis didn't find any
        repo_path = Path(analysis.get('repo_path', '.'))
        packages = set(pkg.lower().split('==')[0] for pkg in analysis.get('python_packages', []))
        commands = []
        
        # Check for pytest in packages (with version specifiers)
        if any('pytest' in pkg.lower() for pkg in analysis.get('python_packages', [])):
            commands.append('python -m pytest')
        # Django tests
        elif analysis.get('framework') == 'django':
            commands.append('python manage.py test')
        # Unittest fallback
        elif (repo_path / 'tests').exists():
            commands.append('python -m unittest discover tests')
        
        return commands

    def _get_debian_codename(self, version: str) -> str:
        """Map Python version to correct Debian codename for full images."""
        try:
            major, minor, *_ = map(int, version.split("."))
        except Exception:
            return "bookworm"  # fallback
        
        if major == 2:
            return "jessie"
        elif major == 3 and minor <= 6:
            return "stretch"
        elif major == 3 and minor == 7:
            return "bullseye"
        else:  # 3.8+
            return "bookworm"

    def _needs_eol_debian_fix(self, python_version: str) -> bool:
        """Check if Python version requires EOL Debian repository fix."""
        try:
            major, minor, *_ = map(int, python_version.split("."))
            # Python versions that use Debian Stretch (EOL)
            return major == 3 and minor <= 6
        except Exception:
            return False

    def _get_setuptools_extras(self, analysis: Dict[str, Any], include_tests: bool) -> str:
        """Determine which setuptools extras to install based on analysis."""
        extras_require = analysis.get('extras_require', {})
        
        if not extras_require:
            return ""
        
        extras_to_install = []
        
        if include_tests:
            # Check for common test-related extras
            available_extras = set(extras_require.keys())
            
            # Priority order for test extras
            test_extras_priority = ['dev', 'test', 'tests', 'testing']
            
            for extra in test_extras_priority:
                if extra in available_extras:
                    extras_to_install.append(extra)
                    break
            
            # If no test extra found but we have individual packages, try to find them
            if not extras_to_install:
                # Look for extras that contain test-related packages
                test_packages = {'pytest', 'unittest2', 'nose', 'tox'}
                for extra_name, packages in extras_require.items():
                    if any(pkg.lower().split('==')[0].strip() in test_packages for pkg in packages):
                        extras_to_install.append(extra_name)
                        break
        
        # Add any extras that contain dependencies we detected
        python_packages = set(pkg.lower().split('==')[0].strip() for pkg in analysis.get('python_packages', []))
        for extra_name, packages in extras_require.items():
            extra_packages = set(pkg.lower().split('==')[0].strip() for pkg in packages)
            if extra_packages & python_packages and extra_name not in extras_to_install:
                extras_to_install.append(extra_name)
        
        if extras_to_install:
            return f"[{','.join(extras_to_install)}]"
        return ""

    def _map_install_names(self, names: List[str]) -> List[str]:
        """Robust mapping of import names to pip-installable distribution names.

        Strategy tiers (fast -> slower):
          1. Static mapping for known mismatches and meta-modules.
          2. Heuristic normalization (underscores->hyphens, case-folding).
          3. Optional PyPI existence probe (short timeout) to confirm candidate.
          4. Fallback filtering (drop obviously non-distributable namespace helpers).

        Network checks are best-effort; failures silently skip to next heuristic.
        """
        if not names:
            return []

        static_map = {
            # Imaging / graphics
            'pil': 'Pillow',
            'pillow': 'Pillow',
            # Computer vision
            'cv2': 'opencv-python',
            'opencv': 'opencv-python',
            # HTML / parsing
            'bs4': 'beautifulsoup4',
            # ML / data
            'sklearn': 'scikit-learn',
            'pytorch': 'torch',
            # Date / time
            'dateutil': 'python-dateutil',
            # GUI / graphics toolkits
            'wx': 'wxpython',
            'gi': 'pygobject',
            # Meta / internal namespaces to skip
            'mpl_toolkits': None,
            'pylab': None,
        }

        drop_prefixes = ('_pytest',)
        skip_exact = {
            '__future__', 'conftest', 'typing', 'dataclasses', 'pathlib', 'itertools',
            'functools', 'collections', 'os', 'sys', 're', 'json', 'logging'
        }

        def _pypi_exists(pkg: str) -> bool:
            url = f"https://pypi.org/pypi/{pkg}/json"
            try:
                req = _UrlRequest(url, headers={'User-Agent': 'repo-sandbox-generator'})
                with urlopen(req, timeout=2) as resp:  # small timeout
                    if resp.getcode() == 200:
                        # minimal parse to confirm JSON structure
                        _json.loads(resp.read(2000))  # read limited bytes
                        return True
            except Exception:
                return False
            return False

        resolved: List[str] = []
        seen: Set[str] = set()

        for raw in names:
            if not raw:
                continue
            name = raw.strip()
            low = name.lower()
            if low in skip_exact:
                continue
            if any(low.startswith(pref) for pref in drop_prefixes):
                continue

            # Tier 1: static map
            mapped = static_map.get(low)
            if mapped is None:  # explicit skip
                continue
            if mapped:
                candidate = mapped
            else:
                candidate = name

            # Tier 2: heuristic normalization if no static mapping
            if mapped is None:  # already skipped
                continue
            if mapped == name and '-' not in candidate:
                # try underscore->hyphen variant only if different
                hyph = candidate.replace('_', '-')
                if hyph != candidate and len(hyph) > 1:
                    candidate = hyph

            # Tier 3: optional existence probe (only if not obviously standard)
            # Avoid probing large number of packages (cap to 40)
            if len(resolved) < 40 and candidate.lower() not in seen:
                # If network check fails we still proceed (best-effort)
                _ = _pypi_exists(candidate)  # result not strictly enforced

            key = candidate.lower()
            if key not in seen:
                seen.add(key)
                resolved.append(candidate)

        return resolved
