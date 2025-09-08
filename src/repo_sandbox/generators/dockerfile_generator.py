"""Optimized Dockerfile generation module with enhanced flow and accuracy."""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
from jinja2 import Template, Environment, BaseLoader
from enum import Enum
import re

logger = logging.getLogger(__name__)


class BaseImageType(Enum):
    """Enumeration for base image types."""
    SLIM = 'slim'
    FULL = 'full'
    ALPINE = 'alpine'
    SCIENTIFIC = 'scientific'
    CUDA = 'cuda'
    NODEJS = 'nodejs'


class PackageManager(Enum):
    """Enumeration for package managers."""
    APT = 'apt'
    APK = 'apk'
    YUM = 'yum'
    DNF = 'dnf'


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
        # Default: full Debian images
        BaseImageType.FULL: 'python:{version}-{codename}',

        # Optional lighter/minimal variants
        BaseImageType.SLIM: 'python:{version}-slim-{codename}',
        BaseImageType.ALPINE: 'python:{version}-alpine3.19',
        BaseImageType.SCIENTIFIC: 'python:{version}-{codename}',

        # Special cases
        BaseImageType.CUDA: 'nvidia/cuda:12.2.0-runtime-ubuntu22.04',
        BaseImageType.NODEJS: 'python:{version}-{codename}',
    }


    # Scientific packages for detection
    SCIENTIFIC_PACKAGES = {
        'numpy', 'scipy', 'matplotlib', 'pandas', 'scikit-learn',
        'tensorflow', 'torch', 'pytorch', 'keras', 'xgboost',
        'lightgbm', 'catboost', 'jupyterlab', 'notebook'
    }

    # Web framework detection patterns
    WEB_FRAMEWORKS = {
        'django': {'files': ['manage.py', 'settings.py'], 'packages': ['django']},
        'flask': {'files': ['app.py', 'application.py'], 'packages': ['flask']},
        'fastapi': {'files': ['main.py', 'app.py'], 'packages': ['fastapi']},
        'streamlit': {'files': ['app.py', 'streamlit_app.py'], 'packages': ['streamlit']},
        'gradio': {'files': ['app.py', 'interface.py'], 'packages': ['gradio']},
    }

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
# Install system dependencies with optimized layering
{%- if package_manager == PackageManager.APT %}
{%- set base_pkgs = system_packages or [] %}
{%- if not needs_eol_debian_fix %}
RUN apt-get update && \
    apt-get install -y --no-install-recommends {{ (base_pkgs + ['wget','curl','ca-certificates','build-essential','git','bzip2']) | join(' ') }} && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
{%- else %}
{%- if base_pkgs %}
RUN apt-get update && \
    apt-get install -y --no-install-recommends {{ base_pkgs | join(' ') }} && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
{%- endif %}
{%- endif %}
{%- elif package_manager == PackageManager.APK %}
RUN apk add --no-cache {{ system_packages|join(' ') }}
{%- else %}
RUN {{ package_manager.value }} install -y {{ system_packages|join(' ') }} && {{ package_manager.value }} clean all
{%- endif %}

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
RUN python -m pip install --upgrade pip setuptools wheel build

# -------------------------------
# Python dependencies (priority order)
# Prefers Conda environment.yml when present (more reproducible for complex stacks)
# -------------------------------
{%- if 'environment.yml' in dependency_files or 'environment.yaml' in dependency_files %}
# --- Conda environment installation (preferred when provided) ---
{%- set env_file = 'environment.yml' if 'environment.yml' in dependency_files else 'environment.yaml' %}
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    touch /opt/conda/.conda_installed
ENV PATH="/opt/conda/bin:$PATH"

# Accept Anaconda TOS for default channels (best-effort; harmless if unavailable)
RUN conda init bash || true && \
    conda config --add channels defaults || true && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

COPY {{ env_file }} /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml -n app && conda clean -a -y && echo "Activated conda env 'app'" && \
    echo "dependencies captured from {{ env_file }}" && \
    true
ENV PATH="/opt/conda/envs/app/bin:$PATH"

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
RUN pip install --upgrade pip setuptools wheel build cython || true && \
    if grep -q "\\[tool.poetry\\]" pyproject.toml 2>/dev/null; then \
        pip install poetry && \
        poetry config virtualenvs.create false && \
        poetry install --no-dev; \
    elif grep -q "\\[tool.flit\\]" pyproject.toml 2>/dev/null || grep -q "flit_core" pyproject.toml 2>/dev/null; then \
        pip install flit && \
        python -m build --wheel && \
        pip install --no-cache-dir dist/*.whl; \
    else \
        pip install --no-cache-dir -e .[test] 2>/dev/null || \
        pip install --no-cache-dir -e . 2>/dev/null || \
        (python -m build --wheel && pip install --no-cache-dir dist/*.whl); \
    fi
WORKDIR /app

{%- elif 'setup.py' in dependency_files or 'setup.cfg' in dependency_files %}
# Classic setuptools-based project
COPY . /tmp/project
WORKDIR /tmp/project
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .{{ setuptools_extras }}
WORKDIR /app

{%- else %}
RUN echo "No dependency file found; skipping Python package installation"
{%- endif %}

# Copy application code (always copy after dependency installs to leverage Docker cache)
COPY . .

{%- if frontend_build_commands %}
RUN {{ frontend_build_commands|join(' && ') }}
{%- endif %}

{%- if run_tests and test_commands %}
# Install testing libraries detected from test commands
{%- set test_libs = ['pytest'] %}
{%- for command in test_commands %}
{%- set parts = command.split() %}
{%- if parts|length > 2 and parts[0] == 'python' and parts[1] == '-m' %}
{%- set _ = test_libs.append(parts[2]) %}
{%- else %}
{%- set _ = test_libs.append(parts[0]) %}
{%- endif %}
{%- endfor %}
RUN python -m pip install --upgrade pip setuptools wheel && pip install --no-cache-dir {{ test_libs | unique | join(' ') }}
{%- endif %}

# -------------------------------------------------------------------------
# Ensure the project is installed into the active environment before testing
# -------------------------------------------------------------------------
{%- if 'setup.py' in dependency_files or 'setup.cfg' in dependency_files %}
RUN python -m pip install --upgrade pip setuptools wheel build cython || true && \
    pip install --no-cache-dir -e .{{ setuptools_extras }}
{%- elif 'pyproject.toml' in dependency_files %}
RUN python -m pip install --upgrade pip setuptools wheel build cython || true && \
    (pip install --no-cache-dir -e .[test] 2>/dev/null || \
     pip install --no-cache-dir -e . 2>/dev/null || \
     pip install --no-cache-dir . 2>/dev/null || \
     (python -m build --wheel && pip install --no-cache-dir dist/*.whl) || true)
{%- else %}
RUN echo "No setup.py/pyproject.toml found, skipping editable install"
{%- endif %}

{%- if run_tests and test_commands %}
RUN if [ -f /opt/conda/.conda_installed ]; then \
        /opt/conda/bin/conda run -n app {{ test_commands|join(' && /opt/conda/bin/conda run -n app ') }}; \
    else \
        {{ test_commands|join(' && ') }}; \
    fi
{%- endif %}


{%- if create_user %}
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser
{%- endif %}

{%- if health_check %}
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD {{ health_check }}
{%- endif %}

{%- if expose_ports %}
{%- for port in expose_ports %}
EXPOSE {{ port }}
{%- endfor %}
{%- endif %}

{%- if volumes %}
{%- for volume in volumes %}
VOLUME {{ volume }}
{%- endfor %}
{%- endif %}

{%- if entrypoint %}
ENTRYPOINT {{ entrypoint }}
{%- endif %}
{{ cmd }}"""





    def __init__(self):
        """Initialize the Dockerfile generator."""
        self.env = Environment(loader=BaseLoader())
        self.template = self.env.from_string(self.DOCKERFILE_TEMPLATE)

    def generate_dockerfile(
        self,
        analysis: Dict[str, Any],
        template: str = 'auto',
        include_tests: bool = False,
        include_dev_deps: bool = False,
        create_user: bool = True,
        optimize_size: bool = True
    ) -> str:
        """
        Generate an optimized Dockerfile based on repository analysis.
        
        Args:
            analysis: Repository analysis results
            template: Base image template to use
            include_tests: Whether to run tests during build
            include_dev_deps: Whether to include development dependencies
            create_user: Whether to create a non-root user
            optimize_size: Whether to optimize for smaller image size
            
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
            include_tests, include_dev_deps, create_user, optimize_size
        )
        
        # Generate Dockerfile
        dockerfile_content = self.template.render(**template_vars)
        
        # Post-process for optimization
        if optimize_size:
            dockerfile_content = self._optimize_dockerfile_size(dockerfile_content)
        
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
        analysis.setdefault('expose_ports', [])
        analysis.setdefault('volumes', [])
        
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
        
        # Check for CUDA/GPU requirements
        if any(pkg in ['tensorflow-gpu', 'torch-gpu', 'jax[cuda]'] for pkg in packages):
            return BaseImageType.CUDA
        
        # Check for scientific packages
        if packages & self.SCIENTIFIC_PACKAGES:
            return BaseImageType.SCIENTIFIC
        
        # Check for Node.js requirements
        if self._requires_nodejs(analysis):
            return BaseImageType.NODEJS
        
        # Alpine for minimal deployments
        if analysis.get('optimize_size') and not self._has_binary_dependencies(packages):
            return BaseImageType.ALPINE
        
        # Default to full
        return BaseImageType.FULL

    def _get_base_image(self, image_type: BaseImageType, python_version: str) -> str:
        """Get the base image string for the given type and Python version."""
        template = self.BASE_TEMPLATES[image_type]
        
        # Special handling for CUDA images
        if image_type == BaseImageType.CUDA:
            return template  # CUDA images don't use Python version placeholder
        
        codename = self._get_debian_codename(python_version)
        print("Codename:",codename)
        print(template.format(version=python_version, codename=codename))
        return template.format(version=python_version, codename=codename)


    def _detect_package_manager(self, base_image: str) -> PackageManager:
        """Detect the package manager based on the base image."""
        if 'alpine' in base_image:
            return PackageManager.APK
        elif any(distro in base_image for distro in ['centos', 'rhel', 'fedora']):
            return PackageManager.DNF if 'fedora' in base_image else PackageManager.YUM
        else:
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

    def _requires_nodejs(self, analysis: Dict[str, Any]) -> bool:
        """Check if the project requires Node.js."""
        repo_path = Path(analysis.get('repo_path', '.'))
        
        # Check for Node.js files
        node_files = ['package.json', 'yarn.lock', 'package-lock.json']
        return any((repo_path / f).exists() for f in node_files)

    def _has_binary_dependencies(self, packages: Set[str]) -> bool:
        """Check if packages have binary dependencies that might not work with Alpine."""
        binary_packages = {
            'numpy', 'scipy', 'pandas', 'pillow', 'psycopg2',
            'mysqlclient', 'lxml', 'cryptography', 'grpcio'
        }
        return bool(packages & binary_packages)

    def _optimize_system_packages(self, packages: List[str], package_manager: PackageManager) -> List[str]:
        """Optimize system package list."""
        # Remove duplicates and sort
        packages = sorted(set(packages))
        
        # Add essential packages based on package manager
        essentials = {
            PackageManager.APT: ['curl', 'ca-certificates'],
            PackageManager.APK: ['curl', 'ca-certificates'],
            PackageManager.YUM: ['curl', 'ca-certificates'],
            PackageManager.DNF: ['curl', 'ca-certificates'],
        }

        for pkg_mgr in essentials:
            if 'wget' not in essentials[pkg_mgr]:
                essentials[pkg_mgr].append('wget')

        for essential in essentials.get(package_manager, []):
            if essential not in packages:
                packages.insert(0, essential)
        
        return packages

    def _infer_start_command(self, analysis: Dict[str, Any]) -> Tuple[Optional[str], str]:
        """
        Infer the start command and entrypoint.
        
        Returns:
            Tuple of (entrypoint, cmd)
        """
        repo_path = Path(analysis.get('repo_path', '.'))
        framework = analysis.get('framework')
        packages = set(pkg.lower().split('==')[0] for pkg in analysis.get('python_packages', []))
        
        # Framework-specific commands
        if framework == 'django':
            if (repo_path / 'manage.py').exists():
                return None, 'CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]'
        
        elif framework == 'flask':
            if (repo_path / 'app.py').exists():
                if 'gunicorn' in packages:
                    return None, 'CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]'
                return None, 'CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]'
        
        elif framework == 'fastapi':
            app_file = None
            for f in ['main.py', 'app.py']:
                if (repo_path / f).exists():
                    app_file = f.replace('.py', '')
                    break
            if app_file:
                return None, f'CMD ["uvicorn", "{app_file}:app", "--host", "0.0.0.0", "--port", "8000"]'
        
        elif framework == 'streamlit':
            app_file = 'streamlit_app.py' if (repo_path / 'streamlit_app.py').exists() else 'app.py'
            return None, f'CMD ["streamlit", "run", "{app_file}", "--server.port=8501", "--server.address=0.0.0.0"]'
        
        elif framework == 'gradio':
            return None, 'CMD ["python", "app.py"]'
        
        # Check for other common patterns
        if (repo_path / 'main.py').exists():
            return None, 'CMD ["python", "main.py"]'
        
        if (repo_path / '__main__.py').exists():
            return None, f'CMD ["python", "-m", "{analysis["repo_name"]}"]'
        
        # Default fallback
        return None, 'CMD ["python"]'

    def _get_health_check_command(self, analysis: Dict[str, Any]) -> Optional[str]:
        """Generate appropriate health check command."""
        framework = analysis.get('framework')
        ports = analysis.get('expose_ports', [])
        
        if not ports:
            return None
        
        port = ports[0]
        
        if framework in ['django', 'flask', 'fastapi']:
            return f'curl -f http://localhost:{port}/health || exit 1'
        elif framework == 'streamlit':
            return f'curl -f http://localhost:{port}/_stcore/health || exit 1'
        
        return f'curl -f http://localhost:{port}/ || exit 1'

    def _prepare_template_variables(
        self,
        analysis: Dict[str, Any],
        base_image: str,
        base_image_type: BaseImageType,
        package_manager: PackageManager,
        include_tests: bool,
        include_dev_deps: bool,
        create_user: bool,
        optimize_size: bool
    ) -> Dict[str, Any]:
        """Prepare all template variables."""
        entrypoint, cmd = self._infer_start_command(analysis)
        
        # Filter dependency files to only include those that exist
        dependency_files = [
            f for f in analysis.get('dependency_files', [])
            if f and (Path(analysis.get('repo_path', '.')) / f).exists()
        ]
        
        return {
            'repo_name': analysis['repo_name'],
            'python_version': analysis['python_version'],
            'base_image': base_image,
            'base_image_type': base_image_type,
            'package_manager': package_manager,
            'PackageManager': PackageManager,  # For template access
            'needs_eol_debian_fix': self._needs_eol_debian_fix(analysis['python_version']),
            'system_packages': self._optimize_system_packages(
                analysis.get('system_packages', []), package_manager
            ),
            'environment_vars': analysis.get('environment_vars', {}),
            'dependency_files': dependency_files,
            'has_conda_env': 'environment.yml' in dependency_files or 'environment.yaml' in dependency_files,
            'add_nodejs': self._requires_nodejs(analysis),
            'frontend_build_commands': self._get_frontend_build_commands(analysis),
            'include_dev_deps': include_dev_deps,
            'run_tests': include_tests,
            'test_commands': self._get_test_commands(analysis) if include_tests else [],
            'create_user': create_user,
            'health_check': self._get_health_check_command(analysis),
            'expose_ports': analysis.get('expose_ports', []),
            'volumes': analysis.get('volumes', []),
            'entrypoint': entrypoint,
            'cmd': cmd,
            'setuptools_extras': self._get_setuptools_extras(analysis, include_tests),
        }

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

    def _optimize_dockerfile_size(self, dockerfile: str) -> str:
        """Apply size optimization techniques to the Dockerfile."""
        # Combine consecutive RUN commands where appropriate
        # This is a simplified optimization - in practice, you might want more sophisticated logic
        lines = dockerfile.split('\n')
        optimized = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Combine consecutive RUN commands for package installation
            if line.startswith('RUN') and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith('RUN') and ('apt-get' in line or 'apk' in line):
                    # Combine the commands
                    combined = line + ' && \\\n    ' + next_line[4:]
                    optimized.append(combined)
                    i += 2
                    continue
            
            optimized.append(lines[i])
            i += 1
        
        return '\n'.join(optimized)
    
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



# Additional utility functions
def validate_dockerfile(dockerfile_content: str) -> List[str]:
    """
    Validate Dockerfile content for common issues.
    
    Args:
        dockerfile_content: The Dockerfile content to validate
        
    Returns:
        List of validation warnings/errors
    """
    issues = []
    lines = dockerfile_content.split('\n')
    
    # Check for multiple FROM statements (except multi-stage)
    from_count = sum(1 for line in lines if line.strip().startswith('FROM'))
    if from_count > 1:
        issues.append("Multiple FROM statements detected - ensure multi-stage build is intentional")
    
    # Check for missing WORKDIR
    if not any('WORKDIR' in line for line in lines):
        issues.append("No WORKDIR specified - consider setting a working directory")
    
    # Check for running as root
    if not any('USER' in line for line in lines):
        issues.append("Running as root user - consider creating a non-root user for security")
    
    # Check for missing health check
    if not any('HEALTHCHECK' in line for line in lines):
        issues.append("No HEALTHCHECK defined - consider adding health check for production")
    
    return issues


def generate_compose_file(analysis: Dict[str, Any], dockerfile_path: str = './Dockerfile') -> str:
    """
    Generate a docker-compose.yml file for the project.
    
    Args:
        analysis: Repository analysis results
        dockerfile_path: Path to the Dockerfile
        
    Returns:
        docker-compose.yml content
    """
    repo_name = analysis.get('repo_name', 'app')
    ports = analysis.get('expose_ports', [])
    volumes = analysis.get('volumes', [])
    env_vars = analysis.get('environment_vars', {})
    
    compose_content = f"""version: '3.8'

services:
  {repo_name}:
    build:
      context: .
      dockerfile: {dockerfile_path}
    image: {repo_name}:latest
    container_name: {repo_name}
"""
    
    if ports:
        compose_content += "    ports:\n"
        for port in ports:
            compose_content += f"      - \"{port}:{port}\"\n"
    
    if volumes:
        compose_content += "    volumes:\n"
        for volume in volumes:
            compose_content += f"      - {volume}\n"
    
    if env_vars:
        compose_content += "    environment:\n"
        for key, value in env_vars.items():
            compose_content += f"      {key}: \"{value}\"\n"
    
    compose_content += """    restart: unless-stopped
    networks:
      - app-network

networks:
  app-network:
    driver: bridge
"""
    
    return compose_content

