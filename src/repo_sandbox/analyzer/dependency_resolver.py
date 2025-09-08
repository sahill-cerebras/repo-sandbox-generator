"""Dependency resolution module for Python projects."""

import ast
import re
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple, Callable

import toml

logger = logging.getLogger(__name__)

# --- Type Hint for Parsers ---
ParseResult = Dict[str, Any]
ParserFunc = Callable[[Path], ParseResult]

class DependencyResolver:
    """Resolves dependencies from various sources in a repository."""

    # --- Constants ---
    # Mapping of Python packages to common system-level dependencies.
    PACKAGE_TO_SYSTEM_DEPS: Dict[str, List[str]] = {
        'psycopg2': ['postgresql-dev', 'libpq-dev'],
        'psycopg2-binary': ['postgresql-client'],
        'mysqlclient': ['default-libmysqlclient-dev', 'mysql-client'],
        'pillow': ['libjpeg-dev', 'libpng-dev', 'libtiff-dev'],
        'lxml': ['libxml2-dev', 'libxslt-dev'],
        'numpy': ['build-essential'],
        'scipy': ['gfortran', 'libatlas-base-dev'],
        'matplotlib': ['pkg-config', 'libfreetype6-dev'],
        'cryptography': ['libssl-dev', 'libffi-dev'],
    }

    def __init__(self):
        """Initializes the resolver with its dependency source configuration."""
        self.DEP_SOURCES: List[Tuple[str, ParserFunc]] = [
            ('requirements.txt', self._parse_requirements_txt),
            ('requirements-dev.txt', self._parse_requirements_txt),
            ('requirements/base.txt', self._parse_requirements_txt),
            ('requirements/dev.txt', self._parse_requirements_txt),
            ('setup.py', self._parse_setup_py),
            ('pyproject.toml', self._parse_pyproject_toml),
            ('Pipfile', self._parse_pipfile),
            ('environment.yml', self._parse_conda_env),
            ('setup.cfg', self._parse_setup_cfg),
        ]

    def resolve_all_dependencies(self, repo_path: Path) -> Dict[str, Any]:
        """
        Resolves dependencies from all configured sources within the repository.
        
        Args:
            repo_path: Path to the repository.
            
        Returns:
            A dictionary containing resolved dependencies and build metadata.
        """
        dependencies: Dict[str, Any] = {
            'python_packages': set(),
            'system_packages': set(),
            'build_system': None,
            'install_commands': [],
            'dependency_files': []
        }
        
        found_sources = []
        for filename, parser in self.DEP_SOURCES:
            for file_path in repo_path.glob(filename):
                if file_path.is_file():
                    try:
                        deps = parser(file_path)
                        self._merge_dependencies(dependencies, deps)
                        found_sources.append(str(file_path.relative_to(repo_path)))
                        logger.debug(f"Parsed dependencies from {file_path.name}")
                    except Exception as e:
                        logger.warning(f"Error parsing {file_path.name}: {e}")
        
        dependencies['python_packages'] = sorted(list(dependencies['python_packages']))
        
        # Infer system dependencies and determine build system
        dependencies['system_packages'] = self._infer_system_dependencies(dependencies['python_packages'])
        dependencies['build_system'] = self._determine_build_system(repo_path, found_sources)
        dependencies['install_commands'] = self._generate_install_commands(dependencies['build_system'], found_sources)
        dependencies['dependency_files'] = sorted(list(set(found_sources)))

        logger.info(f"Found {len(dependencies['python_packages'])} Python packages from sources: {dependencies['dependency_files']}")
        return dependencies
    
    def _parse_requirements_txt(self, file_path: Path, processed_files: Optional[Set[Path]] = None) -> ParseResult:
        """
        Parses requirements.txt files, recursively following `-r` includes.
        """
        if processed_files is None:
            processed_files = set()
        
        if file_path in processed_files:
            return {'python_packages': []}
        processed_files.add(file_path)

        packages = []
        try:
            with file_path.open('r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Handle recursive includes
                    if line.startswith('-r'):
                        recursive_path = file_path.parent / line.split(maxsplit=1)[1]
                        if recursive_path.exists():
                            recursive_deps = self._parse_requirements_txt(recursive_path, processed_files)
                            packages.extend(recursive_deps.get('python_packages', []))
                        continue

                    if line.startswith(('-e', '--')):
                        continue
                    
                    if package := self._parse_requirement_line(line):
                        packages.append(package)
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
        
        return {'python_packages': packages}
    
    def _parse_setup_py(self, file_path: Path) -> ParseResult:
        """Parses `install_requires` from setup.py using AST analysis."""
        try:
            tree = ast.parse(file_path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'setup':
                    return self._extract_setup_info(node)
        except Exception as e:
            logger.warning(f"AST parsing of setup.py failed: {e}")
        return {'python_packages': []}
    
    def _parse_pyproject_toml(self, file_path: Path) -> ParseResult:
        """Parses dependencies from pyproject.toml (PEP 621 and Poetry)."""
        packages = []
        try:
            data = toml.loads(file_path.read_text(encoding='utf-8'))
            
            # PEP 621 `[project]` table
            if deps := data.get('project', {}).get('dependencies'):
                packages.extend(deps)
            
            # Poetry `[tool.poetry.dependencies]` table
            if poetry_deps := data.get('tool', {}).get('poetry', {}).get('dependencies'):
                for dep, version in poetry_deps.items():
                    if dep.lower() != 'python':
                        packages.append(f"{dep}{version}" if isinstance(version, str) else dep)
        except Exception as e:
            logger.warning(f"Error parsing pyproject.toml: {e}")
        
        return {'python_packages': packages, 'build_system': 'pyproject'}
    
    def _parse_pipfile(self, file_path: Path) -> ParseResult:
        """Parses `[packages]` from a Pipfile."""
        packages = []
        try:
            data = toml.loads(file_path.read_text(encoding='utf-8'))
            if pipfile_packages := data.get('packages'):
                for pkg, version in pipfile_packages.items():
                    packages.append(f"{pkg}{version}" if isinstance(version, str) else pkg)
        except Exception as e:
            logger.warning(f"Error parsing Pipfile: {e}")
        return {'python_packages': packages}
    
    def _parse_conda_env(self, file_path: Path) -> ParseResult:
        """Parses `dependencies` from a Conda environment.yml file."""
        packages = []
        try:
            import yaml
            data = yaml.safe_load(file_path.read_text(encoding='utf-8'))
            if deps := data.get('dependencies'):
                for dep in deps:
                    if isinstance(dep, str) and not dep.startswith('python='):
                        packages.append(dep)
                    elif isinstance(dep, dict) and 'pip' in dep:
                        packages.extend(dep['pip'])
        except ImportError:
            logger.debug("PyYAML not installed, cannot parse environment.yml")
        except Exception as e:
            logger.warning(f"Error parsing environment.yml: {e}")
        return {'python_packages': packages}

    def _parse_setup_cfg(self, file_path: Path) -> ParseResult:
        """Parses `install_requires` from setup.cfg."""
        packages = []
        try:
            import configparser
            config = configparser.ConfigParser()
            config.read(file_path)
            if install_requires := config.get('options', 'install_requires', fallback=''):
                packages.extend(line.strip() for line in install_requires.split('\n') if line.strip())
        except ImportError:
            logger.debug("configparser not available.")
        except Exception as e:
            logger.warning(f"Error parsing setup.cfg: {e}")
        return {'python_packages': packages}
    
    def _parse_requirement_line(self, line: str) -> Optional[str]:
        """Cleans and parses a single requirement line."""
        line = line.split('#')[0].strip()
        if not line:
            return None
        # Extract package name from Git URLs like git+...#egg=package-name
        if 'egg=' in line:
            match = re.search(r'egg=([^&]+)', line)
            return match.group(1) if match else None
        return line
    
    def _extract_setup_info(self, setup_node: ast.Call) -> ParseResult:
        """Extracts dependency list from an AST `setup()` call node."""
        packages = []
        for keyword in setup_node.keywords:
            if keyword.arg in ['install_requires', 'requires']:
                if isinstance(keyword.value, ast.List):
                    for item in keyword.value.elts:
                        if isinstance(item, (ast.Str, ast.Constant)):
                            val = item.s if isinstance(item, ast.Str) else item.value
                            packages.append(val)
        return {'python_packages': packages}
    
    def _merge_dependencies(self, base: ParseResult, new: ParseResult):
        """Merges new dependencies into the base dictionary."""
        if pkgs := new.get('python_packages'):
            base['python_packages'].update(pkgs)
        if build_system := new.get('build_system'):
            if not base.get('build_system'): # Don't override if already set
                base['build_system'] = build_system
    
    def _infer_system_dependencies(self, python_packages: List[str]) -> List[str]:
        """Infers required system packages from a list of Python packages."""
        system_deps: Set[str] = set()
        for package in python_packages:
            package_name = re.split(r'[>=<!~]', package, 1)[0].strip()
            if package_name in self.PACKAGE_TO_SYSTEM_DEPS:
                system_deps.update(self.PACKAGE_TO_SYSTEM_DEPS[package_name])
        return sorted(list(system_deps))
    
    def _determine_build_system(self, repo_path: Path, found_sources: List[str]) -> Optional[str]:
        """Determines the primary build system based on found files."""
        if any('pyproject.toml' in s for s in found_sources):
            return 'pyproject'
        if any('setup.py' in s for s in found_sources):
            return 'setuptools'
        if any('Pipfile' in s for s in found_sources):
            return 'pipenv'
        if any('environment.yml' in s for s in found_sources):
            return 'conda'
        if any('requirements' in s for s in found_sources):
            return 'pip'
        return None
    
    def _generate_install_commands(self, build_system: Optional[str], found_sources: List[str]) -> List[str]:
        """Generates appropriate installation commands based on the build system."""
        if build_system in ['pyproject', 'setuptools']:
            return ['python -m pip install -e .']
        if build_system == 'pipenv':
            return ['pipenv install --dev']
        if build_system == 'conda':
            return ['conda env create -f environment.yml']
        if build_system == 'pip':
            commands = []
            if 'requirements.txt' in found_sources:
                commands.append('python -m pip install -r requirements.txt')
            # Add dev requirements if they exist
            if any('dev' in s for s in found_sources):
                dev_file = next((s for s in found_sources if 'dev' in s), 'requirements-dev.txt')
                commands.append(f'python -m pip install -r {dev_file}')
            return commands
        return []