"""Enhanced Python version detection module with improved accuracy and caching."""

import re
import logging
import functools
from pathlib import Path
from typing import Optional, List, Tuple, Set, Union
from dataclasses import dataclass
from enum import Enum
import ast
from pathlib import Path
from typing import Optional
import logging
import toml
from packaging.specifiers import SpecifierSet

logger = logging.getLogger(__name__)


class VersionSource(Enum):
    """Enumeration of version source priorities."""
    PYTHON_VERSION_FILE = 1
    RUNTIME_TXT = 2
    PYPROJECT_TOML = 3
    SETUP_PY = 4
    SETUP_CFG = 5
    DOCKERFILE = 6
    GITHUB_ACTIONS = 7
    TOX_INI = 8
    ENVIRONMENT_YML = 9
    README = 10
    REQUIREMENTS_COMMENT = 11


@dataclass
class PythonVersion:
    """Structured representation of a Python version."""
    major: int
    minor: int
    patch: Optional[int] = None
    source: Optional[VersionSource] = None
    raw: Optional[str] = None
    
    def __str__(self) -> str:
        if self.patch is not None and self.patch >= 0:
            return f"{self.major}.{self.minor}.{self.patch}"
        return f"{self.major}.{self.minor}"
    
    def __lt__(self, other: 'PythonVersion') -> bool:
        return self.as_tuple() < other.as_tuple()
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PythonVersion):
            return False
        return self.as_tuple() == other.as_tuple()
    
    def as_tuple(self) -> Tuple[int, int, int]:
        """Convert to comparable tuple."""
        return (self.major, self.minor, self.patch if self.patch is not None else -1)
    
    @classmethod
    def parse(cls, version_str: str, source: Optional[VersionSource] = None) -> Optional['PythonVersion']:
        """Parse version string into PythonVersion object."""
        if not version_str:
            return None
        
        # Handle wildcards
        version_str = version_str.replace('.*', '')
        
        # Extract version numbers
        match = re.match(r'^(\d+)\.(\d+)(?:\.(\d+))?', version_str.strip())
        if not match:
            return None
        
        major = int(match.group(1))
        minor = int(match.group(2))
        patch = int(match.group(3)) if match.group(3) else None
        
        return cls(major=major, minor=minor, patch=patch, source=source, raw=version_str)


class PythonVersionDetector:
    """Optimized Python version detector with caching and parallel processing."""
    
    # Default version to use when none detected
    DEFAULT_VERSION = "3.11"
    
    # Cache for parsed files to avoid re-reading
    _file_cache: dict = {}
    
    def __init__(self, cache_enabled: bool = True):
        """Initialize detector with optional caching."""
        self.cache_enabled = cache_enabled
        if not cache_enabled:
            self._file_cache.clear()
    
    def detect_python_version(self, repo_path: Path, prefer_latest: bool = True) -> str:
        """
        Detect Python version from multiple sources with optimized performance.
        
        Args:
            repo_path: Path to repository
            prefer_latest: If True, return highest version; else return most authoritative
            
        Returns:
            Detected Python version string
        """
        repo_path = Path(repo_path)
        if not repo_path.exists():
            logger.error(f"Repository path does not exist: {repo_path}")
            return self.DEFAULT_VERSION
        
        detected_versions: List[PythonVersion] = []
        
        # Priority-ordered detection methods
        detection_methods = [
            (VersionSource.PYTHON_VERSION_FILE, '.python-version', self._parse_python_version_file),
            (VersionSource.RUNTIME_TXT, 'runtime.txt', self._parse_runtime_txt),
            (VersionSource.PYPROJECT_TOML, 'pyproject.toml', self._parse_pyproject_python_version),
            (VersionSource.SETUP_PY, 'setup.py', self._parse_setup_py_python_version),
            (VersionSource.SETUP_CFG, 'setup.cfg', self._parse_setup_cfg_python_version),
            (VersionSource.DOCKERFILE, 'Dockerfile', self._parse_dockerfile_python),
            (VersionSource.TOX_INI, 'tox.ini', self._parse_tox_python_versions),
            (VersionSource.ENVIRONMENT_YML, 'environment.yml', self._parse_environment_yml_python_version),
        ]
        
        # Process files with caching
        for source, filename, parser in detection_methods:
            file_path = repo_path / filename
            if file_path.exists():
                versions = self._parse_with_cache(file_path, parser, source)
                if versions:
                    detected_versions.extend(versions)
        
        # Check GitHub Actions
        github_versions = self._parse_github_actions_python(repo_path / '.github/workflows')
        if github_versions:
            detected_versions.extend(github_versions)
        
        # Check README files
        for readme in ['README.md', 'README.rst', 'README.txt']:
            readme_path = repo_path / readme
            if readme_path.exists():
                readme_versions = self._parse_with_cache(
                    readme_path, self._parse_readme_python_version, VersionSource.README
                )
                if readme_versions:
                    detected_versions.extend(readme_versions)
        
        # Check requirements files for comments
        for req_file in repo_path.glob("requirements*.txt"):
            req_versions = self._parse_with_cache(
                req_file, self._parse_requirements_for_python_version, 
                VersionSource.REQUIREMENTS_COMMENT
            )
            if req_versions:
                detected_versions.extend(req_versions)
        
        # Select best version
        # print(detected_versions)
        if detected_versions:
            if prefer_latest:
                best = max(detected_versions)
            else:
                # Prefer most authoritative source
                best = min(detected_versions, key=lambda v: v.source.value if v.source else 999)
            
            logger.info(f"Detected Python version {best} from {best.source.name if best.source else 'unknown'}")
            return str(best)
        
        logger.info(f"No Python version detected, using default: {self.DEFAULT_VERSION}")
        return self.DEFAULT_VERSION
    
    def _parse_with_cache(self, file_path: Path, parser: callable, 
                         source: VersionSource) -> List[PythonVersion]:
        """Parse file with caching support."""
        cache_key = str(file_path)
        
        if self.cache_enabled and cache_key in self._file_cache:
            return self._file_cache[cache_key]
        
        try:
            result = parser(file_path)
            versions = []
            
            if result:
                if isinstance(result, list):
                    for r in result:
                        if isinstance(r, str):
                            v = PythonVersion.parse(r, source)
                            if v:
                                versions.append(v)
                else:
                    v = PythonVersion.parse(result, source)
                    if v:
                        versions.append(v)
            
            if self.cache_enabled:
                self._file_cache[cache_key] = versions
            
            return versions
            
        except Exception as e:
            logger.debug(f"Error parsing {file_path.name}: {e}")
            return []
    
    def _read_file_safely(self, file_path: Path, encoding: str = 'utf-8', 
                         errors: str = 'ignore', max_size: int = 10_000_000) -> Optional[str]:
        """Safely read file with size limit and error handling."""
        try:
            if file_path.stat().st_size > max_size:
                logger.warning(f"File {file_path} exceeds size limit, skipping")
                return None
            return file_path.read_text(encoding=encoding, errors=errors)
        except Exception as e:
            logger.debug(f"Error reading {file_path}: {e}")
            return None
    
    def _parse_python_version_file(self, file_path: Path) -> Optional[str]:
        """Parse .python-version file."""
        content = self._read_file_safely(file_path)
        if content:
            return self._extract_version_from_requirement(content.strip())
        return None
    
    def _parse_runtime_txt(self, file_path: Path) -> Optional[str]:
        """Parse runtime.txt file."""
        content = self._read_file_safely(file_path)
        if content:
            # Handle Heroku format: python-3.11.0
            content = content.strip().replace('python-', '')
            return self._extract_version_from_requirement(content)
        return None
    
    def _parse_pyproject_python_version(self, file_path: Path) -> Optional[str]:
        """
        Parse a pyproject.toml file to determine the Python version requirement.

        Supports:
        - PEP 621 (`project.requires-python`)
        - Poetry (`tool.poetry.dependencies.python`)
        - PDM (`tool.pdm.requires-python`)
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            if not content.strip():
                return None
        except Exception as e:
            logger.debug(f"Failed to read {file_path}: {e}")
            return None

        try:
            data = toml.loads(content)
        except Exception as e:
            logger.debug(f"Error parsing TOML from {file_path}: {e}")
            return None

        # Helper function
        def extract_version(req: str) -> Optional[str]:
            try:
                # Use packaging to parse specifiers
                spec = SpecifierSet(req)
                # Return the first minimum version if possible
                for s in spec:
                    if s.operator in (">=", "=="):
                        return s.version
                return None
            except Exception:
                return None

        # PEP 621
        requires_python = data.get("project", {}).get("requires-python")
        if requires_python:
            version = extract_version(requires_python)
            if version:
                return version

        # Poetry
        poetry_python = data.get("tool", {}).get("poetry", {}).get("dependencies", {}).get("python")
        if poetry_python:
            version = extract_version(poetry_python)
            if version:
                return version

        # PDM
        pdm_python = data.get("tool", {}).get("pdm", {}).get("requires-python")
        if pdm_python:
            version = extract_version(pdm_python)
            if version:
                return version

        return None

    
    def _parse_setup_py_python_version(self, file_path: Path) -> Optional[str]:
        """Parse setup.py using AST for Python version."""
        content = self._read_file_safely(file_path)
        if not content:
            return None
        
        try:
            tree = ast.parse(content)
            
            class SetupVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.python_requires = None
                
                def visit_Call(self, node):
                    if (hasattr(node.func, 'id') and node.func.id == 'setup'):
                        for keyword in node.keywords:
                            if keyword.arg == 'python_requires':
                                if isinstance(keyword.value, ast.Constant):
                                    self.python_requires = keyword.value.value
                                elif isinstance(keyword.value, ast.Str):
                                    self.python_requires = keyword.value.s
                    self.generic_visit(node)
            
            visitor = SetupVisitor()
            visitor.visit(tree)
            
            if visitor.python_requires:
                return self._extract_version_from_requirement(visitor.python_requires)
                
        except Exception as e:
            logger.debug(f"Error parsing setup.py: {e}")
        
        return None
    
    def _parse_setup_cfg_python_version(self, file_path: Path) -> Optional[str]:
        """Parse setup.cfg for Python version."""
        content = self._read_file_safely(file_path)
        if not content:
            return None
        
        try:
            import configparser
            config = configparser.ConfigParser()
            config.read_string(content)
            
            if config.has_option('options', 'python_requires'):
                requires = config.get('options', 'python_requires')
                return self._extract_version_from_requirement(requires)
                
        except Exception as e:
            logger.debug(f"Error parsing setup.cfg: {e}")
        
        return None
    
    def _parse_tox_python_versions(self, file_path: Path) -> List[str]:
        """Parse tox.ini for Python versions."""
        content = self._read_file_safely(file_path)
        if not content:
            return []
        
        try:
            import configparser
            config = configparser.ConfigParser()
            config.read_string(content)
            
            versions = []
            
            # Check envlist
            if config.has_option('tox', 'envlist'):
                envlist = config.get('tox', 'envlist')
                # Extract py38, py39, py310, py311, etc.
                for match in re.finditer(r'py(\d)(\d+)', envlist):
                    major = match.group(1)
                    minor = match.group(2)
                    versions.append(f"{major}.{minor}")
            
            # Check basepython in testenv sections
            for section in config.sections():
                if section.startswith('testenv'):
                    if config.has_option(section, 'basepython'):
                        basepython = config.get(section, 'basepython')
                        version = self._extract_version_from_requirement(basepython)
                        if version:
                            versions.append(version)
            
            return versions
            
        except Exception as e:
            logger.debug(f"Error parsing tox.ini: {e}")
        
        return []
    
    def _parse_github_actions_python(self, workflows_dir: Path) -> List[PythonVersion]:
        """Parse GitHub Actions workflows for Python versions."""
        if not workflows_dir.exists():
            return []
        
        versions = []
        
        try:
            import yaml
            
            for workflow_file in workflows_dir.glob('*.y*ml'):
                content = self._read_file_safely(workflow_file)
                if not content:
                    continue
                
                try:
                    data = yaml.safe_load(content)
                    if not data or not isinstance(data, dict):
                        continue
                    
                    # Check all jobs
                    for job in (data.get('jobs', {}) or {}).values():
                        if not isinstance(job, dict):
                            continue
                        
                        # Check strategy matrix
                        strategy = job.get('strategy', {})
                        if isinstance(strategy, dict):
                            matrix = strategy.get('matrix', {})
                            if isinstance(matrix, dict):
                                python_versions = matrix.get('python-version', [])
                                if isinstance(python_versions, list):
                                    for v in python_versions:
                                        pv = PythonVersion.parse(str(v), VersionSource.GITHUB_ACTIONS)
                                        if pv:
                                            versions.append(pv)
                        
                        # Check steps
                        for step in job.get('steps', []):
                            if not isinstance(step, dict):
                                continue
                            
                            if 'uses' in step and 'setup-python' in str(step['uses']):
                                with_block = step.get('with', {})
                                if isinstance(with_block, dict):
                                    py_ver = with_block.get('python-version')
                                    if py_ver:
                                        pv = PythonVersion.parse(str(py_ver), VersionSource.GITHUB_ACTIONS)
                                        if pv:
                                            versions.append(pv)
                                            
                except yaml.YAMLError as e:
                    logger.debug(f"Error parsing YAML in {workflow_file}: {e}")
                    
        except ImportError:
            logger.debug("PyYAML not available, skipping GitHub Actions parsing")
        except Exception as e:
            logger.debug(f"Error parsing GitHub Actions: {e}")
        
        return versions
    
    def _parse_dockerfile_python(self, file_path: Path) -> Optional[str]:
        """Parse Dockerfile for Python version."""
        content = self._read_file_safely(file_path)
        if not content:
            return None
        
        # Look for FROM python:X.Y.Z patterns
        patterns = [
            r'FROM\s+python:(\d+\.\d+(?:\.\d+)?)',
            r'FROM\s+.*python.*:(\d+\.\d+(?:\.\d+)?)',
            r'ARG\s+PYTHON_VERSION=(\d+\.\d+(?:\.\d+)?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1)
        
        return None
    
    def _parse_environment_yml_python_version(self, file_path: Path) -> Optional[str]:
        """Parse environment.yml for Python version."""
        content = self._read_file_safely(file_path)
        if not content:
            return None
        
        # Look for python=X.Y.Z or python==X.Y.Z
        matches = re.findall(r'^\s*-?\s*python[=]{1,2}(\d+\.\d+(?:\.\d+)?)', 
                           content, re.MULTILINE | re.IGNORECASE)
        if matches:
            return max(matches, key=lambda v: PythonVersion.parse(v).as_tuple() if PythonVersion.parse(v) else (0, 0, 0))
        
        return None
    
    def _parse_readme_python_version(self, file_path: Path) -> Optional[str]:
        """Parse README for Python version mentions."""
        content = self._read_file_safely(file_path, max_size=1_000_000)  # Limit README size
        if not content:
            return None
        
        # Look for Python version patterns
        patterns = [
            r'Python\s+(\d+\.\d+(?:\.\d+)?)\+?',
            r'Python\s+version\s+(\d+\.\d+(?:\.\d+)?)',
            r'requires\s+Python\s+(\d+\.\d+(?:\.\d+)?)',
            r'Python:\s*(\d+\.\d+(?:\.\d+)?)',
        ]
        
        versions = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            versions.extend(matches)
        
        if versions:
            # Return the most frequently mentioned or highest version
            from collections import Counter
            version_counts = Counter(versions)
            return version_counts.most_common(1)[0][0]
        
        return None
    
    def _parse_requirements_for_python_version(self, file_path: Path) -> Optional[str]:
        """Look for Python version hints in requirements file comments."""
        content = self._read_file_safely(file_path)
        if not content:
            return None
        
        lines = content.splitlines()[:100]  # Check first 100 lines
        
        for line in lines:
            if line.strip().startswith('#'):
                # Look for version patterns in comments
                match = re.search(r'python[:\s]+(\d+\.\d+(?:\.\d+)?)', line, re.IGNORECASE)
                if match:
                    return match.group(1)
        
        return None
    
    def _extract_version_from_requirement(self, requirement: str) -> Optional[str]:
        """Extract version from requirement specifier."""
        if not requirement:
            return None
        
        requirement = requirement.strip()
        
        # Handle exact versions
        exact_match = re.search(r'==\s*(\d+\.\d+(?:\.\d+)?)', requirement)
        if exact_match:
            return exact_match.group(1)
        
        # Handle minimum versions
        min_match = re.search(r'>=\s*(\d+\.\d+(?:\.\d+)?)', requirement)
        if min_match:
            return min_match.group(1)
        
        # Handle compatible versions
        compat_match = re.search(r'~=\s*(\d+\.\d+(?:\.\d+)?)', requirement)
        if compat_match:
            return compat_match.group(1)
        
        # Handle wildcards
        wild_match = re.search(r'(\d+\.\d+)\.\*', requirement)
        if wild_match:
            return wild_match.group(1)
        
        # Handle plain versions
        plain_match = re.search(r'(\d+\.\d+(?:\.\d+)?)', requirement)
        if plain_match:
            return plain_match.group(1)
        
        return None
