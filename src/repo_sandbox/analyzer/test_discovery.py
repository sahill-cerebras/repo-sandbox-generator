"""Test framework discovery and configuration."""

import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import sys

logger = logging.getLogger(__name__)


class TestDiscovery:
    """Discovers test frameworks and their configuration in a repository."""

    # --- Constants for Configuration ---
    PYTEST_CONFIG_FILES = ['pytest.ini', 'pyproject.toml', 'setup.cfg', 'tox.ini']
    COMMON_TEST_PATHS = ['tests', 'test', 'testing']
    DEPENDENCY_FILES_FOR_CHECK = [
        'requirements.txt', 'requirements-dev.txt', 'requirements/dev.txt',
        'pyproject.toml', 'setup.py', 'setup.cfg', 'Pipfile'
    ]

    def discover_test_setup(self, repo_path: Path) -> Dict[str, Any]:
        """
        Discovers the test framework and configuration by checking for multiple signals.
        
        Detection precedence: pytest > Django tests > unittest > tox.
        
        Args:
            repo_path: Path to the repository.
            
        Returns:
            A dictionary with the discovered test configuration.
        """
        # Find all potential test directories once to avoid redundant I/O
        test_paths = self._find_test_directories(repo_path)
        
        base_config = {
            'framework': None,
            'test_paths': test_paths,
            'test_commands': [],
            'environment_vars': {},
            'config_files': [],
            'extra_test_imports': []  # dynamically scanned imports used only in tests
        }
        
        if config := self._check_pytest(repo_path, test_paths):
            base_config.update(config)
            # Append scanned imports
            base_config['extra_test_imports'] = self._scan_test_imports(repo_path, test_paths)
            return base_config
        
        if config := self._check_django_tests(repo_path, test_paths):
            base_config.update(config)
            base_config['extra_test_imports'] = self._scan_test_imports(repo_path, test_paths)
            return base_config
        
        if config := self._check_unittest(repo_path, test_paths):
            base_config.update(config)
            base_config['extra_test_imports'] = self._scan_test_imports(repo_path, test_paths)
            return base_config
        
        if config := self._check_tox(repo_path):
            base_config.update(config)
            # tox env usually installs deps; still capture imports
            base_config['extra_test_imports'] = self._scan_test_imports(repo_path, test_paths)
            return base_config
        
        # Fallback for when test directories exist but no framework is detected
        if test_paths:
            base_config['framework'] = 'generic'
            base_config['test_commands'] = ['python -m unittest discover']
            base_config['extra_test_imports'] = self._scan_test_imports(repo_path, test_paths)
        
        return base_config
    
    def _check_pytest(self, repo_path: Path, test_paths: List[str]) -> Optional[Dict[str, Any]]:
        """Checks for pytest by inspecting config files and dependencies."""
        for config_file in self.PYTEST_CONFIG_FILES:
            config_path = repo_path / config_file
            if config_path.exists():
                try:
                    content = config_path.read_text(encoding='utf-8', errors='ignore')
                    if re.search(r'\[tool\.pytest\.ini_options\]|\[pytest\]', content):
                        logger.info(f"Detected pytest configuration in {config_file}")
                        return {
                            'framework': 'pytest',
                            'test_commands': ['python -m pytest'],
                            'test_paths': test_paths,
                            'config_files': [config_file]
                        }
                except Exception as e:
                    logger.debug(f"Error reading {config_file} for pytest check: {e}")
        
        if self._is_dependency_present(repo_path, r'^pytest\b'):
            logger.info("Detected pytest in project dependencies")
            return {
                'framework': 'pytest',
                'test_commands': ['python -m pytest'],
                'test_paths': test_paths
            }
        
        return None
    
    def _check_django_tests(self, repo_path: Path, test_paths: List[str]) -> Optional[Dict[str, Any]]:
        """Checks for a Django test setup."""
        is_django_project = (repo_path / 'manage.py').exists()
        has_django_dep = self._is_dependency_present(repo_path, r'^[Dd]jango\b')
        
        if is_django_project and has_django_dep:
            logger.info("Detected Django test setup via manage.py")
            return {
                'framework': 'django',
                'test_commands': ['python manage.py test'],
                'test_paths': test_paths,
                'environment_vars': {'DJANGO_SETTINGS_MODULE': 'settings'}
            }
            
        return None
    
    def _check_unittest(self, repo_path: Path, test_paths: List[str]) -> Optional[Dict[str, Any]]:
        """Checks for `unittest` imports within discovered test directories."""
        if not test_paths:
            return None
        
        for test_dir_str in test_paths:
            test_dir = repo_path / test_dir_str
            for py_file in list(test_dir.rglob('*.py'))[:20]: # Limit search scope
                try:
                    content = py_file.read_text(encoding='utf-8', errors='ignore')
                    if re.search(r'import unittest|from unittest', content):
                        logger.info(f"Detected unittest import in {py_file.relative_to(repo_path)}")
                        return {
                            'framework': 'unittest',
                            'test_commands': ['python -m unittest discover'],
                            'test_paths': test_paths
                        }
                except Exception:
                    continue
        return None

    def _check_tox(self, repo_path: Path) -> Optional[Dict[str, Any]]:
        """Checks for a tox configuration file."""
        if (repo_path / 'tox.ini').exists():
            logger.info("Detected tox configuration")
            return {
                'framework': 'tox',
                'test_commands': ['tox'],
                'config_files': ['tox.ini']
            }
        return None
    
    def _find_test_directories(self, repo_path: Path) -> List[str]:
        """Find test directories at root and nested (limited depth) for smarter discovery.

        Includes:
          - Root-level: tests, test, testing
          - Nested: any directory named 'tests' (or 'test', 'testing') under typical source roots
            like src/, lib/, package dirs, or directly elsewhere up to depth 5.
        Excludes common virtual env / build artifact directories.
        """
        ignore = {'.git', '.venv', 'venv', 'env', 'build', 'dist', '__pycache__', '.mypy_cache', '.pytest_cache'}
        found: Set[str] = set()

        # Root-level first-pass
        for pattern in self.COMMON_TEST_PATHS:
            p = repo_path / pattern
            if p.is_dir():
                found.add(str(p.relative_to(repo_path)))

        # Candidate parent dirs to search more deeply
        candidate_parents = []
        for name in ['src', 'lib']:
            d = repo_path / name
            if d.is_dir():
                candidate_parents.append(d)
        # Also add top-level package-like dirs (those containing __init__.py)
        for child in repo_path.iterdir():
            if child.is_dir() and (child / '__init__.py').exists():
                candidate_parents.append(child)

        # Breadth-limited recursive glob for nested test dirs
        max_depth = 5
        def depth(rel: Path) -> int:
            return len(rel.parts)

        scanned_dirs: Set[Path] = set()
        queue = list(candidate_parents)
        while queue:
            current = queue.pop()
            if current in scanned_dirs:
                continue
            scanned_dirs.add(current)
            rel = current.relative_to(repo_path)
            if any(part in ignore for part in rel.parts):
                continue
            # If directory name matches a test pattern, record it
            if current.name in self.COMMON_TEST_PATHS:
                found.add(str(rel))
            # Enqueue children if depth limit not reached
            if depth(rel) < max_depth:
                try:
                    for sub in current.iterdir():
                        if sub.is_dir():
                            queue.append(sub)
                except Exception:
                    pass

        return sorted(found)
    
    def _is_dependency_present(self, repo_path: Path, pattern: str) -> bool:
        """Checks for a dependency pattern in common dependency files."""
        regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        
        for dep_file in self.DEPENDENCY_FILES_FOR_CHECK:
            file_path = repo_path / dep_file
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    if regex.search(content):
                        return True
                except Exception as e:
                    logger.debug(f"Could not check dependency in {dep_file}: {e}")
        return False

    # --- New helper to scan test imports ---
    def _scan_test_imports(self, repo_path: Path, test_paths: List[str]) -> List[str]:
        """Scan test files for imported top-level modules not obviously in stdlib.

        Heuristic: collect names from 'import X' and 'from X import'. Filter out
        stdlib modules (using sys.stdlib_module_names when available) and common
        project-local names (those matching repo root package directory names).
        """
        if not test_paths:
            return []
        stdlib: Set[str] = set()
        try:
            # Python 3.10+
            stdlib = set(getattr(sys, 'stdlib_module_names', set()))  # type: ignore
        except Exception:
            pass

        # Add builtin sentinel modules
        stdlib.update({'unittest', 'json', 're', 'pathlib', 'logging', 'datetime', 'typing', 'functools', 'itertools'})

        # Potential local package directories at repo root
        local_packages = {p.name for p in repo_path.iterdir() if p.is_dir() and (p / '__init__.py').exists()}

        collected: Set[str] = set()
        max_files = 1000  # safety limit
        file_count = 0
        for rel_dir in test_paths:
            tdir = repo_path / rel_dir
            if not tdir.exists():
                continue
            for py_file in tdir.rglob('*.py'):
                file_count += 1
                if file_count > max_files:
                    break
                try:
                    txt = py_file.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    continue
                for match in re.finditer(r'^\s*import\s+([a-zA-Z0-9_]+)', txt, re.MULTILINE):
                    collected.add(match.group(1).split('.')[0])
                for match in re.finditer(r'^\s*from\s+([a-zA-Z0-9_]+)\s+import', txt, re.MULTILINE):
                    collected.add(match.group(1).split('.')[0])

        # Filter
        filtered = [m for m in collected if m not in stdlib and m not in local_packages]
        # Common false positives to drop
        blacklist = {'tests'}
        filtered = [m for m in filtered if m not in blacklist]
        return sorted(filtered)