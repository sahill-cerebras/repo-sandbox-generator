"""Test framework discovery and configuration."""

import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

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
            'config_files': []
        }
        
        if config := self._check_pytest(repo_path, test_paths):
            base_config.update(config)
            return base_config
        
        if config := self._check_django_tests(repo_path, test_paths):
            base_config.update(config)
            return base_config
        
        if config := self._check_unittest(repo_path, test_paths):
            base_config.update(config)
            return base_config
        
        if config := self._check_tox(repo_path):
            base_config.update(config)
            return base_config
        
        # Fallback for when test directories exist but no framework is detected
        if test_paths:
            base_config['framework'] = 'generic'
            base_config['test_commands'] = ['python -m unittest discover']
        
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
        """Finds common test directories."""
        return sorted(list({
            str(p.relative_to(repo_path))
            for pattern in self.COMMON_TEST_PATHS
            for p in repo_path.glob(pattern) if p.is_dir()
        }))
    
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