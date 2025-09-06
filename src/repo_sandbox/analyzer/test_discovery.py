"""Test framework discovery and configuration."""

import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class TestDiscovery:
    """Discovers test frameworks and configuration in repositories."""
    
    def discover_test_setup(self, repo_path: Path) -> Dict[str, Any]:
        """
        Discover test framework and configuration.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Dictionary with test configuration
        """
        test_config = {
            'framework': None,
            'test_paths': [],
            'test_commands': [],
            'environment_vars': {},
            'config_files': []
        }
        
        # Check for pytest
        pytest_config = self._check_pytest(repo_path)
        if pytest_config:
            test_config.update(pytest_config)
            return test_config
        
        # Check for Django tests
        django_config = self._check_django_tests(repo_path)
        if django_config:
            test_config.update(django_config)
            return test_config
        
        # Check for unittest
        unittest_config = self._check_unittest(repo_path)
        if unittest_config:
            test_config.update(unittest_config)
            return test_config
        
        # Check for tox
        tox_config = self._check_tox(repo_path)
        if tox_config:
            test_config.update(tox_config)
            return test_config
        
        # Generic test directory discovery
        test_config.update(self._discover_test_directories(repo_path))
        
        return test_config
    
    def _check_pytest(self, repo_path: Path) -> Optional[Dict[str, Any]]:
        """Check for pytest configuration."""
        config_files = ['pytest.ini', 'pyproject.toml', 'setup.cfg', 'tox.ini']
        
        for config_file in config_files:
            config_path = repo_path / config_file
            if config_path.exists():
                try:
                    content = config_path.read_text(encoding='utf-8')
                    if re.search(r'\[tool\.pytest', content) or re.search(r'\[pytest', content):
                        logger.info("Detected pytest configuration")
                        
                        test_paths = self._find_test_directories(repo_path)
                        
                        return {
                            'framework': 'pytest',
                            'test_commands': ['python -m pytest'],
                            'test_paths': test_paths,
                            'config_files': [config_file]
                        }
                except Exception as e:
                    logger.debug(f"Error reading {config_file}: {e}")
        
        # Check if pytest is in dependencies
        if self._has_pytest_dependency(repo_path):
            logger.info("Found pytest in dependencies")
            return {
                'framework': 'pytest',
                'test_commands': ['python -m pytest'],
                'test_paths': self._find_test_directories(repo_path)
            }
        
        return None
    
    def _check_django_tests(self, repo_path: Path) -> Optional[Dict[str, Any]]:
        """Check for Django test configuration."""
        # Look for Django indicators
        if not ((repo_path / 'manage.py').exists() or 
                (repo_path / 'tests' / 'runtests.py').exists()):
            return None
        
        # Check for Django in dependencies
        if not self._has_django_dependency(repo_path):
            return None
        
        logger.info("Detected Django test setup")
        
        test_commands = []
        
        # Check for custom test runner
        if (repo_path / 'tests' / 'runtests.py').exists():
            test_commands.append('./tests/runtests.py --verbosity 2')
        elif (repo_path / 'manage.py').exists():
            test_commands.append('python manage.py test')
        
        return {
            'framework': 'django',
            'test_commands': test_commands,
            'test_paths': self._find_test_directories(repo_path),
            'environment_vars': {
                'DJANGO_SETTINGS_MODULE': 'settings',
            }
        }
    
    def _check_unittest(self, repo_path: Path) -> Optional[Dict[str, Any]]:
        """Check for unittest configuration."""
        # Look for unittest patterns in Python files
        python_files = list(repo_path.rglob('*.py'))
        unittest_found = False
        
        for py_file in python_files[:20]:  # Limit search for performance
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                if re.search(r'import unittest|from unittest', content):
                    unittest_found = True
                    break
            except Exception:
                continue
        
        if unittest_found:
            logger.info("Detected unittest framework")
            return {
                'framework': 'unittest',
                'test_commands': ['python -m unittest discover'],
                'test_paths': self._find_test_directories(repo_path)
            }
        
        return None
    
    def _check_tox(self, repo_path: Path) -> Optional[Dict[str, Any]]:
        """Check for tox configuration."""
        tox_ini = repo_path / 'tox.ini'
        
        if tox_ini.exists():
            logger.info("Detected tox configuration")
            return {
                'framework': 'tox',
                'test_commands': ['tox'],
                'config_files': ['tox.ini']
            }
        
        return None
    
    def _find_test_directories(self, repo_path: Path) -> List[str]:
        """Find common test directory patterns."""
        test_dirs = []
        
        common_test_paths = [
            'tests',
            'test', 
            'testing',
            '*/tests',
            '*/test'
        ]
        
        for pattern in common_test_paths:
            if '*' in pattern:
                matches = list(repo_path.rglob(pattern))
                for match in matches:
                    if match.is_dir():
                        test_dirs.append(str(match.relative_to(repo_path)))
            else:
                test_dir = repo_path / pattern
                if test_dir.exists() and test_dir.is_dir():
                    test_dirs.append(pattern)
        
        return test_dirs
    
    def _discover_test_directories(self, repo_path: Path) -> Dict[str, Any]:
        """Discover test directories without specific framework."""
        test_paths = self._find_test_directories(repo_path)
        
        if test_paths:
            return {
                'framework': 'generic',
                'test_paths': test_paths,
                'test_commands': ['python -m unittest discover']  # Default fallback
            }
        
        return {'framework': None}
    
    def _has_pytest_dependency(self, repo_path: Path) -> bool:
        """Check if pytest is listed in dependencies."""
        return self._check_dependency_files(repo_path, [r'^pytest', r'pytest[>=<]'])
    
    def _has_django_dependency(self, repo_path: Path) -> bool:
        """Check if Django is listed in dependencies."""
        return self._check_dependency_files(repo_path, [r'^[Dd]jango', r'[Dd]jango[>=<]'])
    
    def _check_dependency_files(self, repo_path: Path, patterns: List[str]) -> bool:
        """Check for patterns in dependency files."""
        dep_files = [
            'requirements.txt',
            'requirements-dev.txt',
            'requirements/dev.txt',
            'pyproject.toml',
            'setup.py',
            'setup.cfg',
            'Pipfile'
        ]
        
        for dep_file in dep_files:
            file_path = repo_path / dep_file
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    for pattern in patterns:
                        if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                            return True
                except Exception:
                    continue
        
        return False
