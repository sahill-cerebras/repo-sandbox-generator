"""Main repository analyzer module."""

import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List

from .dependency_resolver import DependencyResolver
from .test_discovery import TestDiscovery
from .version_detector import PythonVersionDetector

logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_PYTHON_VERSION = '3.11'
DEFAULT_EXPOSE_PORT = 8000  # retained constant though expose logic removed


class AnalysisKeys(str, Enum):
    """Keys for the analysis results dictionary for consistency."""
    REPO_PATH = 'repo_path'
    REPO_NAME = 'repo_name'
    PYTHON_VERSION = 'python_version'
    PYTHON_PACKAGES = 'python_packages'
    SYSTEM_PACKAGES = 'system_packages'
    ENVIRONMENT_VARS = 'environment_vars'
    TEST_CONFIG = 'test_config'
    BUILD_SYSTEM = 'build_system'
    ERRORS = 'errors'


class RepositoryAnalyzer:
    """Main class for analyzing repository structure and dependencies."""
    
    def __init__(self):
        self.dependency_resolver = DependencyResolver()
        self.test_discovery = TestDiscovery()
        self.version_detector = PythonVersionDetector()
    
    def analyze_repository(self, repo_path: Path) -> Dict[str, Any]:
        """
        Performs a complete analysis of the specified repository.
        
        Args:
            repo_path: Path to the repository to analyze.
            
        Returns:
            A dictionary containing the complete analysis results.
        """
        repo_path = Path(repo_path).resolve()
        
        analysis: Dict[str, Any] = {
            AnalysisKeys.REPO_PATH: str(repo_path),
            AnalysisKeys.REPO_NAME: repo_path.name,
            AnalysisKeys.PYTHON_VERSION: DEFAULT_PYTHON_VERSION,
            AnalysisKeys.PYTHON_PACKAGES: [],
            AnalysisKeys.SYSTEM_PACKAGES: [],
            AnalysisKeys.ENVIRONMENT_VARS: {},
            AnalysisKeys.TEST_CONFIG: {},
            AnalysisKeys.BUILD_SYSTEM: None,
            AnalysisKeys.ERRORS: []
        }
        
        try:
            # 1. Resolve all dependencies
            logger.info("Resolving dependencies...")
            deps = self.dependency_resolver.resolve_all_dependencies(repo_path)
            analysis.update(deps)
            
            # 2. Detect Python version
            logger.info("Detecting Python version...")
            analysis[AnalysisKeys.PYTHON_VERSION] = self.version_detector.detect_python_version(repo_path)

            # 3. Discover test configuration
            logger.info("Discovering test configuration...")
            analysis[AnalysisKeys.TEST_CONFIG] = self.test_discovery.discover_test_setup(repo_path)
            
            # 4. Apply generic Python project rules
            logger.info("Applying generic Python project rules...")
            analysis = self._apply_generic_rules(analysis)
            
        except Exception as e:
            logger.error(f"A critical error occurred during analysis: {e}", exc_info=logger.isEnabledFor(logging.DEBUG))
            analysis[AnalysisKeys.ERRORS].append(str(e))
        
        return analysis
    
    def _apply_generic_rules(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Apply generic Python project configuration rules based on prior analysis."""
        
        # Set default environment variables for Python
        analysis[AnalysisKeys.ENVIRONMENT_VARS].update({
            'PYTHONUNBUFFERED': '1',
            'PYTHONDONTWRITEBYTECODE': '1'
        })
        
        return analysis