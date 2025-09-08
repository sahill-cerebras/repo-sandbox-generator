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
DEFAULT_PYTHON_VERSION = '3.9'
DEFAULT_EXPOSE_PORT = 8000
CONFIDENCE_NO_DEPS = 0.8
CONFIDENCE_PER_ERROR = 0.9
CONFIDENCE_BOOST_TESTS = 1.1
CONFIDENCE_BOOST_BUILD_SYSTEM = 1.05


class AnalysisKeys(str, Enum):
    """Keys for the analysis results dictionary for consistency."""
    REPO_PATH = 'repo_path'
    REPO_NAME = 'repo_name'
    PYTHON_VERSION = 'python_version'
    PYTHON_PACKAGES = 'python_packages'
    SYSTEM_PACKAGES = 'system_packages'
    ENVIRONMENT_VARS = 'environment_vars'
    TEST_CONFIG = 'test_config'
    INSTALL_COMMANDS = 'install_commands'
    EXPOSE_PORTS = 'expose_ports'
    BUILD_SYSTEM = 'build_system'
    CONFIDENCE = 'confidence'
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
            AnalysisKeys.INSTALL_COMMANDS: [],
            AnalysisKeys.EXPOSE_PORTS: [DEFAULT_EXPOSE_PORT],
            AnalysisKeys.BUILD_SYSTEM: None,
            AnalysisKeys.CONFIDENCE: 1.0,
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
            
            # 5. Calculate final confidence score
            analysis[AnalysisKeys.CONFIDENCE] = self._calculate_confidence(analysis)
            
        except Exception as e:
            logger.error(f"A critical error occurred during analysis: {e}", exc_info=logger.isEnabledFor(logging.DEBUG))
            analysis[AnalysisKeys.CONFIDENCE] = 0.5
            analysis[AnalysisKeys.ERRORS].append(str(e))
        
        return analysis
    
    def _apply_generic_rules(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Apply generic Python project configuration rules based on prior analysis."""
        
        # Set default environment variables for Python
        analysis[AnalysisKeys.ENVIRONMENT_VARS].update({
            'PYTHONUNBUFFERED': '1',
            'PYTHONDONTWRITEBYTECODE': '1'
        })
        
        # Set default install commands ONLY if resolver didn't provide them
        if not analysis[AnalysisKeys.INSTALL_COMMANDS]:
            build_system = analysis[AnalysisKeys.BUILD_SYSTEM]
            if build_system == 'pyproject' or build_system == 'setuptools':
                analysis[AnalysisKeys.INSTALL_COMMANDS] = ['python -m pip install -e .']
            elif build_system == 'pip':
                analysis[AnalysisKeys.INSTALL_COMMANDS] = ['python -m pip install -r requirements.txt']
            else:
                # A safe fallback if no build system was detected
                analysis[AnalysisKeys.INSTALL_COMMANDS] = ['echo "No primary install method detected."']
        
        return analysis
    
    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """
        Calculates a confidence score for the analysis based on detected signals.
        
        Args:
            analysis: The current analysis dictionary.
        
        Returns:
            A confidence score between 0.0 and 1.0.
        """
        confidence = 1.0
        
        # Reduce confidence if no Python packages were found.
        if not analysis[AnalysisKeys.PYTHON_PACKAGES]:
            confidence *= CONFIDENCE_NO_DEPS
        
        # Reduce confidence for each error encountered during the process.
        confidence *= (CONFIDENCE_PER_ERROR ** len(analysis[AnalysisKeys.ERRORS]))
        
        # Increase confidence if a specific test framework was found.
        if analysis[AnalysisKeys.TEST_CONFIG].get('framework'):
            confidence *= CONFIDENCE_BOOST_TESTS
        
        # Increase confidence if a known build system was identified.
        if analysis.get(AnalysisKeys.BUILD_SYSTEM):
            confidence *= CONFIDENCE_BOOST_BUILD_SYSTEM
        
        return min(confidence, 1.0)