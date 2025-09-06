"""Main repository analyzer module."""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from .dependency_resolver import DependencyResolver
from .test_discovery import TestDiscovery
from .version_detector import PythonVersionDetector

logger = logging.getLogger(__name__)


class RepositoryAnalyzer:
    """Main class for analyzing repository structure and dependencies."""
    
    def __init__(self):
        self.dependency_resolver = DependencyResolver()
        self.test_discovery = TestDiscovery()
        self.version_detector = PythonVersionDetector()
    
    def analyze_repository(self, repo_path: Path) -> Dict[str, Any]:
        """
        Complete repository analysis.
        
        Args:
            repo_path: Path to the repository to analyze
            
        Returns:
            Dictionary containing complete analysis results
        """
        repo_path = Path(repo_path)
        
        analysis = {
            'repo_path': str(repo_path.absolute()),
            'repo_name': repo_path.name,
            'python_version': '3.9',
            'python_packages': [],
            'system_packages': [],
            'environment_vars': {},
            # 'frameworks' removed per requirement (framework detection disabled)
            'test_config': {},
            'install_commands': [],
            'expose_ports': [8000],  # Default port
            'build_system': None,
            'confidence': 1.0,
            'errors': []
        }
        
        try:
            # Resolve all dependencies
            logger.info("Resolving dependencies")
            deps = self.dependency_resolver.resolve_all_dependencies(repo_path)
            analysis.update(deps)
            
            # Detect Python version
            logger.info("Detecting Python version")
            analysis['python_version'] = self.version_detector.detect_python_version(repo_path)

            # Framework detection intentionally removed
            
            # Discover test configuration
            logger.info("Discovering test configuration")
            analysis['test_config'] = self.test_discovery.discover_test_setup(repo_path)
            
            # Apply generic Python project rules
            logger.info("Applying generic Python project rules")
            analysis = self._apply_generic_rules(analysis, repo_path)
            
            # Calculate confidence score
            analysis['confidence'] = self._calculate_confidence(analysis)
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            analysis['confidence'] = 0.5
            analysis['errors'].append(str(e))
        
        return analysis
    
    def _apply_generic_rules(self, analysis: Dict[str, Any], repo_path: Path) -> Dict[str, Any]:
        """Apply generic Python project configuration rules."""
        
        # Set default environment variables for Python
        analysis['environment_vars'].update({
            'PYTHONUNBUFFERED': '1',
            'PYTHONDONTWRITEBYTECODE': '1'
        })
        
        # Set default install commands based on detected build system
        if not analysis['install_commands']:
            if (repo_path / 'pyproject.toml').exists():
                analysis['install_commands'] = ['python -m pip install -e .']
                analysis['build_system'] = 'pyproject'
            elif (repo_path / 'setup.py').exists():
                analysis['install_commands'] = ['python -m pip install -e .']
                analysis['build_system'] = 'setuptools'
            elif (repo_path / 'requirements.txt').exists():
                analysis['install_commands'] = ['python -m pip install -r requirements.txt']
                analysis['build_system'] = 'requirements'
            else:
                analysis['install_commands'] = ['echo "No install method detected"']
        
        # Ensure at least one port is exposed
        if not analysis['expose_ports']:
            analysis['expose_ports'] = [8000]
        
        return analysis
    
    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis."""
        confidence = 1.0
        
        # Reduce confidence if no dependencies found
        if not analysis['python_packages']:
            confidence *= 0.8
        
        # Reduce confidence for each error
        confidence *= (0.9 ** len(analysis['errors']))
        
        # Increase confidence if specific indicators are found
        if analysis['test_config'].get('framework'):
            confidence *= 1.1
        
        if analysis.get('build_system'):
            confidence *= 1.05

    # Framework-related confidence adjustments removed
        
        return min(confidence, 1.0)
