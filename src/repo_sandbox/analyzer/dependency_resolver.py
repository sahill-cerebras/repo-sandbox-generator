"""Dependency resolution module."""

import ast
import re
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
import toml
import yaml

logger = logging.getLogger(__name__)


class DependencyResolver:
    """Resolves dependencies from various sources in a repository."""
    
    def resolve_all_dependencies(self, repo_path: Path) -> Dict[str, Any]:
        """
        Resolve dependencies from all available sources.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Dictionary with resolved dependencies and metadata
        """
        dependencies = {
            'python_packages': [],
            'system_packages': [],
            'build_system': None,
            'install_commands': [],
            'dev_dependencies': [],
            'dependency_files': []  # list of files actually parsed (for Docker layering)
        }
        
        # Check all possible dependency sources
        dep_sources = [
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
        
        found_sources = []
        for filename, parser in dep_sources:
            file_path = repo_path / filename
            if file_path.exists():
                try:
                    deps = parser(file_path)
                    dependencies = self._merge_dependencies(dependencies, deps)
                    found_sources.append(filename)
                    logger.debug(f"Parsed dependencies from {filename}")
                except Exception as e:
                    logger.warning(f"Error parsing {filename}: {e}")
        
        # Infer system dependencies from Python packages
        dependencies['system_packages'] = self._infer_system_dependencies(
            dependencies['python_packages']
        )
        
        # Determine build system
        dependencies['build_system'] = self._determine_build_system(repo_path, found_sources)
        
        # Generate install commands
        dependencies['install_commands'] = self._generate_install_commands(
            dependencies['build_system'], found_sources
        )
        
        # Record found sources for downstream tooling (e.g., Docker layering)
        dependencies['dependency_files'] = found_sources

        logger.info(
            f"Found {len(dependencies['python_packages'])} Python packages "
            f"from {len(found_sources)} sources: {found_sources}"
        )
        
        return dependencies
    
    def _parse_requirements_txt(self, file_path: Path) -> Dict[str, List[str]]:
        """Parse requirements.txt format files."""
        packages = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            for line in content.splitlines():
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Skip -e editable installs and -r recursive requirements
                if line.startswith(('-e', '-r', '--')):
                    continue
                
                # Parse package specification
                package = self._parse_requirement_line(line)
                if package:
                    packages.append(package)
        
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
        
        return {'python_packages': packages}
    
    def _parse_setup_py(self, file_path: Path) -> Dict[str, Any]:
        """Parse setup.py using AST analysis."""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            # Look for setup() call
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and 
                    hasattr(node.func, 'id') and 
                    node.func.id == 'setup'):
                    
                    return self._extract_setup_info(node)
        
        except Exception as e:
            logger.warning(f"Error parsing setup.py: {e}")
        
        return {'python_packages': []}
    
    def _parse_pyproject_toml(self, file_path: Path) -> Dict[str, Any]:
        """Parse pyproject.toml files."""
        try:
            content = file_path.read_text(encoding='utf-8')
            data = toml.loads(content)
            
            packages = []
            build_system = None
            
            # Check build-system
            if 'build-system' in data:
                build_system = 'pyproject'
            
            # Check project dependencies (PEP 621)
            if 'project' in data and 'dependencies' in data['project']:
                packages.extend(data['project']['dependencies'])
            
            # Check poetry dependencies
            if 'tool' in data and 'poetry' in data['tool']:
                poetry_data = data['tool']['poetry']
                if 'dependencies' in poetry_data:
                    for dep, version in poetry_data['dependencies'].items():
                        if dep != 'python':  # Skip Python version
                            if isinstance(version, str):
                                packages.append(f"{dep}{version}")
                            else:
                                packages.append(dep)
            
            return {
                'python_packages': packages,
                'build_system': build_system
            }
        
        except Exception as e:
            logger.warning(f"Error parsing pyproject.toml: {e}")
        
        return {'python_packages': []}
    
    def _parse_pipfile(self, file_path: Path) -> Dict[str, List[str]]:
        """Parse Pipfile format."""
        try:
            content = file_path.read_text(encoding='utf-8')
            data = toml.loads(content)
            
            packages = []
            
            # Parse packages section
            if 'packages' in data:
                for package, version in data['packages'].items():
                    if isinstance(version, str):
                        packages.append(f"{package}{version}")
                    else:
                        packages.append(package)
            
            return {'python_packages': packages}
        
        except Exception as e:
            logger.warning(f"Error parsing Pipfile: {e}")
        
        return {'python_packages': []}
    
    def _parse_conda_env(self, file_path: Path) -> Dict[str, List[str]]:
        """Parse conda environment.yml files."""
        try:
            content = file_path.read_text(encoding='utf-8')
            data = yaml.safe_load(content)
            
            packages = []
            
            if 'dependencies' in data:
                for dep in data['dependencies']:
                    if isinstance(dep, str):
                        # Skip conda-specific packages that start with versions
                        if not dep.startswith('python='):
                            packages.append(dep)
                    elif isinstance(dep, dict) and 'pip' in dep:
                        # Handle pip dependencies in conda env
                        packages.extend(dep['pip'])
            
            return {'python_packages': packages}
        
        except Exception as e:
            logger.warning(f"Error parsing environment.yml: {e}")
        
        return {'python_packages': []}
    
    def _parse_setup_cfg(self, file_path: Path) -> Dict[str, List[str]]:
        """Parse setup.cfg files."""
        try:
            import configparser
            
            config = configparser.ConfigParser()
            config.read(file_path)
            
            packages = []
            
            if 'options' in config and 'install_requires' in config['options']:
                install_requires = config['options']['install_requires']
                for line in install_requires.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        packages.append(line)
            
            return {'python_packages': packages}
        
        except Exception as e:
            logger.warning(f"Error parsing setup.cfg: {e}")
        
        return {'python_packages': []}
    
    def _parse_requirement_line(self, line: str) -> Optional[str]:
        """Parse a single requirement line."""
        # Remove inline comments
        line = line.split('#')[0].strip()
        
        # Skip empty lines
        if not line:
            return None
        
        # Handle Git URLs and other complex specs
        if any(proto in line for proto in ['git+', 'hg+', 'svn+', 'bzr+']):
            # Extract package name from Git URLs
            match = re.search(r'egg=([^&]+)', line)
            if match:
                return match.group(1)
            return None
        
        return line
    
    def _extract_setup_info(self, setup_node: ast.Call) -> Dict[str, List[str]]:
        """Extract dependency information from setup() call."""
        packages = []
        
        for keyword in setup_node.keywords:
            if keyword.arg in ['install_requires', 'requires']:
                if isinstance(keyword.value, ast.List):
                    for item in keyword.value.elts:
                        if isinstance(item, ast.Str):
                            packages.append(item.s)
                        elif isinstance(item, ast.Constant) and isinstance(item.value, str):
                            packages.append(item.value)
        
        return {'python_packages': packages}
    
    def _merge_dependencies(self, base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """Merge dependency dictionaries."""
        for key, value in new.items():
            if key in base:
                if isinstance(base[key], list) and isinstance(value, list):
                    # Merge lists and deduplicate
                    base[key] = list(set(base[key] + value))
                elif not base[key]:  # Override if base is empty
                    base[key] = value
            else:
                base[key] = value
        
        return base
    
    def _infer_system_dependencies(self, python_packages: List[str]) -> List[str]:
        """Infer system package dependencies from Python packages."""
        PACKAGE_TO_SYSTEM_DEPS = {
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
        
        system_deps = set()
        
        for package in python_packages:
            # Extract package name without version specifiers
            package_name = re.split(r'[>=<!=~]', package)[0].strip()
            
            if package_name in PACKAGE_TO_SYSTEM_DEPS:
                system_deps.update(PACKAGE_TO_SYSTEM_DEPS[package_name])
        
        return sorted(list(system_deps))
    
    def _determine_build_system(self, repo_path: Path, found_sources: List[str]) -> Optional[str]:
        """Determine the build system used by the repository."""
        if 'pyproject.toml' in found_sources:
            return 'pyproject'
        elif 'setup.py' in found_sources:
            return 'setuptools'
        elif 'Pipfile' in found_sources:
            return 'pipenv'
        elif 'environment.yml' in found_sources:
            return 'conda'
        elif any('requirements' in src for src in found_sources):
            return 'pip'
        else:
            return None
    
    def _generate_install_commands(self, build_system: Optional[str], 
                                 found_sources: List[str]) -> List[str]:
        """Generate appropriate install commands based on build system."""
        commands = []
        
        if build_system == 'pyproject':
            commands.append('python -m pip install -e .')
        elif build_system == 'setuptools':
            commands.append('python -m pip install -e .')
        elif build_system == 'pipenv':
            commands.append('pipenv install')
        elif build_system == 'conda':
            commands.append('conda env create -f environment.yml')
        else:
            # Default pip install
            if 'requirements.txt' in found_sources:
                commands.append('python -m pip install -r requirements.txt')
            
            # Add development dependencies
            dev_files = ['requirements-dev.txt', 'requirements/dev.txt']
            for dev_file in dev_files:
                if dev_file in found_sources:
                    commands.append(f'python -m pip install -r {dev_file}')
                    break
        
        return commands
