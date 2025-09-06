"""Repository Sandbox Generator

Automatically generate Docker configurations for any repository.
"""

from .analyzer.repo_analyzer import RepositoryAnalyzer
from .generators.dockerfile_generator import DockerfileGenerator
from .generators.docker_compose_generator import DockerComposeGenerator

__all__ = [
    "RepositoryAnalyzer",
    "DockerfileGenerator", 
    "DockerComposeGenerator",
]

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
