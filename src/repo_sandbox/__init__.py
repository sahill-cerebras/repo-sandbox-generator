"""Repository Sandbox Generator

Automatically generate Docker configurations for any repository.
"""

from .analyzer.repo_analyzer import RepositoryAnalyzer
from .generators.dockerfile_generator import DockerfileGenerator

__all__ = [
    "RepositoryAnalyzer",
    "DockerfileGenerator",
]

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
