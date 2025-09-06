"""Enhanced Python version detection module.

- Preserves patch versions (e.g., 3.11.2) and supports wildcards (3.11.*).
- Collects ALL versions from multiple sources, then returns the highest.
- Adds README.md/README.rst parsing.
"""

import re
import logging
from pathlib import Path
from typing import Optional, List
import ast
import toml

logger = logging.getLogger(__name__)


class PythonVersionDetector:
    """Detect Python version requirements from various sources."""

    def detect_python_version(self, repo_path: Path) -> str:
        """Detect Python version from multiple sources.

    Sources scanned (all collected):
        1. .python-version
        2. runtime.txt
        3. pyproject.toml (project / poetry)
        4. setup.py (python_requires)
        5. setup.cfg (python_requires)
        6. tox.ini (envlist -> all versions)
        7. GitHub Actions workflows (actions/setup-python)
        8. Dockerfile (FROM python:...)
    9. README.md / README.rst (patterns: Python 3.11.8, Python: 3.11.8+, Python version 3.11)
    10. requirements*.txt (comment hints like '# Python 3.11.8' or '# Python version: 3.11.8+')
    11. environment.yml (lines containing python=X.Y[.Z])

        Returns highest detected version, else default 3.10.
        """
        version_sources = [
            ('.python-version', self._parse_python_version_file),
            ('runtime.txt', self._parse_runtime_txt),
            ('pyproject.toml', self._parse_pyproject_python_version),
            ('setup.py', self._parse_setup_py_python_version),
            ('setup.cfg', self._parse_setup_cfg_python_version),
            ('tox.ini', self._parse_tox_python_versions),
            ('.github/workflows', self._parse_github_actions_python),
            ('Dockerfile', self._parse_dockerfile_python),
            ('README.md', self._parse_readme_python_version),
            ('README.rst', self._parse_readme_python_version),
        ]

        detected_versions: List[str] = []

        for source, parser in version_sources:
            source_path = repo_path / source
            if source_path.exists():
                try:
                    version = parser(source_path)
                    if version:
                        if isinstance(version, list):
                            detected_versions.extend(version)
                        else:
                            detected_versions.append(version)
                except Exception as e:
                    logger.debug(f"Error parsing {source}: {e}")

        # Scan requirements*.txt for commented python version hints
        for req_file in repo_path.glob("requirements*.txt"):
            try:
                rv = self._parse_requirements_for_python_version(req_file)
                if rv:
                    detected_versions.append(rv)
            except Exception as e:
                logger.debug(f"Error parsing {req_file.name} for python version: {e}")

        # Scan environment.yml if present
        env_file = repo_path / "environment.yml"
        if env_file.exists():
            try:
                ev = self._parse_environment_yml_python_version(env_file)
                if ev:
                    detected_versions.append(ev)
            except Exception as e:
                logger.debug(f"Error parsing environment.yml: {e}")

        if detected_versions:
            best = max(detected_versions, key=self._version_key)
            logger.info(f"Detected versions {detected_versions}, best={best}")
            return best

        default_version = "3.10"
        logger.info(f"No Python version detected, using default: {default_version}")
        return default_version

    # ------------------------- Individual Parsers ------------------------- #

    def _parse_python_version_file(self, file_path: Path) -> Optional[str]:
        try:
            content = file_path.read_text().strip()
            return self._extract_version_from_requirement(content)
        except Exception as e:
            logger.debug(f"Error reading .python-version: {e}")
        return None

    def _parse_runtime_txt(self, file_path: Path) -> Optional[str]:
        try:
            content = file_path.read_text().strip()
            return self._extract_version_from_requirement(content)
        except Exception as e:
            logger.debug(f"Error reading runtime.txt: {e}")
        return None

    def _parse_pyproject_python_version(self, file_path: Path) -> Optional[str]:
        try:
            data = toml.loads(file_path.read_text(encoding="utf-8"))

            # PEP 621
            project = data.get("project", {})
            python_req = project.get("requires-python")
            if isinstance(python_req, str):
                return self._extract_version_from_requirement(python_req)

            # Poetry
            poetry_dep = (
                data.get("tool", {}).get("poetry", {}).get("dependencies", {}).get("python")
            )
            if isinstance(poetry_dep, str):
                return self._extract_version_from_requirement(poetry_dep)
        except Exception as e:
            logger.debug(f"Error parsing pyproject.toml: {e}")
        return None

    def _parse_setup_py_python_version(self, file_path: Path) -> Optional[str]:
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup":
                    for kw in node.keywords:
                        if kw.arg == "python_requires":
                            if isinstance(kw.value, ast.Str):
                                req = kw.value.s
                            elif isinstance(kw.value, ast.Constant) and isinstance(
                                kw.value.value, str
                            ):
                                req = kw.value.value
                            else:
                                continue
                            return self._extract_version_from_requirement(req)
        except Exception as e:
            logger.debug(f"Error parsing setup.py: {e}")
        return None

    def _parse_setup_cfg_python_version(self, file_path: Path) -> Optional[str]:
        try:
            import configparser

            cfg = configparser.ConfigParser()
            cfg.read(file_path)
            if cfg.has_option("options", "python_requires"):
                req = cfg.get("options", "python_requires")
                return self._extract_version_from_requirement(req)
        except Exception as e:
            logger.debug(f"Error parsing setup.cfg: {e}")
        return None

    def _parse_tox_python_versions(self, file_path: Path) -> Optional[List[str]]:
        try:
            import configparser

            cfg = configparser.ConfigParser()
            cfg.read(file_path)
            if cfg.has_option("tox", "envlist"):
                envlist = cfg.get("tox", "envlist")
                versions = [f"{m[0]}.{m[1]}" for m in re.findall(r"py(\d)(\d+)", envlist)]
                return versions if versions else None
        except Exception as e:
            logger.debug(f"Error parsing tox.ini: {e}")
        return None

    def _parse_github_actions_python(self, workflows_dir: Path) -> Optional[List[str]]:
        versions = []
        try:
            workflow_files = list(workflows_dir.rglob("*.yml")) + list(
                workflows_dir.rglob("*.yaml")
            )
            import yaml  # local import

            for wf in workflow_files:
                try:
                    data = yaml.safe_load(wf.read_text(encoding="utf-8")) or {}
                    for job in (data.get("jobs") or {}).values():
                        for step in job.get("steps", []):
                            if (
                                isinstance(step, dict)
                                and "uses" in step
                                and "actions/setup-python" in step["uses"]
                            ):
                                pyver = (step.get("with") or {}).get("python-version")
                                if pyver:
                                    v = self._extract_version_from_requirement(str(pyver))
                                    if v:
                                        versions.append(v)
                except Exception as ie:
                    logger.debug(f"Error parsing workflow {wf}: {ie}")
        except Exception as e:
            logger.debug(f"Error reading workflows: {e}")
        return versions or None

    def _parse_dockerfile_python(self, file_path: Path) -> Optional[str]:
        try:
            content = file_path.read_text(encoding="utf-8")
            return self._extract_version_from_requirement(content)
        except Exception as e:
            logger.debug(f"Error parsing Dockerfile: {e}")
        return None

    def _parse_readme_python_version(self, file_path: Path) -> Optional[str]:
        try:
            content = file_path.read_text(encoding="utf-8")
            # Capture variants: 'Python 3.11.8', 'Python: 3.11.8+', 'Python version 3.11.8'
            matches = re.findall(
                r"(?:Python(?:\s+version)?\s*[:=\-]?\s*)(\d+\.\d+(?:\.\d+)?)(?:\+)?",
                content,
                flags=re.IGNORECASE,
            )
            if matches:
                return max(matches, key=self._version_key)
        except Exception as e:
            logger.debug(f"Error parsing README: {e}")
        return None

    def _parse_requirements_for_python_version(self, file_path: Path) -> Optional[str]:
        """Look for commented python version hints in requirements*.txt.

        Examples accepted:
          # Python 3.11.8
          # python: 3.11.8+
          # Python version 3.10
        Only scans first 50 lines to stay fast.
        """
        try:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()[:50]
            candidates = []
            pattern = re.compile(
                r"^\s*#.*?python(?:\s+version)?\s*[:=]?\s*(\d+\.\d+(?:\.\d+)?)(?:\+)?",
                re.IGNORECASE,
            )
            for line in lines:
                m = pattern.search(line)
                if m:
                    candidates.append(m.group(1))
            if candidates:
                return max(candidates, key=self._version_key)
        except Exception as e:
            logger.debug(f"Error scanning requirements comments: {e}")
        return None

    def _parse_environment_yml_python_version(self, file_path: Path) -> Optional[str]:
        """Extract python=X.Y(.Z) from environment.yml dependencies section.
        Accepts both 'python=3.11.8' and 'python==3.11.8'.
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            matches = re.findall(r"python[=]{1,2}(\d+\.\d+(?:\.\d+)?)", content, re.IGNORECASE)
            if matches:
                return max(matches, key=self._version_key)
        except Exception as e:
            logger.debug(f"Error parsing environment.yml: {e}")
        return None

    # ------------------------- Extraction Helper ------------------------- #

    def _extract_version_from_requirement(self, requirement: str) -> Optional[str]:
        """Extract best version token (X.Y[.Z]) from a version spec string."""
        requirement = requirement.strip()
        candidates = re.findall(r"(\d+\.\d+(?:\.\d+)?)", requirement)
        if not candidates:
            wild = re.findall(r"(\d+\.\d+)\.\*", requirement)
            candidates.extend(wild)
        if not candidates:
            return None
        return max(candidates, key=self._version_key)

    def _version_key(self, v: str):
        parts = v.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else -1
        return (major, minor, patch)
