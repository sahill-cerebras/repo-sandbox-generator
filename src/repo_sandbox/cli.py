#!/usr/bin/env python3
"""Command-line interface for Repository Sandbox Generator."""

import json
import tempfile
import shutil
from pathlib import Path
from typing import Optional

import click
from git import Repo

from .analyzer.repo_analyzer import RepositoryAnalyzer
from .generators.dockerfile_generator import DockerfileGenerator
from .generators.docker_compose_generator import DockerComposeGenerator


@click.group()
@click.version_option()
def cli():
    """Repository Sandbox Generator - Generate Docker configurations for any repo"""
    pass


@cli.command()
@click.argument('repo_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', default='./docker-config',
              help='Output directory for generated files')
@click.option('--template', default='auto',
              help='Docker template to use (auto, slim, full, scientific)')
@click.option('--python-version',
              help='Override Python version detection')
@click.option('--include-tests', is_flag=True,
              help='Include test configuration in Docker setup')
@click.option('--include-compose', is_flag=True,
              help='Generate docker-compose.yml file')
@click.option('--copy-source/--no-copy-source', default=True,
              help='Copy full repository source into output directory (for standalone build)')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose output')
def generate(repo_path: Path, output: str, template: str, python_version: Optional[str],
             include_tests: bool, include_compose: bool, copy_source: bool, verbose: bool):
    """Generate Docker configuration for a repository"""
    
    output_path = Path(output)
    
    if verbose:
        click.echo(f"🔍 Analyzing repository: {repo_path}")
    
    # Analyze repository
    analyzer = RepositoryAnalyzer()
    try:
        analysis = analyzer.analyze_repository(repo_path)
    except Exception as e:
        click.echo(f"❌ Error analyzing repository: {e}", err=True)
        raise click.Abort()
    
    # Override Python version if specified
    if python_version:
        analysis['python_version'] = python_version
        if verbose:
            click.echo(f"🐍 Using specified Python version: {python_version}")
    
    # Display analysis results
    if verbose:
        click.echo(f"📊 Analysis Results:")
        click.echo(f"   Python version: {analysis['python_version']}")
        click.echo(f"   Dependencies: {len(analysis['python_packages'])} packages")
        click.echo(f"   System packages: {len(analysis['system_packages'])} packages")
        click.echo(f"   Confidence: {analysis.get('confidence', 1.0):.2f}")
    
    # Generate Docker configuration
    dockerfile_generator = DockerfileGenerator()
    
    try:
        dockerfile_content = dockerfile_generator.generate_dockerfile(
            analysis, 
            template=template,
            include_tests=include_tests
        )
    except Exception as e:
        click.echo(f"❌ Error generating Dockerfile: {e}", err=True)
        raise click.Abort()
    
    # Generate docker-compose if requested
    compose_content = None
    if include_compose:
        compose_generator = DockerComposeGenerator()
        try:
            compose_content = compose_generator.generate_docker_compose(analysis)
        except Exception as e:
            click.echo(f"⚠️  Warning: Could not generate docker-compose.yml: {e}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect names of copied dependency files for reporting
    copied: list[str] = []
    try:
        # Write Dockerfile
        dockerfile_path = output_path / 'Dockerfile'
        dockerfile_path.write_text(dockerfile_content)

        # Write docker-compose.yml if generated
        if compose_content:
            compose_path = output_path / 'docker-compose.yml'
            compose_path.write_text(compose_content)

        # Copy dependency definition files into output so Docker build context has them
        dep_files = [
            'requirements.txt', 'requirements-dev.txt', 'requirements/base.txt', 'requirements/dev.txt',
            'setup.py', 'setup.cfg', 'pyproject.toml', 'Pipfile', 'environment.yml'
        ]
        for fname in dep_files:
            src = repo_path / fname
            if src.exists():
                # flatten nested requirements paths
                dst_name = Path(fname).name
                dst = output_path / dst_name
                try:
                    shutil.copy2(src, dst)
                    copied.append(dst_name)
                except Exception as copy_err:
                    if verbose:
                        click.echo(f"   ⚠️  Skipped copying {fname}: {copy_err}")

        # Write analysis results
        analysis_path = output_path / 'analysis.json'
        analysis_path.write_text(json.dumps(analysis, indent=2))

        # Write .dockerignore
        dockerignore_content = dockerfile_generator.generate_dockerignore(analysis)
        dockerignore_path = output_path / '.dockerignore'
        dockerignore_path.write_text(dockerignore_content)

        # Optionally copy source code tree for standalone build context
        if copy_source:
            if verbose:
                click.echo("🗂  Copying source tree into output (excluding common ignore patterns)...")
            ignore_names = {'.git', output_path.name, '__pycache__', '.mypy_cache', '.pytest_cache', 'build', 'dist'}
            for src_item in repo_path.rglob('*'):
                rel = src_item.relative_to(repo_path)
                if any(part in ignore_names for part in rel.parts):
                    continue
                dest = output_path / rel
                try:
                    if src_item.is_dir():
                        dest.mkdir(parents=True, exist_ok=True)
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_item, dest)
                except Exception as copy_err:
                    if verbose:
                        click.echo(f"   ⚠️  Skipped {rel}: {copy_err}")
    except Exception as e:
        click.echo(f"❌ Error writing files: {e}", err=True)
        raise click.Abort()
    
    # Success message
    click.echo(f"✅ Docker configuration generated in: {output_path}")
    click.echo("📁 Generated files:")
    click.echo(f"   - Dockerfile")
    if compose_content:
        click.echo(f"   - docker-compose.yml")
    click.echo(f"   - analysis.json")
    click.echo(f"   - .dockerignore")
    if copy_source:
        click.echo(f"   - (source tree copied)")
    if copied:
        click.echo(f"   - dependency files copied: {', '.join(copied)}")
    
    # Build instructions
    click.echo(f"\n🚀 To build and run:")
    click.echo(f"   cd {output_path}")
    click.echo(f"   docker build -t {analysis['repo_name'].lower()} .")
    if compose_content:
        click.echo(f"   docker-compose up")
    else:
        click.echo(f"   docker run -p 8000:8000 {analysis['repo_name'].lower()}")


@cli.command()
@click.argument('repo_url')
@click.option('--clone-dir',
              help='Temporary clone directory (default: system temp)')
@click.option('--output', '-o', default='./docker-config',
              help='Output directory for generated files')
@click.option('--copy-source/--no-copy-source', default=True,
              help='Copy full repository source into output directory (default on for from-git)')
@click.option('--cleanup/--no-cleanup', default=True,
              help='Clean up cloned repository after processing')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose output')
def from_git(repo_url: str, clone_dir: Optional[str], output: str,
             copy_source: bool, cleanup: bool, verbose: bool):
    """Generate Docker config directly from a Git repository"""
    
    # Determine clone directory
    if clone_dir:
        clone_path = Path(clone_dir)
        temp_dir = None
    else:
        temp_dir = tempfile.mkdtemp(prefix='repo-sandbox-')
        clone_path = Path(temp_dir)
    
    if verbose:
        click.echo(f"📥 Cloning repository: {repo_url}")
        click.echo(f"📁 Clone directory: {clone_path}")
    
    try:
        # Clone repository
        repo = Repo.clone_from(repo_url, clone_path)
        if verbose:
            click.echo(f"✅ Repository cloned successfully")
        
        # Generate Docker configuration
        ctx = click.get_current_context()
        ctx.invoke(
            generate,
            repo_path=clone_path,
            output=output,
            template='auto',
            python_version=None,
            include_tests=False,
            include_compose=False,
            copy_source=copy_source,
            verbose=verbose,
        )
        
    except Exception as e:
        click.echo(f"❌ Error processing Git repository: {e}", err=True)
        raise click.Abort()
    
    finally:
        # Cleanup if requested and using temporary directory
        if cleanup and temp_dir:
            try:
                shutil.rmtree(clone_path)
                if verbose:
                    click.echo(f"🧹 Cleaned up temporary directory")
            except Exception as e:
                click.echo(f"⚠️  Warning: Could not clean up {clone_path}: {e}")


@cli.command()
@click.argument('repo_path', type=click.Path(exists=True, path_type=Path))
@click.option('--format', 'output_format', default='json', 
              type=click.Choice(['json', 'yaml', 'text']),
              help='Output format for analysis results')
def analyze(repo_path: Path, output_format: str):
    """Analyze repository without generating Docker files"""
    
    analyzer = RepositoryAnalyzer()
    try:
        analysis = analyzer.analyze_repository(repo_path)
    except Exception as e:
        click.echo(f"❌ Error analyzing repository: {e}", err=True)
        raise click.Abort()
    
    if output_format == 'json':
        click.echo(json.dumps(analysis, indent=2))
    elif output_format == 'yaml':
        import yaml
        click.echo(yaml.dump(analysis, default_flow_style=False))
    else:  # text format
        click.echo(f"Repository Analysis for: {analysis['repo_name']}")
        click.echo(f"{'='*50}")
        click.echo(f"Python version: {analysis['python_version']}")
        click.echo(f"Python packages: {len(analysis['python_packages'])}")
        click.echo(f"System packages: {len(analysis['system_packages'])}")
        click.echo(f"Test framework: {analysis.get('test_config', {}).get('framework', 'None')}")
        click.echo(f"Confidence: {analysis.get('confidence', 1.0):.2f}")


if __name__ == '__main__':
    cli()
