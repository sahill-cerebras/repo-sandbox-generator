"""Git utilities module."""

import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def is_git_repo(directory: Path) -> bool:
    """
    Check if directory is a git repository.
    
    Args:
        directory: Directory to check
        
    Returns:
        True if git repository
    """
    return (directory / '.git').exists()


def get_git_info(directory: Path) -> Dict[str, Any]:
    """
    Get git repository information.
    
    Args:
        directory: Git repository directory
        
    Returns:
        Dictionary with git info
    """
    info = {}
    
    if not is_git_repo(directory):
        return info
    
    try:
        # Get current branch
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            info['branch'] = result.stdout.strip()
    except Exception as e:
        logger.debug(f"Error getting git branch: {e}")
    
    try:
        # Get remote origin URL
        result = subprocess.run(
            ['git', 'remote', 'get-url', 'origin'],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            info['remote_url'] = result.stdout.strip()
    except Exception as e:
        logger.debug(f"Error getting git remote: {e}")
    
    try:
        # Get last commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            info['commit_hash'] = result.stdout.strip()
    except Exception as e:
        logger.debug(f"Error getting git commit: {e}")
    
    return info


def clone_repository(repo_url: str, destination: Path, 
                    branch: Optional[str] = None) -> bool:
    """
    Clone git repository to destination.
    
    Args:
        repo_url: Repository URL
        destination: Destination directory
        branch: Specific branch to clone
        
    Returns:
        True if successful
    """
    try:
        cmd = ['git', 'clone']
        
        if branch:
            cmd.extend(['-b', branch])
        
        cmd.extend([repo_url, str(destination)])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode == 0:
            logger.info(f"Successfully cloned {repo_url} to {destination}")
            return True
        else:
            logger.error(f"Git clone failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"Git clone timeout for {repo_url}")
        return False
    except Exception as e:
        logger.error(f"Error cloning repository: {e}")
        return False


def parse_git_url(url: str) -> Optional[Dict[str, str]]:
    """
    Parse git URL to extract components.
    
    Args:
        url: Git repository URL
        
    Returns:
        Dictionary with URL components or None
    """
    try:
        # Handle SSH URLs
        if url.startswith('git@'):
            # Format: git@github.com:owner/repo.git
            parts = url.replace('git@', '').replace(':', '/').split('/')
            if len(parts) >= 3:
                return {
                    'host': parts[0],
                    'owner': parts[1],
                    'repo': parts[2].replace('.git', '')
                }
        else:
            # Handle HTTP/HTTPS URLs
            parsed = urlparse(url)
            if parsed.netloc and parsed.path:
                path_parts = parsed.path.strip('/').split('/')
                if len(path_parts) >= 2:
                    return {
                        'host': parsed.netloc,
                        'owner': path_parts[0],
                        'repo': path_parts[1].replace('.git', '')
                    }
    except Exception as e:
        logger.debug(f"Error parsing git URL {url}: {e}")
    
    return None


def get_repo_name_from_url(url: str) -> Optional[str]:
    """
    Extract repository name from URL.
    
    Args:
        url: Git repository URL
        
    Returns:
        Repository name or None
    """
    parsed = parse_git_url(url)
    return parsed['repo'] if parsed else None
