"""File utilities module."""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


def find_files(directory: Path, patterns: List[str], 
               max_depth: Optional[int] = None) -> List[Path]:
    """
    Find files matching patterns in directory.
    
    Args:
        directory: Directory to search in
        patterns: List of glob patterns to match
        max_depth: Maximum depth to search (None for unlimited)
        
    Returns:
        List of matching file paths
    """
    matches = []
    
    for pattern in patterns:
        try:
            if max_depth is None:
                matches.extend(directory.rglob(pattern))
            else:
                # Simple depth limiting by checking path parts
                for match in directory.rglob(pattern):
                    relative_parts = match.relative_to(directory).parts
                    if len(relative_parts) <= max_depth:
                        matches.append(match)
        except Exception as e:
            logger.debug(f"Error searching for pattern {pattern}: {e}")
    
    return list(set(matches))  # Remove duplicates


def read_text_file(file_path: Path, encoding: str = 'utf-8', 
                   errors: str = 'ignore') -> Optional[str]:
    """
    Safely read text file content.
    
    Args:
        file_path: Path to file
        encoding: Text encoding
        errors: How to handle encoding errors
        
    Returns:
        File content or None if error
    """
    try:
        return file_path.read_text(encoding=encoding, errors=errors)
    except Exception as e:
        logger.debug(f"Error reading {file_path}: {e}")
        return None


def ensure_directory(directory: Path) -> bool:
    """
    Ensure directory exists, creating if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        True if directory exists/created, False on error
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {e}")
        return False


def copy_file(src: Path, dst: Path, overwrite: bool = False) -> bool:
    """
    Copy file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Whether to overwrite existing files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if dst.exists() and not overwrite:
            logger.warning(f"Destination {dst} exists, skipping copy")
            return False
        
        ensure_directory(dst.parent)
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        logger.error(f"Error copying {src} to {dst}: {e}")
        return False


def get_file_size(file_path: Path) -> Optional[int]:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes or None if error
    """
    try:
        return file_path.stat().st_size
    except Exception as e:
        logger.debug(f"Error getting size of {file_path}: {e}")
        return None


def is_text_file(file_path: Path, max_check_bytes: int = 1024) -> bool:
    """
    Check if file appears to be a text file.
    
    Args:
        file_path: Path to file
        max_check_bytes: Number of bytes to check
        
    Returns:
        True if appears to be text file
    """
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(max_check_bytes)
            
        # Check for null bytes (common in binary files)
        if b'\x00' in chunk:
            return False
        
        # Try to decode as UTF-8
        try:
            chunk.decode('utf-8')
            return True
        except UnicodeDecodeError:
            return False
            
    except Exception:
        return False


def clean_filename(filename: str) -> str:
    """
    Clean filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    import re
    
    # Replace invalid characters with underscores
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove multiple underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    
    # Remove leading/trailing underscores and dots
    cleaned = cleaned.strip('_.')
    
    return cleaned or 'unnamed'
