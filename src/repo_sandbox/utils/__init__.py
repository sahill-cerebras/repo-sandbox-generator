"""Utilities package initialization."""

from .file_utils import (
    find_files,
    read_text_file,
    ensure_directory,
    copy_file,
    get_file_size,
    is_text_file,
    clean_filename
)

from .git_utils import (
    is_git_repo,
    get_git_info,
    clone_repository,
    parse_git_url,
    get_repo_name_from_url
)

__all__ = [
    # File utilities
    'find_files',
    'read_text_file', 
    'ensure_directory',
    'copy_file',
    'get_file_size',
    'is_text_file',
    'clean_filename',
    
    # Git utilities
    'is_git_repo',
    'get_git_info',
    'clone_repository',
    'parse_git_url',
    'get_repo_name_from_url'
]
