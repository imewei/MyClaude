#!/usr/bin/env python3
"""
Code Modifier Utilities for Command Executors
Provides safe code modification operations with backup and rollback
"""

import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Callable
from datetime import datetime
import hashlib


class ModificationError(Exception):
    """Custom exception for code modification errors"""
    pass


class CodeModifier:
    """Utility class for safe code modifications"""

    def __init__(self, work_dir: Path = None):
        self.work_dir = work_dir or Path.cwd()
        self.backup_dir: Optional[Path] = None
        self.modifications: List[Dict] = []

    def create_backup(self, files: Optional[List[Path]] = None) -> Path:
        """
        Create backup of files before modification

        Args:
            files: List of files to backup (None for all tracked files)

        Returns:
            Path to backup directory
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.backup_dir = Path(tempfile.mkdtemp(prefix=f'backup_{timestamp}_'))

        if files is None:
            # Backup entire work directory
            shutil.copytree(self.work_dir, self.backup_dir / 'full_backup',
                           ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc',
                                                        'node_modules', 'venv', '.venv'))
        else:
            # Backup specific files
            for file_path in files:
                if not file_path.is_absolute():
                    file_path = self.work_dir / file_path

                if file_path.exists():
                    relative_path = file_path.relative_to(self.work_dir)
                    backup_path = self.backup_dir / relative_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, backup_path)

        return self.backup_dir

    def restore_backup(self) -> None:
        """Restore files from backup"""
        if not self.backup_dir or not self.backup_dir.exists():
            raise ModificationError("No backup available to restore")

        # Restore files from backup
        full_backup = self.backup_dir / 'full_backup'
        if full_backup.exists():
            # Full directory restore
            for item in full_backup.rglob('*'):
                if item.is_file():
                    relative_path = item.relative_to(full_backup)
                    target_path = self.work_dir / relative_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target_path)
        else:
            # Individual file restore
            for item in self.backup_dir.rglob('*'):
                if item.is_file():
                    relative_path = item.relative_to(self.backup_dir)
                    target_path = self.work_dir / relative_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target_path)

    def cleanup_backup(self) -> None:
        """Remove backup directory"""
        if self.backup_dir and self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
            self.backup_dir = None

    def modify_file(self, file_path: Path, modifier_func: Callable[[str], str],
                    description: str = "") -> bool:
        """
        Safely modify a file using a modifier function

        Args:
            file_path: Path to file to modify
            modifier_func: Function that takes file content and returns modified content
            description: Description of modification

        Returns:
            True if modification was successful
        """
        if not file_path.is_absolute():
            file_path = self.work_dir / file_path

        if not file_path.exists():
            raise ModificationError(f"File does not exist: {file_path}")

        try:
            # Read original content
            with open(file_path, 'r') as f:
                original_content = f.read()

            # Calculate hash for verification
            original_hash = hashlib.sha256(original_content.encode()).hexdigest()

            # Apply modification
            modified_content = modifier_func(original_content)

            # Write modified content
            with open(file_path, 'w') as f:
                f.write(modified_content)

            # Record modification
            self.modifications.append({
                'file': str(file_path),
                'description': description,
                'original_hash': original_hash,
                'timestamp': datetime.now().isoformat()
            })

            return True

        except Exception as e:
            raise ModificationError(f"Failed to modify {file_path}: {e}")

    def replace_in_file(self, file_path: Path, old_text: str, new_text: str,
                       count: int = -1) -> int:
        """
        Replace text in a file

        Args:
            file_path: Path to file
            old_text: Text to replace
            new_text: Replacement text
            count: Maximum number of replacements (-1 for all)

        Returns:
            Number of replacements made
        """
        replacements = 0

        def replace_func(content: str) -> str:
            nonlocal replacements
            if count == -1:
                new_content = content.replace(old_text, new_text)
                replacements = content.count(old_text)
            else:
                new_content = content.replace(old_text, new_text, count)
                replacements = min(content.count(old_text), count)
            return new_content

        self.modify_file(file_path, replace_func,
                        f"Replace '{old_text[:50]}...' with '{new_text[:50]}...'")

        return replacements

    def insert_at_line(self, file_path: Path, line_number: int, text: str) -> None:
        """
        Insert text at a specific line

        Args:
            file_path: Path to file
            line_number: Line number to insert at (1-based)
            text: Text to insert
        """
        def insert_func(content: str) -> str:
            lines = content.splitlines(keepends=True)
            if line_number < 1 or line_number > len(lines) + 1:
                raise ValueError(f"Invalid line number: {line_number}")

            lines.insert(line_number - 1, text + '\n')
            return ''.join(lines)

        self.modify_file(file_path, insert_func,
                        f"Insert at line {line_number}")

    def delete_lines(self, file_path: Path, start_line: int, end_line: int) -> None:
        """
        Delete lines from a file

        Args:
            file_path: Path to file
            start_line: Starting line number (1-based, inclusive)
            end_line: Ending line number (1-based, inclusive)
        """
        def delete_func(content: str) -> str:
            lines = content.splitlines(keepends=True)
            if start_line < 1 or end_line > len(lines) or start_line > end_line:
                raise ValueError(f"Invalid line range: {start_line}-{end_line}")

            del lines[start_line - 1:end_line]
            return ''.join(lines)

        self.modify_file(file_path, delete_func,
                        f"Delete lines {start_line}-{end_line}")

    def add_import(self, file_path: Path, import_statement: str,
                   position: str = 'top') -> None:
        """
        Add an import statement to a file

        Args:
            file_path: Path to Python file
            import_statement: Import statement to add
            position: Where to add ('top', 'after_docstring')
        """
        def add_import_func(content: str) -> str:
            lines = content.splitlines(keepends=True)

            # Check if import already exists
            if any(import_statement in line for line in lines):
                return content

            insert_pos = 0

            if position == 'after_docstring':
                # Find end of module docstring
                in_docstring = False
                for i, line in enumerate(lines):
                    if '"""' in line or "'''" in line:
                        if not in_docstring:
                            in_docstring = True
                        else:
                            insert_pos = i + 1
                            break

            # Skip shebang and encoding declarations
            while insert_pos < len(lines) and \
                  (lines[insert_pos].startswith('#!') or
                   lines[insert_pos].startswith('# -*- coding')):
                insert_pos += 1

            # Insert import
            lines.insert(insert_pos, import_statement + '\n')
            return ''.join(lines)

        self.modify_file(file_path, add_import_func,
                        f"Add import: {import_statement}")

    def remove_import(self, file_path: Path, import_pattern: str) -> int:
        """
        Remove import statements matching a pattern

        Args:
            file_path: Path to Python file
            import_pattern: Pattern to match (e.g., 'from foo import')

        Returns:
            Number of imports removed
        """
        removed_count = 0

        def remove_import_func(content: str) -> str:
            nonlocal removed_count
            lines = content.splitlines(keepends=True)
            new_lines = []

            for line in lines:
                if import_pattern in line and \
                   (line.strip().startswith('import ') or
                    line.strip().startswith('from ')):
                    removed_count += 1
                    continue
                new_lines.append(line)

            return ''.join(new_lines)

        self.modify_file(file_path, remove_import_func,
                        f"Remove imports matching: {import_pattern}")

        return removed_count

    def format_file(self, file_path: Path, formatter: str = 'black') -> bool:
        """
        Format a file using a code formatter

        Args:
            file_path: Path to file
            formatter: Formatter to use ('black', 'prettier', 'rustfmt', etc.)

        Returns:
            True if formatting was successful
        """
        import subprocess

        if not file_path.is_absolute():
            file_path = self.work_dir / file_path

        try:
            if formatter == 'black':
                subprocess.run(['black', str(file_path)], check=True, capture_output=True)
            elif formatter == 'prettier':
                subprocess.run(['prettier', '--write', str(file_path)],
                             check=True, capture_output=True)
            elif formatter == 'rustfmt':
                subprocess.run(['rustfmt', str(file_path)], check=True, capture_output=True)
            else:
                raise ValueError(f"Unknown formatter: {formatter}")

            return True

        except subprocess.CalledProcessError:
            return False
        except FileNotFoundError:
            raise ModificationError(f"Formatter '{formatter}' not found")

    def get_modification_summary(self) -> Dict:
        """Get summary of all modifications made"""
        return {
            'total_modifications': len(self.modifications),
            'files_modified': len(set(m['file'] for m in self.modifications)),
            'modifications': self.modifications,
            'backup_location': str(self.backup_dir) if self.backup_dir else None
        }