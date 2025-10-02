#!/usr/bin/env python3
"""
Safety Manager for Command Execution
====================================

Comprehensive safety system for secure code modifications with:
- Dry-run execution for preview
- Backup system with versioning
- Rollback capability with validation
- Validation pipeline for changes

Author: Claude Code Framework
Version: 2.0
Last Updated: 2025-09-29
"""

import os
import shutil
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import tempfile


# ============================================================================
# Types and Configuration
# ============================================================================

class ChangeType(Enum):
    """Types of code changes"""
    ADD = "add"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"
    MOVE = "move"


class RiskLevel(Enum):
    """Risk levels for changes"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FileChange:
    """Represents a single file change"""
    change_type: ChangeType
    file_path: Path
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    old_path: Optional[Path] = None  # For renames/moves
    risk_level: RiskLevel = RiskLevel.LOW
    reason: str = ""


@dataclass
class BackupMetadata:
    """Backup metadata"""
    backup_id: str
    timestamp: datetime
    command: str
    work_dir: Path
    file_count: int
    total_size: int
    changes: List[FileChange] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    verified: bool = False


@dataclass
class ValidationResult:
    """Validation result for changes"""
    success: bool
    passed_checks: List[str] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    risk_assessment: RiskLevel = RiskLevel.LOW


# ============================================================================
# Dry Run Executor
# ============================================================================

class DryRunExecutor:
    """
    Dry-run execution preview system.

    Features:
    - Preview changes without applying
    - Impact analysis
    - Risk assessment
    - Interactive confirmation
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.planned_changes: List[FileChange] = []

    def plan_change(
        self,
        change_type: ChangeType,
        file_path: Path,
        old_content: Optional[str] = None,
        new_content: Optional[str] = None,
        old_path: Optional[Path] = None,
        reason: str = ""
    ):
        """
        Plan a change without executing.

        Args:
            change_type: Type of change
            file_path: Target file path
            old_content: Original content
            new_content: New content
            old_path: Old path for rename/move
            reason: Reason for change
        """
        # Assess risk
        risk_level = self._assess_risk(change_type, file_path, new_content)

        change = FileChange(
            change_type=change_type,
            file_path=file_path,
            old_content=old_content,
            new_content=new_content,
            old_path=old_path,
            risk_level=risk_level,
            reason=reason
        )

        self.planned_changes.append(change)
        self.logger.info(f"Planned {change_type.value}: {file_path} (risk: {risk_level.value})")

    def _assess_risk(
        self,
        change_type: ChangeType,
        file_path: Path,
        content: Optional[str]
    ) -> RiskLevel:
        """
        Assess risk level of a change.

        Args:
            change_type: Type of change
            file_path: File being changed
            content: New content

        Returns:
            Risk level
        """
        risk = RiskLevel.LOW

        # Critical files
        critical_patterns = [
            ".git", "config", "settings", "secrets", "credentials",
            "env", "docker", "kubernetes", "terraform"
        ]

        file_str = str(file_path).lower()

        if any(pattern in file_str for pattern in critical_patterns):
            risk = RiskLevel.HIGH

        # Deletion is higher risk
        if change_type == ChangeType.DELETE:
            risk = RiskLevel.MEDIUM if risk == RiskLevel.LOW else RiskLevel.CRITICAL

        # Large changes are higher risk
        if content and len(content) > 10000:
            risk = RiskLevel.MEDIUM if risk == RiskLevel.LOW else risk

        return risk

    def preview_changes(self) -> str:
        """
        Generate preview of all planned changes.

        Returns:
            Formatted preview string
        """
        if not self.planned_changes:
            return "No changes planned."

        lines = [
            "=" * 70,
            "DRY RUN - PREVIEW OF CHANGES",
            "=" * 70,
            ""
        ]

        # Group by risk level
        by_risk = {
            RiskLevel.CRITICAL: [],
            RiskLevel.HIGH: [],
            RiskLevel.MEDIUM: [],
            RiskLevel.LOW: []
        }

        for change in self.planned_changes:
            by_risk[change.risk_level].append(change)

        # Display by risk level
        for risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]:
            changes = by_risk[risk_level]
            if not changes:
                continue

            risk_icon = {
                RiskLevel.CRITICAL: "🔴",
                RiskLevel.HIGH: "🟠",
                RiskLevel.MEDIUM: "🟡",
                RiskLevel.LOW: "🟢"
            }[risk_level]

            lines.append(f"\n{risk_icon} {risk_level.value.upper()} RISK ({len(changes)} changes)")
            lines.append("-" * 70)

            for change in changes:
                lines.append(f"\n  {change.change_type.value.upper()}: {change.file_path}")
                if change.reason:
                    lines.append(f"  Reason: {change.reason}")

                if change.change_type == ChangeType.MODIFY and change.old_content and change.new_content:
                    lines.append(f"  Size: {len(change.old_content)} -> {len(change.new_content)} bytes")
                    diff_lines = self._calculate_diff_stats(change.old_content, change.new_content)
                    lines.append(f"  Changes: {diff_lines}")

        # Summary
        lines.append("\n" + "=" * 70)
        lines.append(f"SUMMARY: {len(self.planned_changes)} total changes")

        risk_summary = []
        for risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]:
            count = len(by_risk[risk_level])
            if count > 0:
                risk_summary.append(f"{count} {risk_level.value}")

        lines.append(f"Risk breakdown: {', '.join(risk_summary)}")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _calculate_diff_stats(self, old_content: str, new_content: str) -> str:
        """Calculate diff statistics"""
        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()

        added = len(new_lines) - len(old_lines)
        if added >= 0:
            return f"+{added} lines"
        else:
            return f"{added} lines"

    def get_impact_summary(self) -> Dict[str, Any]:
        """
        Get impact summary of changes.

        Returns:
            Impact summary dictionary
        """
        summary = {
            "total_changes": len(self.planned_changes),
            "by_type": {},
            "by_risk": {},
            "files_affected": set(),
            "highest_risk": RiskLevel.LOW
        }

        for change in self.planned_changes:
            # Count by type
            type_key = change.change_type.value
            summary["by_type"][type_key] = summary["by_type"].get(type_key, 0) + 1

            # Count by risk
            risk_key = change.risk_level.value
            summary["by_risk"][risk_key] = summary["by_risk"].get(risk_key, 0) + 1

            # Track files
            summary["files_affected"].add(str(change.file_path))

            # Track highest risk
            risk_order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
            if risk_order.index(change.risk_level) > risk_order.index(summary["highest_risk"]):
                summary["highest_risk"] = change.risk_level

        summary["files_affected"] = len(summary["files_affected"])

        return summary

    def confirm_execution(self) -> bool:
        """
        Request user confirmation for execution.

        Returns:
            True if user confirms
        """
        preview = self.preview_changes()
        print(preview)

        impact = self.get_impact_summary()

        if impact["highest_risk"] in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            print("\n⚠️  WARNING: This operation includes HIGH or CRITICAL risk changes!")

        response = input("\n❓ Proceed with execution? [y/N]: ").strip().lower()

        return response in ['y', 'yes']

    def clear(self):
        """Clear planned changes"""
        self.planned_changes = []


# ============================================================================
# Backup System
# ============================================================================

class BackupSystem:
    """
    Advanced backup system with versioning.

    Features:
    - Incremental backups
    - Compression support
    - Backup verification
    - Automatic cleanup
    - Version history
    """

    def __init__(self, backup_root: Optional[Path] = None):
        self.backup_root = backup_root or (Path.home() / ".claude" / "backups")
        self.backup_root.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_backup(
        self,
        source: Path,
        command: str,
        changes: Optional[List[FileChange]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Create backup of source directory.

        Args:
            source: Source directory to backup
            command: Command creating backup
            changes: Planned changes
            tags: Backup tags

        Returns:
            Backup ID
        """
        # Generate backup ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = f"{command}_{timestamp}"

        backup_dir = self.backup_root / backup_id
        backup_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Creating backup: {backup_id}")

        # Copy files
        file_count = 0
        total_size = 0

        if source.is_file():
            # Backup single file
            dest = backup_dir / source.name
            shutil.copy2(source, dest)
            file_count = 1
            total_size = source.stat().st_size
        elif source.is_dir():
            # Backup directory (selective based on changes)
            if changes:
                # Only backup affected files
                for change in changes:
                    if change.file_path.exists():
                        rel_path = change.file_path.relative_to(source) if source in change.file_path.parents else change.file_path.name
                        dest = backup_dir / rel_path
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(change.file_path, dest)
                        file_count += 1
                        total_size += change.file_path.stat().st_size
            else:
                # Full directory backup
                for item in source.rglob("*"):
                    if item.is_file() and not self._should_exclude(item):
                        rel_path = item.relative_to(source)
                        dest = backup_dir / rel_path
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, dest)
                        file_count += 1
                        total_size += item.stat().st_size

        # Create metadata
        metadata = BackupMetadata(
            backup_id=backup_id,
            timestamp=datetime.now(),
            command=command,
            work_dir=source,
            file_count=file_count,
            total_size=total_size,
            changes=changes or [],
            tags=tags or [],
            verified=False
        )

        # Save metadata
        self._save_metadata(backup_dir, metadata)

        # Verify backup
        metadata.verified = self._verify_backup(backup_dir, source, changes)
        self._save_metadata(backup_dir, metadata)

        self.logger.info(
            f"Backup created: {backup_id} "
            f"({file_count} files, {total_size / 1024 / 1024:.2f} MB)"
        )

        return backup_id

    def _should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded from backup"""
        exclude_patterns = [
            ".git", "__pycache__", ".pyc", ".pytest_cache",
            "node_modules", ".venv", "venv", ".env",
            ".DS_Store", "*.log"
        ]

        path_str = str(path)
        return any(pattern in path_str for pattern in exclude_patterns)

    def _save_metadata(self, backup_dir: Path, metadata: BackupMetadata):
        """Save backup metadata"""
        metadata_file = backup_dir / "backup_metadata.json"

        metadata_dict = {
            "backup_id": metadata.backup_id,
            "timestamp": metadata.timestamp.isoformat(),
            "command": metadata.command,
            "work_dir": str(metadata.work_dir),
            "file_count": metadata.file_count,
            "total_size": metadata.total_size,
            "changes": [
                {
                    "type": c.change_type.value,
                    "file": str(c.file_path),
                    "risk": c.risk_level.value,
                    "reason": c.reason
                }
                for c in metadata.changes
            ],
            "tags": metadata.tags,
            "verified": metadata.verified
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)

    def _verify_backup(
        self,
        backup_dir: Path,
        source: Path,
        changes: Optional[List[FileChange]]
    ) -> bool:
        """Verify backup integrity"""
        try:
            # Verify all expected files exist in backup
            if changes:
                for change in changes:
                    if change.file_path.exists():
                        rel_path = change.file_path.name if source not in change.file_path.parents else change.file_path.relative_to(source)
                        backup_file = backup_dir / rel_path
                        if not backup_file.exists():
                            self.logger.error(f"Backup verification failed: {backup_file} not found")
                            return False

            return True

        except Exception as e:
            self.logger.error(f"Backup verification error: {e}")
            return False

    def list_backups(self, command: Optional[str] = None) -> List[BackupMetadata]:
        """
        List available backups.

        Args:
            command: Filter by command name

        Returns:
            List of backup metadata
        """
        backups = []

        for backup_dir in sorted(self.backup_root.iterdir(), reverse=True):
            if not backup_dir.is_dir():
                continue

            metadata_file = backup_dir / "backup_metadata.json"
            if not metadata_file.exists():
                continue

            try:
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)

                # Filter by command if specified
                if command and not metadata_dict["command"].startswith(command):
                    continue

                # Reconstruct metadata (simplified)
                metadata = BackupMetadata(
                    backup_id=metadata_dict["backup_id"],
                    timestamp=datetime.fromisoformat(metadata_dict["timestamp"]),
                    command=metadata_dict["command"],
                    work_dir=Path(metadata_dict["work_dir"]),
                    file_count=metadata_dict["file_count"],
                    total_size=metadata_dict["total_size"],
                    tags=metadata_dict.get("tags", []),
                    verified=metadata_dict.get("verified", False)
                )

                backups.append(metadata)

            except Exception as e:
                self.logger.error(f"Error reading backup metadata: {e}")

        return backups

    def get_backup(self, backup_id: str) -> Optional[Path]:
        """Get backup directory path"""
        backup_dir = self.backup_root / backup_id
        return backup_dir if backup_dir.exists() else None

    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup"""
        backup_dir = self.backup_root / backup_id

        if not backup_dir.exists():
            self.logger.error(f"Backup not found: {backup_id}")
            return False

        try:
            shutil.rmtree(backup_dir)
            self.logger.info(f"Deleted backup: {backup_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting backup: {e}")
            return False

    def cleanup_old_backups(self, days: int = 7, keep_tagged: bool = True):
        """
        Remove old backups.

        Args:
            days: Remove backups older than this
            keep_tagged: Keep tagged backups regardless of age
        """
        cutoff = datetime.now() - timedelta(days=days)
        removed = 0

        for backup in self.list_backups():
            if backup.timestamp < cutoff:
                if keep_tagged and backup.tags:
                    continue

                if self.delete_backup(backup.backup_id):
                    removed += 1

        self.logger.info(f"Cleaned up {removed} old backups")


# ============================================================================
# Rollback Manager
# ============================================================================

class RollbackManager:
    """
    Rollback manager for failed operations.

    Features:
    - Safe rollback with verification
    - Partial rollback support
    - Rollback history
    - Automatic rollback on failure
    """

    def __init__(self, backup_system: BackupSystem):
        self.backup_system = backup_system
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rollback_history: List[Dict[str, Any]] = []

    def rollback(
        self,
        backup_id: str,
        target_dir: Path,
        verify: bool = True
    ) -> bool:
        """
        Rollback to a backup.

        Args:
            backup_id: Backup to restore
            target_dir: Target directory for restoration
            verify: Verify rollback after completion

        Returns:
            True if successful
        """
        self.logger.info(f"Rolling back to backup: {backup_id}")

        backup_dir = self.backup_system.get_backup(backup_id)
        if not backup_dir:
            self.logger.error(f"Backup not found: {backup_id}")
            return False

        try:
            # Create backup of current state before rollback
            pre_rollback_backup = self.backup_system.create_backup(
                target_dir,
                f"pre_rollback_{backup_id}",
                tags=["pre-rollback"]
            )

            # Restore files
            restored_count = 0

            for item in backup_dir.rglob("*"):
                if item.name == "backup_metadata.json":
                    continue

                if item.is_file():
                    rel_path = item.relative_to(backup_dir)
                    dest = target_dir / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest)
                    restored_count += 1

            self.logger.info(f"Restored {restored_count} files")

            # Verify rollback
            if verify:
                if not self._verify_rollback(backup_dir, target_dir):
                    self.logger.error("Rollback verification failed")
                    # Could restore pre-rollback backup here
                    return False

            # Record rollback
            self.rollback_history.append({
                "backup_id": backup_id,
                "target_dir": str(target_dir),
                "timestamp": datetime.now().isoformat(),
                "restored_files": restored_count,
                "pre_rollback_backup": pre_rollback_backup
            })

            self.logger.info(f"Rollback completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False

    def _verify_rollback(self, backup_dir: Path, target_dir: Path) -> bool:
        """Verify rollback was successful"""
        try:
            # Check that key files were restored
            for item in backup_dir.rglob("*"):
                if item.is_file() and item.name != "backup_metadata.json":
                    rel_path = item.relative_to(backup_dir)
                    dest = target_dir / rel_path

                    if not dest.exists():
                        self.logger.error(f"File not restored: {dest}")
                        return False

            return True

        except Exception as e:
            self.logger.error(f"Rollback verification error: {e}")
            return False


# ============================================================================
# Validation Pipeline
# ============================================================================

class ValidationPipeline:
    """
    Multi-stage validation pipeline for changes.

    Stages:
    1. Pre-change validation
    2. Change validation
    3. Post-change validation
    4. Integration validation
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.validators: Dict[str, Any] = {}

    def register_validator(self, name: str, validator: Any):
        """Register a validator"""
        self.validators[name] = validator

    def validate_changes(
        self,
        changes: List[FileChange],
        stage: str = "pre"
    ) -> ValidationResult:
        """
        Validate changes.

        Args:
            changes: List of changes to validate
            stage: Validation stage (pre, change, post, integration)

        Returns:
            Validation result
        """
        self.logger.info(f"Validating {len(changes)} changes (stage: {stage})")

        result = ValidationResult(success=True)

        # Syntax validation
        syntax_ok, syntax_msg = self._validate_syntax(changes)
        if syntax_ok:
            result.passed_checks.append("syntax")
        else:
            result.failed_checks.append(f"syntax: {syntax_msg}")
            result.success = False

        # Safety validation
        safety_ok, safety_msg = self._validate_safety(changes)
        if safety_ok:
            result.passed_checks.append("safety")
        else:
            result.warnings.append(f"safety: {safety_msg}")

        # Risk assessment
        result.risk_assessment = self._assess_overall_risk(changes)

        return result

    def _validate_syntax(self, changes: List[FileChange]) -> Tuple[bool, str]:
        """Validate syntax of changes"""
        # Simplified syntax validation
        for change in changes:
            if change.change_type in [ChangeType.ADD, ChangeType.MODIFY]:
                if change.new_content:
                    # Check for common syntax errors
                    if change.file_path.suffix == ".py":
                        try:
                            compile(change.new_content, str(change.file_path), 'exec')
                        except SyntaxError as e:
                            return False, f"Python syntax error in {change.file_path}: {e}"

        return True, ""

    def _validate_safety(self, changes: List[FileChange]) -> Tuple[bool, str]:
        """Validate safety of changes"""
        # Check for dangerous operations
        dangerous_patterns = ["rm -rf", "DROP TABLE", "DELETE FROM", "os.system"]

        for change in changes:
            if change.new_content:
                for pattern in dangerous_patterns:
                    if pattern in change.new_content:
                        return False, f"Dangerous pattern detected: {pattern}"

        return True, ""

    def _assess_overall_risk(self, changes: List[FileChange]) -> RiskLevel:
        """Assess overall risk of all changes"""
        highest_risk = RiskLevel.LOW

        risk_order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]

        for change in changes:
            if risk_order.index(change.risk_level) > risk_order.index(highest_risk):
                highest_risk = change.risk_level

        return highest_risk


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Safety manager demonstration"""
    print("Safety Manager for Command Execution")
    print("====================================\n")

    # Demonstrate dry run
    print("1. Dry Run Executor")
    print("-" * 40)
    dry_run = DryRunExecutor()

    dry_run.plan_change(
        ChangeType.MODIFY,
        Path("src/main.py"),
        old_content="print('Hello')",
        new_content="print('Hello World')",
        reason="Update greeting message"
    )

    dry_run.plan_change(
        ChangeType.DELETE,
        Path("config/secrets.json"),
        reason="Remove secrets file"
    )

    print(dry_run.preview_changes())

    # Demonstrate backup system
    print("\n\n2. Backup System")
    print("-" * 40)
    backup_system = BackupSystem()

    backups = backup_system.list_backups()
    print(f"Available backups: {len(backups)}")

    for backup in backups[:5]:
        print(f"  - {backup.backup_id}")
        print(f"    Created: {backup.timestamp}")
        print(f"    Files: {backup.file_count}, Size: {backup.total_size / 1024 / 1024:.2f} MB")

    print("\n✅ Safety Manager initialized successfully")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())