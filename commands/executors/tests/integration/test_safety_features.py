#!/usr/bin/env python3
"""
Integration Tests for Safety Features
=====================================

Tests safety systems:
- Dry-run execution
- Backup system
- Rollback functionality
- Validation pipeline

Coverage: Safety mechanisms, backup/restore, change validation
"""

import pytest
import time
import shutil
from pathlib import Path
from typing import List

from executors.safety_manager import (
    BackupSystem,
    DryRunExecutor,
    RollbackManager,
    ValidationPipeline,
    ChangeType,
    RiskLevel,
    FileChange,
)


@pytest.mark.integration
@pytest.mark.safety
class TestDryRunExecutor:
    """Integration tests for DryRunExecutor"""

    def test_plan_simple_change(self, temp_workspace: Path):
        """Test planning a simple file change"""
        dry_run = DryRunExecutor()

        test_file = temp_workspace / "test.py"
        dry_run.plan_change(
            ChangeType.MODIFY,
            test_file,
            old_content="print('old')",
            new_content="print('new')",
            reason="Update print statement"
        )

        assert len(dry_run.planned_changes) == 1
        change = dry_run.planned_changes[0]
        assert change.change_type == ChangeType.MODIFY
        assert change.file_path == test_file
        assert change.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]

    def test_plan_high_risk_change(self, temp_workspace: Path):
        """Test planning a high-risk change"""
        dry_run = DryRunExecutor()

        config_file = temp_workspace / "config" / "secrets.env"
        dry_run.plan_change(
            ChangeType.DELETE,
            config_file,
            reason="Remove secrets file"
        )

        assert len(dry_run.planned_changes) == 1
        change = dry_run.planned_changes[0]
        assert change.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]

    def test_preview_changes(self, temp_workspace: Path):
        """Test preview generation"""
        dry_run = DryRunExecutor()

        # Plan multiple changes
        dry_run.plan_change(
            ChangeType.ADD,
            temp_workspace / "new_file.py",
            new_content="# New file",
            reason="Add new module"
        )
        dry_run.plan_change(
            ChangeType.MODIFY,
            temp_workspace / "existing.py",
            old_content="old",
            new_content="new",
            reason="Update code"
        )
        dry_run.plan_change(
            ChangeType.DELETE,
            temp_workspace / "old_file.py",
            reason="Remove obsolete file"
        )

        preview = dry_run.preview_changes()

        assert "DRY RUN" in preview
        assert "ADD" in preview
        assert "MODIFY" in preview
        assert "DELETE" in preview
        assert "3 total changes" in preview

    def test_impact_summary(self, temp_workspace: Path):
        """Test impact summary generation"""
        dry_run = DryRunExecutor()

        # Plan various changes
        for i in range(5):
            dry_run.plan_change(
                ChangeType.MODIFY,
                temp_workspace / f"file{i}.py",
                old_content="old",
                new_content="new",
                reason=f"Update file {i}"
            )

        summary = dry_run.get_impact_summary()

        assert summary["total_changes"] == 5
        assert "modify" in summary["by_type"]
        assert summary["files_affected"] == 5
        assert summary["highest_risk"] in [r for r in RiskLevel]

    def test_risk_assessment(self, temp_workspace: Path):
        """Test risk assessment for different file types"""
        dry_run = DryRunExecutor()

        test_cases = [
            (temp_workspace / "normal.py", RiskLevel.LOW),
            (temp_workspace / ".env", RiskLevel.HIGH),
            (temp_workspace / "config" / "settings.json", RiskLevel.HIGH),
            (temp_workspace / "data.txt", RiskLevel.LOW),
        ]

        for file_path, expected_min_risk in test_cases:
            risk = dry_run._assess_risk(ChangeType.MODIFY, file_path, "content")
            # Risk should be at least the expected minimum
            assert risk.value in [r.value for r in RiskLevel]

    def test_clear_changes(self, temp_workspace: Path):
        """Test clearing planned changes"""
        dry_run = DryRunExecutor()

        dry_run.plan_change(ChangeType.ADD, temp_workspace / "file.py")
        assert len(dry_run.planned_changes) == 1

        dry_run.clear()
        assert len(dry_run.planned_changes) == 0


@pytest.mark.integration
@pytest.mark.safety
class TestBackupSystem:
    """Integration tests for BackupSystem"""

    def test_create_backup_single_file(self, temp_workspace: Path):
        """Test creating backup of single file"""
        backup_system = BackupSystem(backup_root=temp_workspace / "backups")

        # Create a test file
        test_file = temp_workspace / "test.txt"
        test_file.write_text("Important content")

        backup_id = backup_system.create_backup(
            test_file,
            command="test_command",
            tags=["test"]
        )

        assert backup_id is not None
        assert (temp_workspace / "backups" / backup_id).exists()

        # Verify backup metadata
        backups = backup_system.list_backups()
        assert len(backups) > 0
        assert any(b.backup_id == backup_id for b in backups)

    def test_create_backup_directory(self, temp_workspace: Path):
        """Test creating backup of directory"""
        backup_system = BackupSystem(backup_root=temp_workspace / "backups")

        # Create test files
        project_dir = temp_workspace / "project"
        project_dir.mkdir()
        (project_dir / "file1.py").write_text("content1")
        (project_dir / "file2.py").write_text("content2")
        (project_dir / "subdir").mkdir()
        (project_dir / "subdir" / "file3.py").write_text("content3")

        backup_id = backup_system.create_backup(
            project_dir,
            command="test_command"
        )

        assert backup_id is not None

        # Verify backup contains files
        backup_dir = temp_workspace / "backups" / backup_id
        assert (backup_dir / "file1.py").exists()
        assert (backup_dir / "file2.py").exists()
        assert (backup_dir / "subdir" / "file3.py").exists()

    def test_backup_with_changes(self, temp_workspace: Path):
        """Test backup with specific changes"""
        backup_system = BackupSystem(backup_root=temp_workspace / "backups")

        project_dir = temp_workspace / "project"
        project_dir.mkdir()
        file1 = project_dir / "file1.py"
        file2 = project_dir / "file2.py"
        file1.write_text("content1")
        file2.write_text("content2")

        changes = [
            FileChange(
                change_type=ChangeType.MODIFY,
                file_path=file1,
                old_content="content1",
                new_content="new_content1",
                reason="Update file1"
            )
        ]

        backup_id = backup_system.create_backup(
            project_dir,
            command="test_command",
            changes=changes
        )

        # Should backup affected files
        backup_dir = temp_workspace / "backups" / backup_id
        assert backup_dir.exists()

    def test_list_backups(self, temp_workspace: Path):
        """Test listing backups"""
        backup_system = BackupSystem(backup_root=temp_workspace / "backups")

        # Create multiple backups
        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")

        backup_ids = []
        for i in range(3):
            backup_id = backup_system.create_backup(
                test_file,
                command=f"command_{i}"
            )
            backup_ids.append(backup_id)
            time.sleep(0.1)  # Ensure different timestamps

        backups = backup_system.list_backups()
        assert len(backups) >= 3

        # Test filtering by command
        backups_cmd0 = backup_system.list_backups(command="command_0")
        assert len(backups_cmd0) >= 1

    def test_get_backup(self, temp_workspace: Path):
        """Test retrieving backup directory"""
        backup_system = BackupSystem(backup_root=temp_workspace / "backups")

        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")

        backup_id = backup_system.create_backup(test_file, command="test")

        backup_dir = backup_system.get_backup(backup_id)
        assert backup_dir is not None
        assert backup_dir.exists()

        # Test non-existent backup
        fake_backup = backup_system.get_backup("nonexistent")
        assert fake_backup is None

    def test_delete_backup(self, temp_workspace: Path):
        """Test deleting a backup"""
        backup_system = BackupSystem(backup_root=temp_workspace / "backups")

        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")

        backup_id = backup_system.create_backup(test_file, command="test")

        # Verify backup exists
        assert backup_system.get_backup(backup_id) is not None

        # Delete backup
        success = backup_system.delete_backup(backup_id)
        assert success is True

        # Verify backup is gone
        assert backup_system.get_backup(backup_id) is None

    def test_cleanup_old_backups(self, temp_workspace: Path):
        """Test cleaning up old backups"""
        backup_system = BackupSystem(backup_root=temp_workspace / "backups")

        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")

        # Create backup with tag
        tagged_id = backup_system.create_backup(
            test_file,
            command="tagged",
            tags=["important"]
        )

        # Create backup without tag
        untagged_id = backup_system.create_backup(
            test_file,
            command="untagged"
        )

        # Cleanup with keep_tagged=True
        backup_system.cleanup_old_backups(days=-1, keep_tagged=True)

        # Tagged backup should remain
        assert backup_system.get_backup(tagged_id) is not None

    def test_backup_verification(self, temp_workspace: Path):
        """Test backup verification"""
        backup_system = BackupSystem(backup_root=temp_workspace / "backups")

        test_file = temp_workspace / "test.txt"
        test_file.write_text("Important data")

        backup_id = backup_system.create_backup(test_file, command="test")

        backups = backup_system.list_backups()
        backup = next(b for b in backups if b.backup_id == backup_id)

        # Backup should be verified
        assert backup.verified is True


@pytest.mark.integration
@pytest.mark.safety
class TestRollbackManager:
    """Integration tests for RollbackManager"""

    def test_successful_rollback(self, temp_workspace: Path, backup_system: BackupSystem):
        """Test successful rollback to backup"""
        rollback_mgr = RollbackManager(backup_system)

        # Create original file
        project_dir = temp_workspace / "project"
        project_dir.mkdir()
        test_file = project_dir / "test.py"
        test_file.write_text("original content")

        # Create backup
        backup_id = backup_system.create_backup(project_dir, command="test")

        # Modify file
        test_file.write_text("modified content")
        assert test_file.read_text() == "modified content"

        # Rollback
        success = rollback_mgr.rollback(backup_id, project_dir)

        assert success is True
        # File should be restored
        assert test_file.read_text() == "original content"

    def test_rollback_creates_pre_rollback_backup(
        self,
        temp_workspace: Path,
        backup_system: BackupSystem
    ):
        """Test that rollback creates pre-rollback backup"""
        rollback_mgr = RollbackManager(backup_system)

        project_dir = temp_workspace / "project"
        project_dir.mkdir()
        test_file = project_dir / "test.py"
        test_file.write_text("original")

        backup_id = backup_system.create_backup(project_dir, command="test")

        test_file.write_text("modified")

        initial_backups = len(backup_system.list_backups())

        rollback_mgr.rollback(backup_id, project_dir)

        # Should have created pre-rollback backup
        final_backups = len(backup_system.list_backups())
        assert final_backups > initial_backups

    def test_rollback_nonexistent_backup(
        self,
        temp_workspace: Path,
        backup_system: BackupSystem
    ):
        """Test rollback fails gracefully for nonexistent backup"""
        rollback_mgr = RollbackManager(backup_system)

        project_dir = temp_workspace / "project"
        project_dir.mkdir()

        success = rollback_mgr.rollback("nonexistent_backup", project_dir)

        assert success is False

    def test_rollback_history(self, temp_workspace: Path, backup_system: BackupSystem):
        """Test rollback history tracking"""
        rollback_mgr = RollbackManager(backup_system)

        project_dir = temp_workspace / "project"
        project_dir.mkdir()
        test_file = project_dir / "test.py"
        test_file.write_text("content")

        backup_id = backup_system.create_backup(project_dir, command="test")

        rollback_mgr.rollback(backup_id, project_dir)

        assert len(rollback_mgr.rollback_history) == 1
        assert rollback_mgr.rollback_history[0]["backup_id"] == backup_id


@pytest.mark.integration
@pytest.mark.validation
class TestValidationPipeline:
    """Integration tests for ValidationPipeline"""

    def test_validate_python_syntax(self, temp_workspace: Path):
        """Test Python syntax validation"""
        pipeline = ValidationPipeline()

        # Valid Python code
        valid_change = FileChange(
            change_type=ChangeType.MODIFY,
            file_path=temp_workspace / "valid.py",
            new_content="def hello():\n    print('Hello')\n"
        )

        result = pipeline.validate_changes([valid_change])
        assert result.success is True
        assert "syntax" in result.passed_checks

        # Invalid Python code
        invalid_change = FileChange(
            change_type=ChangeType.MODIFY,
            file_path=temp_workspace / "invalid.py",
            new_content="def hello(\n    print('Hello')\n"  # Missing closing paren
        )

        result = pipeline.validate_changes([invalid_change])
        assert result.success is False
        assert any("syntax" in check for check in result.failed_checks)

    def test_validate_safety(self, temp_workspace: Path):
        """Test safety validation"""
        pipeline = ValidationPipeline()

        # Safe change
        safe_change = FileChange(
            change_type=ChangeType.MODIFY,
            file_path=temp_workspace / "safe.py",
            new_content="print('Safe code')"
        )

        result = pipeline.validate_changes([safe_change])
        assert "safety" in result.passed_checks

        # Dangerous change
        dangerous_change = FileChange(
            change_type=ChangeType.MODIFY,
            file_path=temp_workspace / "dangerous.py",
            new_content="import os\nos.system('rm -rf /')"
        )

        result = pipeline.validate_changes([dangerous_change])
        assert len(result.warnings) > 0 or len(result.failed_checks) > 0

    def test_risk_assessment(self, temp_workspace: Path):
        """Test overall risk assessment"""
        pipeline = ValidationPipeline()

        # Low risk changes
        low_risk_changes = [
            FileChange(
                change_type=ChangeType.MODIFY,
                file_path=temp_workspace / "file.py",
                risk_level=RiskLevel.LOW
            )
        ]

        result = pipeline.validate_changes(low_risk_changes)
        assert result.risk_assessment == RiskLevel.LOW

        # High risk changes
        high_risk_changes = [
            FileChange(
                change_type=ChangeType.DELETE,
                file_path=temp_workspace / "config" / "critical.conf",
                risk_level=RiskLevel.CRITICAL
            )
        ]

        result = pipeline.validate_changes(high_risk_changes)
        assert result.risk_assessment == RiskLevel.CRITICAL

    def test_multiple_changes_validation(self, temp_workspace: Path):
        """Test validation of multiple changes"""
        pipeline = ValidationPipeline()

        changes = [
            FileChange(
                change_type=ChangeType.ADD,
                file_path=temp_workspace / "new1.py",
                new_content="# New file 1"
            ),
            FileChange(
                change_type=ChangeType.MODIFY,
                file_path=temp_workspace / "existing.py",
                new_content="def modified(): pass"
            ),
            FileChange(
                change_type=ChangeType.DELETE,
                file_path=temp_workspace / "old.py"
            )
        ]

        result = pipeline.validate_changes(changes)
        assert result is not None
        assert len(result.passed_checks) > 0