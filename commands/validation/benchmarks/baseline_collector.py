#!/usr/bin/env python3
"""
Baseline Collector - Collects and stores baseline metrics for comparison.
"""

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Baseline:
    """Baseline metrics for a project/scenario combination."""
    project_name: str
    scenario_name: str
    metrics: Dict[str, Any]
    timestamp: datetime
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'metrics': json.dumps(self.metrics)
        }


class BaselineCollector:
    """Collects and manages baseline metrics."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize baseline collector.

        Args:
            db_path: Path to SQLite database
        """
        if db_path is None:
            validation_dir = Path(__file__).parent.parent
            db_path = validation_dir / "data" / "baselines.db"

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_name TEXT NOT NULL,
                    scenario_name TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    version TEXT NOT NULL,
                    UNIQUE(project_name, scenario_name, version)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_project_scenario
                ON baselines(project_name, scenario_name)
            """)

    def collect(
        self,
        project_name: str,
        scenario_name: str,
        metrics: Dict[str, Any]
    ) -> Baseline:
        """Collect baseline metrics.

        Args:
            project_name: Name of project
            scenario_name: Name of scenario
            metrics: Metrics dictionary

        Returns:
            Baseline object
        """
        baseline = Baseline(
            project_name=project_name,
            scenario_name=scenario_name,
            metrics=metrics,
            timestamp=datetime.now()
        )

        self.store(baseline)
        return baseline

    def store(self, baseline: Baseline) -> None:
        """Store baseline in database.

        Args:
            baseline: Baseline to store
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO baselines
                (project_name, scenario_name, metrics, timestamp, version)
                VALUES (?, ?, ?, ?, ?)
            """, (
                baseline.project_name,
                baseline.scenario_name,
                json.dumps(baseline.metrics),
                baseline.timestamp.isoformat(),
                baseline.version
            ))

    def get(
        self,
        project_name: str,
        scenario_name: str,
        version: str = "1.0"
    ) -> Optional[Baseline]:
        """Get baseline for project/scenario.

        Args:
            project_name: Name of project
            scenario_name: Name of scenario
            version: Baseline version

        Returns:
            Baseline object or None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM baselines
                WHERE project_name = ? AND scenario_name = ? AND version = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (project_name, scenario_name, version))

            row = cursor.fetchone()
            if not row:
                return None

            return Baseline(
                project_name=row['project_name'],
                scenario_name=row['scenario_name'],
                metrics=json.loads(row['metrics']),
                timestamp=datetime.fromisoformat(row['timestamp']),
                version=row['version']
            )

    def get_all(self, project_name: Optional[str] = None) -> List[Baseline]:
        """Get all baselines, optionally filtered by project.

        Args:
            project_name: Optional project name filter

        Returns:
            List of baselines
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if project_name:
                cursor = conn.execute("""
                    SELECT * FROM baselines
                    WHERE project_name = ?
                    ORDER BY timestamp DESC
                """, (project_name,))
            else:
                cursor = conn.execute("""
                    SELECT * FROM baselines
                    ORDER BY timestamp DESC
                """)

            baselines = []
            for row in cursor.fetchall():
                baselines.append(Baseline(
                    project_name=row['project_name'],
                    scenario_name=row['scenario_name'],
                    metrics=json.loads(row['metrics']),
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    version=row['version']
                ))

            return baselines