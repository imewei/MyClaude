#!/usr/bin/env python3
"""
Feedback Collector - Collects and analyzes user feedback.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Feedback:
    """User feedback entry."""
    user_id: str
    feedback_type: str  # 'satisfaction', 'bug', 'feature_request', 'performance'
    rating: Optional[int]  # 1-5 for satisfaction surveys
    comment: str
    project_name: Optional[str]
    scenario_name: Optional[str]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'user_id': self.user_id,
            'feedback_type': self.feedback_type,
            'rating': self.rating,
            'comment': self.comment,
            'project_name': self.project_name,
            'scenario_name': self.scenario_name,
            'timestamp': self.timestamp.isoformat()
        }


class FeedbackCollector:
    """Collects and stores user feedback."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize feedback collector."""
        if db_path is None:
            validation_dir = Path(__file__).parent.parent
            db_path = validation_dir / "data" / "feedback.db"

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    rating INTEGER,
                    comment TEXT,
                    project_name TEXT,
                    scenario_name TEXT,
                    timestamp TEXT NOT NULL
                )
            """)

    def collect(self, feedback: Feedback) -> None:
        """Store feedback."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO feedback
                (user_id, feedback_type, rating, comment, project_name, scenario_name, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.user_id,
                feedback.feedback_type,
                feedback.rating,
                feedback.comment,
                feedback.project_name,
                feedback.scenario_name,
                feedback.timestamp.isoformat()
            ))

    def get_average_satisfaction(self) -> float:
        """Get average satisfaction rating."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT AVG(rating) FROM feedback
                WHERE feedback_type = 'satisfaction' AND rating IS NOT NULL
            """)
            result = cursor.fetchone()[0]
            return result if result else 0.0

    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get feedback summary."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT feedback_type, COUNT(*), AVG(rating)
                FROM feedback
                GROUP BY feedback_type
            """)

            summary = {}
            for row in cursor.fetchall():
                summary[row[0]] = {
                    'count': row[1],
                    'avg_rating': row[2] if row[2] else None
                }

            return summary