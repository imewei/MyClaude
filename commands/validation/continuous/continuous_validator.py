#!/usr/bin/env python3
"""
Continuous Validator - Runs validation continuously and monitors for regressions.
"""

import asyncio
import logging
import schedule
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.executor import ValidationExecutor
from validation.benchmarks.regression_detector import RegressionDetector


class ContinuousValidator:
    """Runs validation continuously on a schedule."""

    def __init__(
        self,
        validation_dir: Optional[Path] = None,
        schedule_interval: str = "daily"
    ):
        """Initialize continuous validator.

        Args:
            validation_dir: Directory containing validation configuration
            schedule_interval: 'hourly', 'daily', 'weekly'
        """
        self.validation_dir = validation_dir or Path(__file__).parent.parent
        self.schedule_interval = schedule_interval

        self.executor = ValidationExecutor(validation_dir=self.validation_dir)
        self.regression_detector = RegressionDetector()

        self._setup_logging()
        self._setup_schedule()

    def _setup_logging(self) -> None:
        """Setup logging."""
        log_dir = self.validation_dir / "logs" / "continuous"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"continuous_{datetime.now().strftime('%Y%m%d')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_schedule(self) -> None:
        """Setup validation schedule."""
        if self.schedule_interval == "hourly":
            schedule.every().hour.do(self.run_validation)
        elif self.schedule_interval == "daily":
            schedule.every().day.at("02:00").do(self.run_validation)
        elif self.schedule_interval == "weekly":
            schedule.every().monday.at("02:00").do(self.run_validation)
        else:
            raise ValueError(f"Invalid schedule interval: {self.schedule_interval}")

    def run_validation(self) -> None:
        """Run a validation cycle."""
        self.logger.info("Starting scheduled validation run")

        try:
            # Run validation
            results = self.executor.run_validation()

            # Check for regressions
            regressions_found = False
            for result in results:
                should_fail, regressions = self.regression_detector.check_for_regressions(
                    result.project_name,
                    result.scenario_name,
                    result.metrics,
                    fail_on_severity='high'
                )

                if regressions:
                    regressions_found = True
                    self.logger.warning(
                        f"Regressions detected in {result.project_name}/{result.scenario_name}"
                    )
                    for regression in regressions:
                        self.logger.warning(f"  {regression}")

            # Generate report
            report_paths = self.executor.generate_report(
                formats=['html', 'json', 'markdown']
            )

            self.logger.info(f"Validation completed. Reports: {report_paths}")

            if regressions_found:
                self._send_alert("Regressions detected in validation run")

        except Exception as e:
            self.logger.error(f"Validation run failed: {e}", exc_info=True)
            self._send_alert(f"Validation run failed: {e}")

    def _send_alert(self, message: str) -> None:
        """Send alert (placeholder for actual alert system).

        Args:
            message: Alert message
        """
        self.logger.critical(f"ALERT: {message}")
        # TODO: Integrate with actual alert system (email, Slack, etc.)

    def start(self) -> None:
        """Start continuous validation."""
        self.logger.info(f"Starting continuous validation with {self.schedule_interval} schedule")

        # Run immediately on start
        self.run_validation()

        # Run on schedule
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Continuous Validation")
    parser.add_argument(
        "--interval",
        choices=["hourly", "daily", "weekly"],
        default="daily",
        help="Schedule interval"
    )

    args = parser.parse_args()

    validator = ContinuousValidator(schedule_interval=args.interval)
    validator.start()