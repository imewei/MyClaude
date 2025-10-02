#!/usr/bin/env python3
"""
Validation Dashboard - Web-based dashboard for validation monitoring.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from flask import Flask, render_template_string, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


class ValidationDashboard:
    """Web dashboard for validation monitoring."""

    def __init__(self, validation_dir: Optional[Path] = None):
        """Initialize dashboard."""
        self.validation_dir = validation_dir or Path(__file__).parent.parent

        if not FLASK_AVAILABLE:
            raise RuntimeError("Flask is required for dashboard. Install with: pip install flask")

        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Setup Flask routes."""

        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return render_template_string(DASHBOARD_HTML)

        @self.app.route('/api/summary')
        def api_summary():
            """Get validation summary."""
            return jsonify(self._get_summary())

        @self.app.route('/api/results')
        def api_results():
            """Get validation results."""
            return jsonify(self._get_results())

    def _get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        # Load latest results
        reports_dir = self.validation_dir / "reports"
        if not reports_dir.exists():
            return {'total': 0, 'successful': 0, 'failed': 0}

        # Find latest report
        json_files = list(reports_dir.glob("**/validation_report.json"))
        if not json_files:
            return {'total': 0, 'successful': 0, 'failed': 0}

        latest = max(json_files, key=lambda p: p.stat().st_mtime)

        with open(latest) as f:
            data = json.load(f)

        return data.get('summary', {})

    def _get_results(self) -> List[Dict[str, Any]]:
        """Get validation results."""
        reports_dir = self.validation_dir / "reports"
        if not reports_dir.exists():
            return []

        json_files = list(reports_dir.glob("**/validation_report.json"))
        if not json_files:
            return []

        latest = max(json_files, key=lambda p: p.stat().st_mtime)

        with open(latest) as f:
            data = json.load(f)

        return data.get('results', [])

    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False) -> None:
        """Run dashboard server."""
        print(f"Starting validation dashboard on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


# HTML template for dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Validation Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #333; }
        .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .card-value { font-size: 48px; font-weight: bold; color: #4CAF50; }
        .card-label { color: #666; margin-top: 10px; }
        table { width: 100%; background: white; border-collapse: collapse; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        th, td { padding: 15px; text-align: left; }
        th { background: #4CAF50; color: white; }
        tr:nth-child(even) { background: #f9f9f9; }
        .success { color: #4CAF50; font-weight: bold; }
        .failure { color: #f44336; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Validation Dashboard</h1>
        <p id="last-updated">Loading...</p>

        <div class="cards" id="summary-cards">
            <div class="card">
                <div class="card-value" id="total-validations">-</div>
                <div class="card-label">Total Validations</div>
            </div>
            <div class="card">
                <div class="card-value" style="color: #4CAF50;" id="successful">-</div>
                <div class="card-label">Successful</div>
            </div>
            <div class="card">
                <div class="card-value" style="color: #f44336;" id="failed">-</div>
                <div class="card-label">Failed</div>
            </div>
            <div class="card">
                <div class="card-value" id="success-rate">-</div>
                <div class="card-label">Success Rate</div>
            </div>
        </div>

        <h2>Recent Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Project</th>
                    <th>Scenario</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Timestamp</th>
                </tr>
            </thead>
            <tbody id="results-table">
                <tr><td colspan="5">Loading...</td></tr>
            </tbody>
        </table>
    </div>

    <script>
        async function loadData() {
            try {
                // Load summary
                const summaryResp = await fetch('/api/summary');
                const summary = await summaryResp.json();

                document.getElementById('total-validations').textContent = summary.total || 0;
                document.getElementById('successful').textContent = summary.successful || 0;
                document.getElementById('failed').textContent = summary.failed || 0;
                document.getElementById('success-rate').textContent =
                    (summary.success_rate || 0).toFixed(1) + '%';

                // Load results
                const resultsResp = await fetch('/api/results');
                const results = await resultsResp.json();

                const tbody = document.getElementById('results-table');
                tbody.innerHTML = '';

                results.forEach(result => {
                    const row = tbody.insertRow();
                    row.insertCell(0).textContent = result.project_name;
                    row.insertCell(1).textContent = result.scenario_name;

                    const statusCell = row.insertCell(2);
                    statusCell.textContent = result.success ? '✓ Pass' : '✗ Fail';
                    statusCell.className = result.success ? 'success' : 'failure';

                    row.insertCell(3).textContent = result.duration_seconds.toFixed(1) + 's';
                    row.insertCell(4).textContent = new Date(result.timestamp).toLocaleString();
                });

                document.getElementById('last-updated').textContent =
                    'Last updated: ' + new Date().toLocaleString();

            } catch (error) {
                console.error('Error loading data:', error);
            }
        }

        // Load data on page load
        loadData();

        // Refresh every 30 seconds
        setInterval(loadData, 30000);
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    dashboard = ValidationDashboard()
    dashboard.run(debug=True)