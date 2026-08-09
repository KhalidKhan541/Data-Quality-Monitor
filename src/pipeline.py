import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from .schema_validator import SchemaValidator
from .drift_detector import DriftDetector
from .anomaly_detector import ColumnAnomalyDetector
from .alert_manager import AlertManager
from .dashboard import TrendDashboard

class DataQualityPipeline:
    """Main pipeline for data quality monitoring."""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.alert_manager = AlertManager(os.path.join(output_dir, 'alerts'))
        self.dashboard = TrendDashboard(output_dir)
        
        self.schema_validator = None
        self.drift_detector = None
        self.anomaly_detector = ColumnAnomalyDetector()
        
        self.results = {}
    
    def run_full_check(self, current_data: pd.DataFrame,
                      reference_data: pd.DataFrame = None,
                      schema_rules: list = None,
                      run_id: str = None) -> Dict[str, Any]:
        """Run complete data quality check."""
        if run_id is None:
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info("=" * 60)
        self.logger.info(f"DATA QUALITY CHECK: {run_id}")
        self.logger.info("=" * 60)
        
        # Schema validation
        schema_result = self._run_schema_validation(current_data, schema_rules)
        
        # Drift detection
        drift_result = self._run_drift_detection(current_data, reference_data)
        
        # Anomaly detection
        anomaly_result = self._run_anomaly_detection(current_data, reference_data)
        
        # Generate alerts
        self.alert_manager.clear_alerts()
        self.alert_manager.check_schema_validation(schema_result)
        self.alert_manager.check_drift(drift_result)
        self.alert_manager.check_anomalies(anomaly_result)
        
        # Generate failure summary
        failure_summary = self.alert_manager.generate_failure_summary(
            schema_result, drift_result, anomaly_result
        )
        
        # Compute scores
        schema_score = schema_result.get('success_rate', 0)
        drift_score = drift_result.get('composite_score', 0)
        anomaly_score = self._anomaly_to_score(anomaly_result)
        
        # Record run
        self.dashboard.record_run(
            run_id=run_id,
            validation=schema_result,
            drift=drift_result,
            anomalies=anomaly_result,
            schema_score=schema_score,
            drift_score=drift_score,
            anomaly_score=anomaly_score,
        )
        
        # Generate dashboard
        dashboard_path = self.dashboard.generate_dashboard()
        
        # Compile results
        self.results = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'schema': schema_result,
            'drift': drift_result,
            'anomalies': anomaly_result,
            'failure_summary': failure_summary,
            'scores': {
                'schema': schema_score,
                'drift': drift_score,
                'anomaly': anomaly_score,
                'overall': self.dashboard._compute_overall(schema_score, drift_score, anomaly_score),
            },
            'dashboard_path': dashboard_path,
            'alerts_path': os.path.join(self.output_dir, 'alerts', 'alerts.json'),
            'summary_path': os.path.join(self.output_dir, 'alerts', 'failure_summary.json'),
        }
        
        # Save alerts
        self.alert_manager.save_alerts()
        
        # Print summary
        self._print_summary(self.results)
        
        return self.results
    
    def _run_schema_validation(self, df: pd.DataFrame,
                               rules: list = None) -> Dict[str, Any]:
        """Run schema validation."""
        self.logger.info("Running schema validation...")
        
        self.schema_validator = SchemaValidator()
        
        if rules:
            for rule in rules:
                self.schema_validator.add_rule(**rule)
        else:
            self._auto_generate_rules(df)
        
        return self.schema_validator.validate(df)
    
    def _auto_generate_rules(self, df: pd.DataFrame):
        """Auto-generate schema rules from dataframe."""
        for col in df.columns:
            self.schema_validator.expect_column_to_exist(col)
            
            if df[col].isna().mean() == 0:
                self.schema_validator.expect_column_values_to_not_be_null(col)
    
    def _run_drift_detection(self, current: pd.DataFrame,
                            reference: pd.DataFrame = None) -> Dict[str, Any]:
        """Run drift detection."""
        self.logger.info("Running drift detection...")
        
        if reference is None:
            reference = current.sample(min(1000, len(current)), random_state=42)
        
        self.drift_detector = DriftDetector(reference)
        return self.drift_detector.detect_all_drift(current)
    
    def _run_anomaly_detection(self, current: pd.DataFrame,
                              reference: pd.DataFrame = None) -> Dict[str, Any]:
        """Run anomaly detection."""
        self.logger.info("Running anomaly detection...")
        
        if reference is not None:
            self.anomaly_detector.fit_baseline(reference)
        
        return self.anomaly_detector.detect_anomalies(current)
    
    def _anomaly_to_score(self, anomaly_result: Dict) -> float:
        """Convert anomaly result to score 0-100."""
        severity = anomaly_result.get('severity', 'healthy')
        severity_map = {'healthy': 95, 'low': 80, 'medium': 60, 'high': 30, 'critical': 10}
        return severity_map.get(severity, 50)
    
    def _print_summary(self, results: Dict):
        """Print quality check summary."""
        scores = results['scores']
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("DATA QUALITY SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Schema Score:  {scores['schema']:.1f}%")
        self.logger.info(f"Drift Score:   {scores['drift']:.1f}")
        self.logger.info(f"Anomaly Score: {scores['anomaly']:.1f}")
        self.logger.info(f"Overall Score: {scores['overall']:.1f}")
        self.logger.info(f"Health: {results['failure_summary']['overall_health']}")
        self.logger.info(f"Alerts: {results['failure_summary']['total_alerts']}")
        self.logger.info("=" * 60)
    
    def validate_expectations(self, df: pd.DataFrame,
                             expectations_path: str) -> Dict[str, Any]:
        """Validate against saved expectations file."""
        import json
        with open(expectations_path) as f:
            expectations = json.load(f)
        
        validator = SchemaValidator()
        for exp in expectations.get('expectations', []):
            validator.add_rule(**exp)
        
        return validator.validate(df)
    
    def generate_report(self) -> str:
        """Generate HTML quality report."""
        if not self.results:
            return ''
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Data Quality Report - {self.results['run_id']}</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; }}
        .card {{ background: #1e293b; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; border: 1px solid #334155; }}
        .score {{ font-size: 2rem; font-weight: bold; }}
        .healthy {{ color: #10b981; }}
        .warning {{ color: #f59e0b; }}
        .critical {{ color: #ef4444; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 0.5rem; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ color: #6366f1; }}
    </style>
</head>
<body>
    <h1>Data Quality Report</h1>
    <p>Run: {self.results['run_id']} | {self.results['timestamp']}</p>
    
    <div class="card">
        <h2>Quality Scores</h2>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;">
            <div><div class="score">{self.results['scores']['schema']:.1f}%</div><div>Schema</div></div>
            <div><div class="score">{self.results['scores']['drift']:.1f}</div><div>Drift</div></div>
            <div><div class="score">{self.results['scores']['anomaly']:.1f}</div><div>Anomaly</div></div>
            <div><div class="score">{self.results['scores']['overall']:.1f}</div><div>Overall</div></div>
        </div>
    </div>
    
    <div class="card">
        <h2>Health: {self.results['failure_summary']['overall_health'].upper()}</h2>
        <p>Alerts: {self.results['failure_summary']['total_alerts']} 
           (Critical: {self.results['failure_summary']['critical']}, 
            High: {self.results['failure_summary']['high']})</p>
    </div>
    
    <div class="card">
        <h2>Top Issues</h2>
        <ul>
        {''.join(f'<li>[{i["severity"]}] {i["title"]}</li>' for i in self.results['failure_summary'].get('top_issues', [])[:5])}
        </ul>
    </div>
    
    <div class="card">
        <h2>Recommendations</h2>
        <ul>
        {''.join(f'<li>{r}</li>' for r in self.results['failure_summary'].get('recommendations', []))}
        </ul>
    </div>
</body>
</html>"""
        
        report_path = os.path.join(self.output_dir, 'quality_report.html')
        with open(report_path, 'w') as f:
            f.write(html)
        
        return report_path
