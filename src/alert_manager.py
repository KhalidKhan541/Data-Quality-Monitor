import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class Alert:
    """Data quality alert."""
    alert_id: str
    timestamp: str
    severity: str
    category: str
    title: str
    message: str
    details: Dict[str, Any]
    pipeline_run_id: str = None
    acknowledged: bool = False

class AlertManager:
    """Manage alerts for data quality failures."""
    
    SEVERITY_LEVELS = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1, 'info': 0}
    
    def __init__(self, output_dir: str = 'output/alerts'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.alerts: List[Alert] = []
        self.alert_history: List[Dict] = []
    
    def check_schema_validation(self, validation_result: Dict) -> List[Alert]:
        """Generate alerts from schema validation results."""
        alerts = []
        
        if not validation_result.get('passed_all', True):
            failed_rules = [r for r in validation_result.get('results', []) if not r.get('passed')]
            
            for rule in failed_rules:
                severity = 'critical' if validation_result['success_rate'] < 80 else 'high'
                
                alert = Alert(
                    alert_id=self._generate_id(),
                    timestamp=datetime.now().isoformat(),
                    severity=severity,
                    category='schema_validation',
                    title=f"Schema validation failed: {rule.get('rule', 'unknown')}",
                    message=rule.get('message', 'No details'),
                    details={'rule': rule},
                )
                alerts.append(alert)
            
            if validation_result['success_rate'] < 50:
                critical_alert = Alert(
                    alert_id=self._generate_id(),
                    timestamp=datetime.now().isoformat(),
                    severity='critical',
                    category='schema_validation',
                    title='Major schema validation failure',
                    message=f"Only {validation_result['success_rate']:.1f}% of rules passed",
                    details=validation_result,
                )
                alerts.append(critical_alert)
        
        self.alerts.extend(alerts)
        return alerts
    
    def check_drift(self, drift_results: Dict) -> List[Alert]:
        """Generate alerts from drift detection results."""
        alerts = []
        
        psi = drift_results.get('psi', {})
        if psi.get('severity') == 'significant':
            alert = Alert(
                alert_id=self._generate_id(),
                timestamp=datetime.now().isoformat(),
                severity='critical',
                category='distribution_drift',
                title='Significant distribution drift detected',
                message=f"Overall PSI: {psi['overall_psi']:.4f} (threshold: 0.25)",
                details=psi,
            )
            alerts.append(alert)
        elif psi.get('severity') == 'moderate':
            alert = Alert(
                alert_id=self._generate_id(),
                timestamp=datetime.now().isoformat(),
                severity='medium',
                category='distribution_drift',
                title='Moderate distribution drift detected',
                message=f"Overall PSI: {psi['overall_psi']:.4f}",
                details=psi,
            )
            alerts.append(alert)
        
        # Column-level drift alerts
        for col, col_psi in psi.get('columns', {}).items():
            if col_psi.get('severity') == 'significant':
                alert = Alert(
                    alert_id=self._generate_id(),
                    timestamp=datetime.now().isoformat(),
                    severity='high',
                    category='distribution_drift',
                    title=f'Column drift: {col}',
                    message=f"PSI: {col_psi['psi']:.4f}",
                    details=col_psi,
                )
                alerts.append(alert)
        
        # KS test alerts
        ks = drift_results.get('ks_test', {})
        if ks.get('shift_detected'):
            shifted_cols = [col for col, r in ks.get('columns', {}).items() if r.get('significant')]
            alert = Alert(
                alert_id=self._generate_id(),
                timestamp=datetime.now().isoformat(),
                severity='high',
                category='statistical_shift',
                title='Statistical distribution shift detected',
                message=f"KS test significant for: {', '.join(shifted_cols[:5])}",
                details=ks,
            )
            alerts.append(alert)
        
        self.alerts.extend(alerts)
        return alerts
    
    def check_anomalies(self, anomaly_results: Dict) -> List[Alert]:
        """Generate alerts from anomaly detection results."""
        alerts = []
        
        severity = anomaly_results.get('severity', 'healthy')
        if severity in ('critical', 'high'):
            alert = Alert(
                alert_id=self._generate_id(),
                timestamp=datetime.now().isoformat(),
                severity=severity,
                category='anomaly_detection',
                title=f'Data anomalies detected: {severity}',
                message=f"{anomaly_results['total_anomalies']} anomalies in {anomaly_results['n_columns_with_anomalies']} columns",
                details=anomaly_results,
            )
            alerts.append(alert)
        
        for col, col_anomalies in anomaly_results.get('anomalies', {}).items():
            high_severity = [a for a in col_anomalies if a.get('severity') == 'high']
            if high_severity:
                alert = Alert(
                    alert_id=self._generate_id(),
                    timestamp=datetime.now().isoformat(),
                    severity='high',
                    category='anomaly_detection',
                    title=f'High-severity anomaly in column: {col}',
                    message=high_severity[0].get('message', ''),
                    details={'column': col, 'anomalies': high_severity},
                )
                alerts.append(alert)
        
        self.alerts.extend(alerts)
        return alerts
    
    def generate_failure_summary(self, validation: Dict = None,
                                 drift: Dict = None,
                                 anomalies: Dict = None) -> Dict[str, Any]:
        """Generate comprehensive failure summary."""
        all_alerts = self.alerts.copy()
        
        critical = [a for a in all_alerts if a.severity == 'critical']
        high = [a for a in all_alerts if a.severity == 'high']
        medium = [a for a in all_alerts if a.severity == 'medium']
        low = [a for a in all_alerts if a.severity == 'low']
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_alerts': len(all_alerts),
            'critical': len(critical),
            'high': len(high),
            'medium': len(medium),
            'low': len(low),
            'overall_health': self._compute_health(all_alerts),
            'alerts_by_category': {},
            'top_issues': [],
            'recommendations': [],
        }
        
        # Group by category
        for alert in all_alerts:
            cat = alert.category
            if cat not in summary['alerts_by_category']:
                summary['alerts_by_category'][cat] = 0
            summary['alerts_by_category'][cat] += 1
        
        # Top issues
        summary['top_issues'] = [
            {'title': a.title, 'severity': a.severity}
            for a in sorted(all_alerts, key=lambda x: self.SEVERITY_LEVELS.get(x.severity, 0), reverse=True)[:10]
        ]
        
        # Recommendations
        if validation and not validation.get('passed_all'):
            summary['recommendations'].append('Fix schema validation failures before proceeding')
        if drift and drift.get('composite_score', 0) > 30:
            summary['recommendations'].append('Investigate distribution drift and retrain models')
        if anomalies and anomalies.get('severity') in ('critical', 'high'):
            summary['recommendations'].append('Review data anomalies in affected columns')
        
        # Save summary
        self._save_summary(summary)
        
        return summary
    
    def _compute_health(self, alerts: List[Alert]) -> str:
        """Compute overall health status."""
        if not alerts:
            return 'healthy'
        
        max_severity = max(self.SEVERITY_LEVELS.get(a.severity, 0) for a in alerts)
        
        if max_severity >= 4:
            return 'critical'
        elif max_severity >= 3:
            return 'degraded'
        elif max_severity >= 2:
            return 'warning'
        else:
            return 'healthy'
    
    def _generate_id(self) -> str:
        """Generate unique alert ID."""
        return f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.alerts)}"
    
    def _save_summary(self, summary: Dict):
        """Save summary to file."""
        path = os.path.join(self.output_dir, 'failure_summary.json')
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def save_alerts(self):
        """Save all alerts to file."""
        path = os.path.join(self.output_dir, 'alerts.json')
        alerts_data = [asdict(a) for a in self.alerts]
        with open(path, 'w') as f:
            json.dump(alerts_data, f, indent=2, default=str)
    
    def get_alert_counts(self) -> Dict[str, int]:
        """Get alert counts by severity."""
        counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for alert in self.alerts:
            if alert.severity in counts:
                counts[alert.severity] += 1
        return counts
    
    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts.clear()
