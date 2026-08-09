import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

class TrendDashboard:
    """Track data quality score over time across pipeline runs."""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        self.runs_file = os.path.join(output_dir, 'pipeline_runs.json')
        self.runs = self._load_runs()
    
    def _load_runs(self) -> List[Dict]:
        """Load historical run data."""
        if os.path.exists(self.runs_file):
            with open(self.runs_file) as f:
                return json.load(f)
        return []
    
    def record_run(self, run_id: str, validation: Dict = None,
                   drift: Dict = None, anomalies: Dict = None,
                   schema_score: float = None, drift_score: float = None,
                   anomaly_score: float = None):
        """Record a pipeline run for trend tracking."""
        run = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'schema_score': schema_score,
            'drift_score': drift_score,
            'anomaly_score': anomaly_score,
            'overall_score': self._compute_overall(schema_score, drift_score, anomaly_score),
            'validation': validation,
            'drift': drift,
            'anomalies': anomalies,
        }
        
        self.runs.append(run)
        self._save_runs()
        
        return run
    
    def _compute_overall(self, schema: float = None, drift: float = None,
                        anomaly: float = None) -> float:
        """Compute overall quality score (0-100, higher is better)."""
        scores = []
        weights = []
        
        if schema is not None:
            scores.append(schema)
            weights.append(0.4)
        
        if drift is not None:
            drift_quality = max(0, 100 - drift)
            scores.append(drift_quality)
            weights.append(0.35)
        
        if anomaly is not None:
            anomaly_quality = max(0, 100 - anomaly)
            scores.append(anomaly_quality)
            weights.append(0.25)
        
        if not scores:
            return 0.0
        
        return float(np.average(scores, weights=weights[:len(scores)]))
    
    def _save_runs(self):
        """Save runs to file."""
        os.makedirs(os.path.dirname(self.runs_file), exist_ok=True)
        with open(self.runs_file, 'w') as f:
            json.dump(self.runs, f, indent=2, default=str)
    
    def generate_dashboard(self, save_path: str = None) -> str:
        """Generate trend dashboard visualization."""
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'dashboard.png')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.3)
        
        # 1. Overall Quality Score Trend
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_quality_trend(ax1)
        
        # 2. Component Scores
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_component_scores(ax2)
        
        # 3. Alert Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_alert_distribution(ax3)
        
        # 4. Run History Table
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_run_history(ax4)
        
        fig.suptitle('Data Quality Monitor - Trend Dashboard',
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
        
        return save_path
    
    def _plot_quality_trend(self, ax):
        """Plot overall quality score over runs."""
        if not self.runs:
            ax.text(0.5, 0.5, 'No run data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, color='#94a3b8')
            return
        
        run_ids = [r['run_id'][:8] for r in self.runs]
        scores = [r.get('overall_score', 0) for r in self.runs]
        
        x = range(len(scores))
        ax.plot(x, scores, 'o-', color='#6366f1', linewidth=2, markersize=8, label='Overall')
        
        # Color zones
        ax.axhspan(80, 100, alpha=0.1, color='#10b981', label='Healthy')
        ax.axhspan(50, 80, alpha=0.1, color='#f59e0b', label='Warning')
        ax.axhspan(0, 50, alpha=0.1, color='#ef4444', label='Critical')
        
        ax.set_xticks(x)
        ax.set_xticklabels(run_ids, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Quality Score', fontsize=11)
        ax.set_title('Data Quality Score Trend', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#0f172a')
    
    def _plot_component_scores(self, ax):
        """Plot component scores breakdown."""
        if not self.runs:
            ax.text(0.5, 0.5, 'No run data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='#94a3b8')
            return
        
        schema_scores = [r.get('schema_score', 0) for r in self.runs]
        drift_scores = [r.get('drift_score', 0) for r in self.runs]
        anomaly_scores = [r.get('anomaly_score', 0) for r in self.runs]
        
        x = range(len(self.runs))
        ax.plot(x, schema_scores, 's-', color='#10b981', linewidth=2, label='Schema')
        ax.plot(x, drift_scores, '^-', color='#f59e0b', linewidth=2, label='Drift')
        ax.plot(x, anomaly_scores, 'o-', color='#ef4444', linewidth=2, label='Anomaly')
        
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Component Scores', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#0f172a')
    
    def _plot_alert_distribution(self, ax):
        """Plot alert distribution pie chart."""
        alert_counts = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
        
        for run in self.runs:
            alerts = run.get('validation', {}).get('results', [])
            for a in alerts:
                if not a.get('passed'):
                    alert_counts['High'] += 1
        
        if sum(alert_counts.values()) == 0:
            alert_counts = {'No Alerts': 1}
        
        colors = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981']
        labels = list(alert_counts.keys())
        sizes = list(alert_counts.values())
        
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors[:len(labels)],
            autopct='%1.0f%%', textprops={'color': '#e2e8f0'}
        )
        ax.set_title('Alert Distribution', fontsize=13, fontweight='bold')
    
    def _plot_run_history(self, ax):
        """Plot run history table."""
        ax.axis('off')
        
        if not self.runs:
            ax.text(0.5, 0.5, 'No run history', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='#94a3b8')
            return
        
        recent_runs = self.runs[-10:]
        
        headers = ['Run ID', 'Timestamp', 'Schema', 'Drift', 'Anomaly', 'Overall']
        table_data = []
        
        for run in recent_runs:
            table_data.append([
                run['run_id'][:12],
                run['timestamp'][:19],
                f"{run.get('schema_score', 0):.1f}" if run.get('schema_score') is not None else 'N/A',
                f"{run.get('drift_score', 0):.1f}" if run.get('drift_score') is not None else 'N/A',
                f"{run.get('anomaly_score', 0):.1f}" if run.get('anomaly_score') is not None else 'N/A',
                f"{run.get('overall_score', 0):.1f}" if run.get('overall_score') is not None else 'N/A',
            ])
        
        table = ax.table(cellText=table_data, colLabels=headers,
                        loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('#334155')
            if row == 0:
                cell.set_facecolor('#1e293b')
                cell.set_text_props(color='#6366f1', fontweight='bold')
            else:
                cell.set_facecolor('#0f172a')
                cell.set_text_props(color='#e2e8f0')
        
        ax.set_title('Recent Pipeline Runs', fontsize=13, fontweight='bold', pad=20)
