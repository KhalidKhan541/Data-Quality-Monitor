import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Optional

class ColumnAnomalyDetector:
    """Column-level anomaly detection on data ingestion."""
    
    def __init__(self, history: pd.DataFrame = None, sensitivity: float = 3.0):
        self.history = history
        self.sensitivity = sensitivity
        self.baselines = {}
    
    def fit_baseline(self, df: pd.DataFrame):
        """Compute baseline statistics from reference data."""
        self.history = df
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            series = df[col].dropna()
            self.baselines[col] = {
                'mean': float(series.mean()),
                'std': float(series.std()),
                'median': float(series.median()),
                'mad': float(np.median(np.abs(series - series.median()))),
                'q25': float(np.percentile(series, 25)),
                'q75': float(np.percentile(series, 75)),
                'iqr': float(np.percentile(series, 75) - np.percentile(series, 25)),
                'min': float(series.min()),
                'max': float(series.max()),
                'skew': float(series.skew()),
                'kurtosis': float(series.kurtosis()),
                'n_samples': len(series),
                'null_rate': float(df[col].isna().mean()),
            }
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            value_counts = df[col].value_counts(normalize=True)
            self.baselines[col] = {
                'n_categories': len(value_counts),
                'top_categories': value_counts.head(10).to_dict(),
                'entropy': float(stats.entropy(value_counts.values)),
                'null_rate': float(df[col].isna().mean()),
                'mode': str(value_counts.index[0]) if len(value_counts) > 0 else None,
            }
        
        return self.baselines
    
    def detect_anomalies(self, batch: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in a new data batch."""
        if not self.baselines:
            self.fit_baseline(batch)
        
        anomalies = {}
        
        for col in batch.columns:
            if col in self.baselines:
                col_anomalies = self._detect_column_anomalies(batch[col], col)
                if col_anomalies:
                    anomalies[col] = col_anomalies
        
        total_anomalies = sum(len(v) for v in anomalies.values())
        
        return {
            'n_columns_analyzed': len(batch.columns),
            'n_columns_with_anomalies': len(anomalies),
            'total_anomalies': total_anomalies,
            'anomalies': anomalies,
            'severity': self._compute_severity(total_anomalies, len(batch)),
        }
    
    def _detect_column_anomalies(self, series: pd.Series, col_name: str) -> List[Dict]:
        """Detect anomalies in a single column."""
        baseline = self.baselines.get(col_name)
        if not baseline:
            return []
        
        anomalies = []
        
        if pd.api.types.is_numeric_dtype(series):
            anomalies.extend(self._check_numeric_anomalies(series, col_name, baseline))
        else:
            anomalies.extend(self._check_categorical_anomalies(series, col_name, baseline))
        
        # Null rate anomaly
        null_rate = series.isna().mean()
        baseline_null = baseline.get('null_rate', 0)
        if abs(null_rate - baseline_null) > 0.1:
            anomalies.append({
                'type': 'null_rate_drift',
                'column': col_name,
                'severity': 'high' if abs(null_rate - baseline_null) > 0.2 else 'medium',
                'message': f'Null rate changed from {baseline_null:.2%} to {null_rate:.2%}',
                'details': {'current': null_rate, 'baseline': baseline_null},
            })
        
        return anomalies
    
    def _check_numeric_anomalies(self, series: pd.Series, col_name: str,
                                 baseline: Dict) -> List[Dict]:
        """Check for numeric anomalies."""
        anomalies = []
        data = series.dropna()
        
        if len(data) == 0:
            return anomalies
        
        mean = float(data.mean())
        std = float(data.std()) if len(data) > 1 else 0
        baseline_mean = baseline['mean']
        baseline_std = baseline['std']
        
        # Mean shift
        if baseline_std > 0:
            z_score = abs(mean - baseline_mean) / baseline_std
            if z_score > self.sensitivity:
                anomalies.append({
                    'type': 'mean_shift',
                    'column': col_name,
                    'severity': 'high' if z_score > self.sensitivity * 2 else 'medium',
                    'message': f'Mean shifted from {baseline_mean:.4f} to {mean:.4f} (z={z_score:.2f})',
                    'details': {'z_score': z_score, 'current_mean': mean, 'baseline_mean': baseline_mean},
                })
        
        # Variance change
        if baseline_std > 0 and std > 0:
            var_ratio = std / baseline_std
            if var_ratio > 2 or var_ratio < 0.5:
                anomalies.append({
                    'type': 'variance_change',
                    'column': col_name,
                    'severity': 'medium',
                    'message': f'Std changed from {baseline_std:.4f} to {std:.4f} (ratio={var_ratio:.2f})',
                    'details': {'variance_ratio': var_ratio},
                })
        
        # New min/max (outside historical range)
        new_min = float(data.min())
        new_max = float(data.max())
        if new_min < baseline['min'] or new_max > baseline['max']:
            anomalies.append({
                'type': 'range_violation',
                'column': col_name,
                'severity': 'low',
                'message': f'New range [{new_min:.4f}, {new_max:.4f}] exceeds historical [{baseline["min"]:.4f}, {baseline["max"]:.4f}]',
                'details': {'new_min': new_min, 'new_max': new_max},
            })
        
        # Skewness change
        if len(data) > 10:
            skew = float(data.skew())
            if abs(skew - baseline['skew']) > 1:
                anomalies.append({
                    'type': 'distribution_change',
                    'column': col_name,
                    'severity': 'medium',
                    'message': f'Skewness changed from {baseline["skew"]:.4f} to {skew:.4f}',
                    'details': {'current_skew': skew, 'baseline_skew': baseline['skew']},
                })
        
        # KS test for distribution shift
        if len(data) > 20 and baseline['n_samples'] > 20:
            ks_stat, p_value = stats.ks_2samp(
                self.history[col_name].dropna().values,
                data.values
            )
            if p_value < 0.01:
                anomalies.append({
                    'type': 'distribution_shift',
                    'column': col_name,
                    'severity': 'high',
                    'message': f'KS test p-value={p_value:.6f}, distribution shifted',
                    'details': {'ks_statistic': float(ks_stat), 'p_value': float(p_value)},
                })
        
        return anomalies
    
    def _check_categorical_anomalies(self, series: pd.Series, col_name: str,
                                     baseline: Dict) -> List[Dict]:
        """Check for categorical anomalies."""
        anomalies = []
        data = series.dropna()
        
        if len(data) == 0:
            return anomalies
        
        # New categories
        current_categories = set(data.unique())
        known_categories = set(baseline.get('top_categories', {}).keys())
        new_categories = current_categories - known_categories
        
        if new_categories and len(new_categories) > 0:
            anomalies.append({
                'type': 'new_categories',
                'column': col_name,
                'severity': 'medium',
                'message': f'{len(new_categories)} new categories detected: {list(new_categories)[:5]}',
                'details': {'new_categories': list(new_categories)},
            })
        
        # Category distribution change
        current_dist = data.value_counts(normalize=True)
        baseline_dist = pd.Series(baseline.get('top_categories', {}))
        
        if len(baseline_dist) > 0 and len(current_dist) > 0:
            all_cats = set(current_dist.index) | set(baseline_dist.index)
            chi2 = sum(
                ((current_dist.get(c, 0) - baseline_dist.get(c, 0)) ** 2) /
                max(baseline_dist.get(c, 0.001), 0.001)
                for c in all_cats
            )
            
            if chi2 > 10:
                anomalies.append({
                    'type': 'category_distribution_shift',
                    'column': col_name,
                    'severity': 'medium',
                    'message': f'Category distribution shifted (chi2={chi2:.2f})',
                    'details': {'chi2': float(chi2)},
                })
        
        return anomalies
    
    def _compute_severity(self, n_anomalies: int, n_rows: int) -> str:
        """Compute overall severity."""
        anomaly_rate = n_anomalies / max(n_rows, 1)
        if anomaly_rate > 0.1:
            return 'critical'
        elif anomaly_rate > 0.05:
            return 'high'
        elif anomaly_rate > 0.01:
            return 'medium'
        elif n_anomalies > 0:
            return 'low'
        else:
            return 'healthy'
