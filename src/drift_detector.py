import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Tuple, List, Optional

class DriftDetector:
    """Detect distribution drift using PSI, KL divergence, and KS test."""
    
    def __init__(self, reference_data: pd.DataFrame):
        self.reference = reference_data
        self.numeric_cols = reference_data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = reference_data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def population_stability_index(self, current_data: pd.DataFrame,
                                   n_bins: int = 10) -> Dict[str, Any]:
        """Compute PSI for all columns.
        
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.25: Moderate change
        PSI >= 0.25: Significant change
        """
        results = {}
        
        for col in self.numeric_cols:
            if col in current_data.columns:
                psi = self._compute_psi_numeric(
                    self.reference[col].dropna(),
                    current_data[col].dropna(),
                    n_bins
                )
                results[col] = {
                    'psi': psi,
                    'severity': self._psi_severity(psi),
                    'type': 'numeric',
                }
        
        for col in self.categorical_cols:
            if col in current_data.columns:
                psi = self._compute_psi_categorical(
                    self.reference[col].dropna(),
                    current_data[col].dropna()
                )
                results[col] = {
                    'psi': psi,
                    'severity': self._psi_severity(psi),
                    'type': 'categorical',
                }
        
        overall_psi = np.mean([r['psi'] for r in results.values()]) if results else 0
        
        return {
            'overall_psi': float(overall_psi),
            'severity': self._psi_severity(overall_psi),
            'columns': results,
            'n_stable': sum(1 for r in results.values() if r['severity'] == 'stable'),
            'n_moderate': sum(1 for r in results.values() if r['severity'] == 'moderate'),
            'n_significant': sum(1 for r in results.values() if r['severity'] == 'significant'),
        }
    
    def _compute_psi_numeric(self, reference: pd.Series, current: pd.Series,
                            n_bins: int = 10) -> float:
        """Compute PSI for numeric column using binning."""
        combined = pd.concat([reference, current])
        bins = np.percentile(combined.dropna(), np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        
        ref_hist, _ = np.histogram(reference, bins=bins, density=True)
        cur_hist, _ = np.histogram(current, bins=bins, density=True)
        
        ref_hist = ref_hist / ref_hist.sum() if ref_hist.sum() > 0 else ref_hist
        cur_hist = cur_hist / cur_hist.sum() if cur_hist.sum() > 0 else cur_hist
        
        eps = 1e-10
        ref_hist = np.clip(ref_hist, eps, None)
        cur_hist = np.clip(cur_hist, eps, None)
        
        psi = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))
        return float(psi)
    
    def _compute_psi_categorical(self, reference: pd.Series, current: pd.Series) -> float:
        """Compute PSI for categorical column."""
        ref_counts = reference.value_counts(normalize=True)
        cur_counts = current.value_counts(normalize=True)
        
        all_categories = set(ref_counts.index) | set(cur_counts.index)
        
        eps = 1e-10
        psi = 0.0
        
        for cat in all_categories:
            ref_val = ref_counts.get(cat, eps)
            cur_val = cur_counts.get(cat, eps)
            psi += (cur_val - ref_val) * np.log(cur_val / ref_val)
        
        return float(psi)
    
    def _psi_severity(self, psi: float) -> str:
        if psi < 0.1:
            return 'stable'
        elif psi < 0.25:
            return 'moderate'
        else:
            return 'significant'
    
    def kl_divergence(self, current_data: pd.DataFrame,
                     bins: int = 50) -> Dict[str, Any]:
        """Compute KL divergence for numeric columns."""
        results = {}
        
        for col in self.numeric_cols:
            if col in current_data.columns:
                kl = self._compute_kl_divergence(
                    self.reference[col].dropna(),
                    current_data[col].dropna(),
                    bins
                )
                results[col] = {
                    'kl_divergence': kl,
                    'interpretation': self._kl_interpretation(kl),
                }
        
        return {
            'columns': results,
            'mean_kl': float(np.mean([r['kl_divergence'] for r in results.values()])) if results else 0,
        }
    
    def _compute_kl_divergence(self, reference: pd.Series, current: pd.Series,
                              bins: int = 50) -> float:
        """Compute KL(P||Q) where P=current, Q=reference."""
        combined = pd.concat([reference, current])
        hist_range = (combined.min(), combined.max())
        
        ref_hist, _ = np.histogram(reference, bins=bins, range=hist_range, density=True)
        cur_hist, _ = np.histogram(current, bins=bins, range=hist_range, density=True)
        
        eps = 1e-10
        ref_hist = np.clip(ref_hist, eps, None)
        cur_hist = np.clip(cur_hist, eps, None)
        
        ref_hist = ref_hist / ref_hist.sum()
        cur_hist = cur_hist / cur_hist.sum()
        
        kl = np.sum(cur_hist * np.log(cur_hist / ref_hist))
        return float(max(kl, 0))
    
    def _kl_interpretation(self, kl: float) -> str:
        if kl < 0.1:
            return 'minimal divergence'
        elif kl < 0.5:
            return 'moderate divergence'
        elif kl < 1.0:
            return 'notable divergence'
        else:
            return 'high divergence'
    
    def ks_test(self, current_data: pd.DataFrame,
               alpha: float = 0.05) -> Dict[str, Any]:
        """Two-sample KS test for distribution shift."""
        results = {}
        
        for col in self.numeric_cols:
            if col in current_data.columns:
                ref = self.reference[col].dropna()
                cur = current_data[col].dropna()
                
                if len(ref) > 0 and len(cur) > 0:
                    ks_stat, p_value = stats.ks_2samp(ref, cur)
                    
                    results[col] = {
                        'ks_statistic': float(ks_stat),
                        'p_value': float(p_value),
                        'significant': p_value < alpha,
                        'interpretation': 'distribution shifted' if p_value < alpha else 'no significant shift',
                    }
        
        n_significant = sum(1 for r in results.values() if r['significant'])
        
        return {
            'alpha': alpha,
            'columns': results,
            'n_significant': n_significant,
            'n_tested': len(results),
            'shift_detected': n_significant > 0,
        }
    
    def mannwhitney_test(self, current_data: pd.DataFrame,
                        alpha: float = 0.05) -> Dict[str, Any]:
        """Mann-Whitney U test for distribution shift (non-parametric)."""
        results = {}
        
        for col in self.numeric_cols:
            if col in current_data.columns:
                ref = self.reference[col].dropna()
                cur = current_data[col].dropna()
                
                if len(ref) > 0 and len(cur) > 0:
                    u_stat, p_value = stats.mannwhitneyu(ref, cur, alternative='two-sided')
                    
                    results[col] = {
                        'u_statistic': float(u_stat),
                        'p_value': float(p_value),
                        'significant': p_value < alpha,
                    }
        
        return {
            'alpha': alpha,
            'columns': results,
            'n_significant': sum(1 for r in results.values() if r['significant']),
        }
    
    def wasserstein_distance(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Compute Wasserstein (Earth Mover's) distance."""
        results = {}
        
        for col in self.numeric_cols:
            if col in current_data.columns:
                ref = self.reference[col].dropna()
                cur = current_data[col].dropna()
                
                if len(ref) > 0 and len(cur) > 0:
                    dist = stats.wasserstein_distance(ref, cur)
                    
                    results[col] = {
                        'distance': float(dist),
                        'normalized': float(dist / ref.std()) if ref.std() > 0 else 0,
                    }
        
        return {'columns': results}
    
    def detect_all_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Run all drift detection methods."""
        psi_results = self.population_stability_index(current_data)
        kl_results = self.kl_divergence(current_data)
        ks_results = self.ks_test(current_data)
        ws_results = self.wasserstein_distance(current_data)
        
        # Composite drift score (0-100)
        drift_score = self._compute_composite_score(psi_results, ks_results)
        
        return {
            'psi': psi_results,
            'kl_divergence': kl_results,
            'ks_test': ks_results,
            'wasserstein': ws_results,
            'composite_score': drift_score,
            'overall_drift': self._overall_drift_assessment(drift_score),
        }
    
    def _compute_composite_score(self, psi_results: Dict, ks_results: Dict) -> float:
        """Compute composite drift score 0-100."""
        psi_score = min(psi_results['overall_psi'] / 0.25 * 50, 50)
        
        ks_cols = ks_results.get('columns', {})
        ks_ratio = ks_results.get('n_significant', 0) / max(ks_results.get('n_tested', 1), 1)
        ks_score = ks_ratio * 50
        
        return float(psi_score + ks_score)
    
    def _overall_drift_assessment(self, score: float) -> str:
        if score < 10:
            return 'healthy'
        elif score < 30:
            return 'warning'
        else:
            return 'critical'
    
    def column_drift_summary(self, current_data: pd.DataFrame) -> pd.DataFrame:
        """Summary DataFrame of drift per column."""
        psi_results = self.population_stability_index(current_data)
        ks_results = self.ks_test(current_data)
        
        rows = []
        all_cols = set(self.numeric_cols + self.categorical_cols)
        
        for col in all_cols:
            if col in current_data.columns:
                psi_val = psi_results['columns'].get(col, {}).get('psi', 0)
                ks_pval = ks_results['columns'].get(col, {}).get('p_value', 1)
                
                rows.append({
                    'column': col,
                    'psi': psi_val,
                    'psi_severity': psi_results['columns'].get(col, {}).get('severity', 'unknown'),
                    'ks_p_value': ks_pval,
                    'ks_significant': ks_pval < 0.05,
                })
        
        return pd.DataFrame(rows).sort_values('psi', ascending=False)
