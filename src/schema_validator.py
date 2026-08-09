import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

class SchemaValidator:
    """Great Expectations-style schema validation with custom rules engine."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rules = []
        self.results = []
    
    def add_rule(self, rule_type: str, column: str = None, **kwargs):
        """Add a validation rule."""
        rule = {'type': rule_type, 'column': column, 'params': kwargs}
        self.rules.append(rule)
        return self
    
    def expect_column_values_to_not_be_null(self, column: str) -> 'SchemaValidator':
        self.rules.append({'type': 'not_null', 'column': column})
        return self
    
    def expect_column_values_to_be_unique(self, column: str) -> 'SchemaValidator':
        self.rules.append({'type': 'unique', 'column': column})
        return self
    
    def expect_column_values_to_be_in_set(self, column: str, values: list) -> 'SchemaValidator':
        self.rules.append({'type': 'in_set', 'column': column, 'params': {'values': values}})
        return self
    
    def expect_column_values_to_be_between(self, column: str, min_val=None, max_val=None) -> 'SchemaValidator':
        self.rules.append({'type': 'between', 'column': column, 'params': {'min': min_val, 'max': max_val}})
        return self
    
    def expect_column_values_to_match_regex(self, column: str, pattern: str) -> 'SchemaValidator':
        self.rules.append({'type': 'regex', 'column': column, 'params': {'pattern': pattern}})
        return self
    
    def expect_column_to_exist(self, column: str) -> 'SchemaValidator':
        self.rules.append({'type': 'column_exists', 'column': column})
        return self
    
    def expect_table_row_count_to_be_between(self, min_rows: int = None, max_rows: int = None) -> 'SchemaValidator':
        self.rules.append({'type': 'row_count', 'params': {'min': min_rows, 'max': max_rows}})
        return self
    
    def expect_column_mean_to_be_between(self, column: str, min_val: float = None, max_val: float = None) -> 'SchemaValidator':
        self.rules.append({'type': 'mean_between', 'column': column, 'params': {'min': min_val, 'max': max_val}})
        return self
    
    def expect_column_std_to_be_between(self, column: str, min_val: float = None, max_val: float = None) -> 'SchemaValidator':
        self.rules.append({'type': 'std_between', 'column': column, 'params': {'min': min_val, 'max': max_val}})
        return self
    
    def expect_correlation_between_columns(self, col1: str, col2: str, min_corr: float = None, max_corr: float = None) -> 'SchemaValidator':
        self.rules.append({'type': 'correlation', 'column': f'{col1},{col2}', 'params': {'min': min_corr, 'max': max_corr}})
        return self
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute all validation rules against the dataframe."""
        self.results = []
        start_time = datetime.now()
        
        for rule in self.rules:
            result = self._execute_rule(df, rule)
            self.results.append(result)
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        n_passed = sum(1 for r in self.results if r['passed'])
        n_failed = sum(1 for r in self.results if not r['passed'])
        
        return {
            'timestamp': start_time.isoformat(),
            'duration_seconds': total_time,
            'total_rules': len(self.results),
            'passed': n_passed,
            'failed': n_failed,
            'success_rate': n_passed / len(self.results) * 100 if self.results else 0,
            'results': self.results,
            'passed_all': n_failed == 0,
        }
    
    def _execute_rule(self, df: pd.DataFrame, rule: Dict) -> Dict[str, Any]:
        """Execute a single validation rule."""
        rule_type = rule['type']
        column = rule.get('column')
        params = rule.get('params', {})
        
        try:
            if rule_type == 'not_null':
                return self._check_not_null(df, column)
            elif rule_type == 'unique':
                return self._check_unique(df, column)
            elif rule_type == 'in_set':
                return self._check_in_set(df, column, params['values'])
            elif rule_type == 'between':
                return self._check_between(df, column, params.get('min'), params.get('max'))
            elif rule_type == 'regex':
                return self._check_regex(df, column, params['pattern'])
            elif rule_type == 'column_exists':
                return self._check_column_exists(df, column)
            elif rule_type == 'row_count':
                return self._check_row_count(df, params.get('min'), params.get('max'))
            elif rule_type == 'mean_between':
                return self._check_mean_between(df, column, params.get('min'), params.get('max'))
            elif rule_type == 'std_between':
                return self._check_std_between(df, column, params.get('min'), params.get('max'))
            elif rule_type == 'correlation':
                cols = column.split(',')
                return self._check_correlation(df, cols[0], cols[1], params.get('min'), params.get('max'))
            else:
                return self._make_result(False, rule_type, column, f"Unknown rule type: {rule_type}")
        except Exception as e:
            return self._make_result(False, rule_type, column, f"Error: {str(e)}")
    
    def _check_not_null(self, df: pd.DataFrame, column: str) -> Dict:
        null_count = df[column].isna().sum()
        total = len(df)
        null_pct = null_count / total * 100
        passed = null_count == 0
        return self._make_result(passed, 'not_null', column,
                                f"Null count: {null_count} ({null_pct:.2f}%)")
    
    def _check_unique(self, df: pd.DataFrame, column: str) -> Dict:
        dup_count = df[column].duplicated().sum()
        passed = dup_count == 0
        return self._make_result(passed, 'unique', column,
                                f"Duplicates: {dup_count}")
    
    def _check_in_set(self, df: pd.DataFrame, column: str, values: list) -> Dict:
        invalid = df[~df[column].isin(values)]
        n_invalid = len(invalid)
        passed = n_invalid == 0
        return self._make_result(passed, 'in_set', column,
                                f"Invalid values: {n_invalid}")
    
    def _check_between(self, df: pd.DataFrame, column: str,
                      min_val=None, max_val=None) -> Dict:
        series = df[column].dropna()
        violations = 0
        if min_val is not None:
            violations += (series < min_val).sum()
        if max_val is not None:
            violations += (series > max_val).sum()
        passed = violations == 0
        return self._make_result(passed, 'between', column,
                                f"Out of range: {violations}")
    
    def _check_regex(self, df: pd.DataFrame, column: str, pattern: str) -> Dict:
        import re
        non_null = df[column].dropna()
        matches = non_null.astype(str).str.match(pattern).sum()
        n_non_null = len(non_null)
        n_mismatch = n_non_null - matches
        passed = n_mismatch == 0
        return self._make_result(passed, 'regex', column,
                                f"Regex mismatches: {n_mismatch}")
    
    def _check_column_exists(self, df: pd.DataFrame, column: str) -> Dict:
        passed = column in df.columns
        return self._make_result(passed, 'column_exists', column,
                                f"Column {'exists' if passed else 'missing'}")
    
    def _check_row_count(self, df: pd.DataFrame, min_val=None, max_val=None) -> Dict:
        count = len(df)
        violations = False
        msg = f"Row count: {count}"
        if min_val is not None and count < min_val:
            violations = True
            msg += f" (min: {min_val})"
        if max_val is not None and count > max_val:
            violations = True
            msg += f" (max: {max_val})"
        return self._make_result(not violations, 'row_count', None, msg)
    
    def _check_mean_between(self, df: pd.DataFrame, column: str,
                           min_val=None, max_val=None) -> Dict:
        mean = df[column].mean()
        passed = True
        if min_val is not None and mean < min_val:
            passed = False
        if max_val is not None and mean > max_val:
            passed = False
        return self._make_result(passed, 'mean_between', column,
                                f"Mean: {mean:.4f}")
    
    def _check_std_between(self, df: pd.DataFrame, column: str,
                          min_val=None, max_val=None) -> Dict:
        std = df[column].std()
        passed = True
        if min_val is not None and std < min_val:
            passed = False
        if max_val is not None and std > max_val:
            passed = False
        return self._make_result(passed, 'std_between', column,
                                f"Std: {std:.4f}")
    
    def _check_correlation(self, df: pd.DataFrame, col1: str, col2: str,
                          min_corr=None, max_corr=None) -> Dict:
        corr = df[col1].corr(df[col2])
        passed = True
        if min_corr is not None and corr < min_corr:
            passed = False
        if max_corr is not None and corr > max_corr:
            passed = False
        return self._make_result(passed, 'correlation', f'{col1},{col2}',
                                f"Correlation: {corr:.4f}")
    
    def _make_result(self, passed: bool, rule_type: str,
                    column: str, message: str) -> Dict:
        return {
            'passed': passed,
            'rule': rule_type,
            'column': column,
            'message': message,
        }
    
    def save_rules(self, path: str):
        """Save rules to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.rules, f, indent=2, default=str)
    
    def load_rules(self, path: str):
        """Load rules from JSON file."""
        with open(path) as f:
            self.rules = json.load(f)
    
    def generate_schema_expectations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Auto-generate expectations from a reference dataframe."""
        expectations = []
        
        for col in df.columns:
            expectations.append({'type': 'column_exists', 'column': col})
            
            null_pct = df[col].isna().mean()
            if null_pct == 0:
                expectations.append({'type': 'not_null', 'column': col})
            
            if df[col].nunique() == len(df[col].dropna()):
                expectations.append({'type': 'unique', 'column': col})
            
            if pd.api.types.is_numeric_dtype(df[col]):
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                expectations.append({
                    'type': 'between', 'column': col,
                    'params': {'min': float(q1), 'max': float(q99)}
                })
        
        return {
            'dataset_columns': len(df.columns),
            'dataset_rows': len(df),
            'expectations': expectations,
        }
