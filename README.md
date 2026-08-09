# Data Quality Monitoring System

Production-grade data quality monitoring with Great Expectations-style validation, PSI/KL divergence/KS test for drift detection, column-level anomaly detection, and trend dashboard.

## Features

| Feature | Description |
|---------|-------------|
| **Schema Validation** | Great Expectations-style rules engine (not_null, unique, between, regex, correlation) |
| **PSI** | Population Stability Index for distribution drift detection |
| **KL Divergence** | Kullback-Leibler divergence for statistical shift measurement |
| **KS Test** | Kolmogorov-Smirnov test for distribution comparison |
| **Anomaly Detection** | Column-level anomaly detection on ingestion (mean shift, variance change, new categories) |
| **Alerting** | Automated alerts with failure summary and severity levels |
| **Trend Dashboard** | Track data quality score over time across pipeline runs |

## Quick Start

```bash
pip install -r requirements.txt

# Basic quality check
python run.py data.csv

# With reference data for drift detection
python run.py data.csv --reference reference.csv

# With custom schema rules
python run.py data.csv --schema-rules rules.json --output reports
```

## Usage Examples

### Python API

```python
from src.pipeline import DataQualityPipeline

pipeline = DataQualityPipeline(output_dir='output')
results = pipeline.run_full_check(
    current_data=df,
    reference_data=ref_df,
)
print(f"Health: {results['failure_summary']['overall_health']}")
```

### Schema Rules

```python
from src.schema_validator import SchemaValidator

validator = SchemaValidator()
validator.expect_column_values_to_not_be_null('user_id')
validator.expect_column_values_to_be_unique('user_id')
validator.expect_column_values_to_be_between('age', 0, 150)
validator.expect_column_values_to_match_regex('email', r'^[\w\.-]+@[\w\.-]+\.\w+$')

results = validator.validate(df)
```

### Drift Detection

```python
from src.drift_detector import DriftDetector

detector = DriftDetector(reference_df)
drift = detector.detect_all_drift(current_df)

print(f"PSI: {drift['psi']['overall_psi']:.4f}")
print(f"Drift: {drift['overall_drift']}")
```

## Architecture

```
Data-Quality-Monitor/
├── src/
│   ├── schema_validator.py    # Great Expectations-style rules engine
│   ├── drift_detector.py      # PSI, KL divergence, KS test
│   ├── anomaly_detector.py    # Column-level anomaly detection
│   ├── alert_manager.py       # Alert generation and failure summary
│   ├── dashboard.py           # Trend dashboard tracking
│   └── pipeline.py            # Main orchestrator
├── run.py                     # CLI entry point
├── requirements.txt
└── README.md
```

## Dependencies

- `great_expectations` - Schema validation framework
- `scipy` - KS test, KL divergence
- `pandas`, `numpy` - Data manipulation
- `matplotlib` - Dashboard visualization
