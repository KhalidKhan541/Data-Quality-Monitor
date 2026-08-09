import argparse
import pandas as pd
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description='Data Quality Monitoring System')
    parser.add_argument('input', help='CSV file to validate')
    parser.add_argument('--reference', '-r', help='Reference CSV for drift detection')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--schema-rules', '-s', help='Schema rules JSON file')
    parser.add_argument('--run-id', help='Pipeline run ID')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(args.input):
        logger.error(f"File not found: {args.input}")
        sys.exit(1)
    
    logger.info(f"Loading {args.input}...")
    current_data = pd.read_csv(args.input)
    logger.info(f"Shape: {current_data.shape}")
    
    reference_data = None
    if args.reference and os.path.exists(args.reference):
        reference_data = pd.read_csv(args.reference)
        logger.info(f"Reference data: {reference_data.shape}")
    
    schema_rules = None
    if args.schema_rules and os.path.exists(args.schema_rules):
        import json
        with open(args.schema_rules) as f:
            schema_rules = json.load(f)
    
    from src.pipeline import DataQualityPipeline
    
    pipeline = DataQualityPipeline(output_dir=args.output)
    results = pipeline.run_full_check(
        current_data=current_data,
        reference_data=reference_data,
        schema_rules=schema_rules,
        run_id=args.run_id,
    )
    
    report_path = pipeline.generate_report()
    
    print(f"\nQuality check complete!")
    print(f"Overall Score: {results['scores']['overall']:.1f}")
    print(f"Health: {results['failure_summary']['overall_health']}")
    print(f"Report: {report_path}")
    print(f"Dashboard: {results['dashboard_path']}")

if __name__ == '__main__':
    main()
