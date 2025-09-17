"""
Example: Complete Interactive Dashboard with Scout Data Discovery Integration

This example demonstrates how to use the interactive dashboard components
to create a complete data exploration application.
"""
import sys
import os

# Add the dashboard components to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from components.data.connector import ScoutDataConnector
from components.data.processor import DataProcessor
from components.data.validator import DataValidator
from components.charts.factory import ChartFactory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    
    # Generate sample dataset
    n_rows = 1000
    data = {
        'date': pd.date_range('2023-01-01', periods=n_rows, freq='D'),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        'value': np.random.normal(100, 15, n_rows),
        'count': np.random.poisson(10, n_rows),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_rows),
        'is_active': np.random.choice([True, False], n_rows),
        'score': np.random.uniform(0, 100, n_rows)
    }
    
    return pd.DataFrame(data)

def demonstrate_data_processing():
    """Demonstrate data processing capabilities"""
    print("\n=== Data Processing Demo ===")
    
    # Create sample data
    df = create_sample_data()
    print(f"Created sample dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # Initialize processor
    processor = DataProcessor()
    
    # Infer data types
    data_types = processor.infer_data_types(df)
    print("\nInferred data types:")
    for col, dtype in data_types.items():
        print(f"  {col}: {dtype}")
    
    # Clean data
    df_clean = processor.clean_data(df)
    print(f"\nCleaned data: {len(df_clean)} rows remaining")
    
    # Get data summary
    summary = processor.get_data_summary(df_clean)
    print(f"\nData summary:")
    print(f"  Shape: {summary['shape']}")
    print(f"  Memory usage: {summary['memory_usage'] / 1024:.1f} KB")
    print(f"  Numeric columns: {len(summary['numeric_summary'])}")
    print(f"  Categorical columns: {len(summary['categorical_summary'])}")
    
    return df_clean

def demonstrate_data_validation():
    """Demonstrate data validation capabilities"""
    print("\n=== Data Validation Demo ===")
    
    # Create sample data
    df = create_sample_data()
    
    # Add some quality issues for demonstration
    df.loc[50:100, 'value'] = np.nan  # Missing values
    df.loc[200:210, 'category'] = 'E'  # New category
    df = pd.concat([df, df.head(10)])  # Duplicates
    
    # Initialize validator
    validator = DataValidator()
    
    # Perform validation
    validation_results = validator.validate_dataset(df)
    
    print(f"Overall quality score: {validation_results['overall_score']:.2f}")
    print(f"Quality level: {validation_results['quality_level']}")
    
    print(f"\nValidation results:")
    for rule, result in validation_results['rule_results'].items():
        print(f"  {rule}: {result['score']:.2f}")
    
    if validation_results['issues']:
        print(f"\nIssues found:")
        for issue in validation_results['issues'][:5]:  # Show first 5 issues
            print(f"  - {issue}")
    
    if validation_results['recommendations']:
        print(f"\nRecommendations:")
        for rec in validation_results['recommendations'][:3]:  # Show first 3 recommendations
            print(f"  - {rec}")

def demonstrate_chart_factory():
    """Demonstrate chart factory capabilities"""
    print("\n=== Chart Factory Demo ===")
    
    # Create sample data
    df = create_sample_data()
    
    # Initialize chart factory
    chart_factory = ChartFactory()
    
    # Suggest chart types
    suggestions = chart_factory.suggest_chart_types(df, 'category', 'value')
    print(f"Suggested chart types for 'category' vs 'value': {suggestions}")
    
    # Create different types of charts
    chart_examples = [
        ('bar', 'category', 'value', None),
        ('scatter', 'value', 'score', 'category'),
        ('line', 'date', 'value', None),
        ('histogram', 'value', None, None),
        ('box', 'category', 'value', None)
    ]
    
    print(f"\nCreating example charts:")
    for chart_type, x_col, y_col, color_col in chart_examples:
        try:
            fig = chart_factory.create_chart(
                df=df,
                chart_type=chart_type,
                x_column=x_col,
                y_column=y_col,
                color_column=color_col
            )
            print(f"  ✓ Created {chart_type} chart ({x_col} vs {y_col})")
        except Exception as e:
            print(f"  ✗ Failed to create {chart_type} chart: {e}")
    
    # Get chart options
    options = chart_factory.get_chart_options(df)
    print(f"\nAvailable options:")
    print(f"  X columns: {len(options['x_columns'])}")
    print(f"  Y columns: {len(options['y_columns'])}")
    print(f"  Color columns: {len(options['color_columns'])}")

def demonstrate_scout_connector():
    """Demonstrate Scout connector capabilities (mock)"""
    print("\n=== Scout Connector Demo ===")
    
    # Note: This would normally connect to a real Scout API
    # For demo purposes, we'll show the interface
    
    connector = ScoutDataConnector(base_url="http://localhost:8000")
    
    print("Scout Data Connector initialized")
    print("Available methods:")
    print("  - search_datasets(query, domain, limit)")
    print("  - assess_dataset_quality(dataset_id)")
    print("  - download_dataset_sample(dataset_id, sample_size)")
    print("  - get_dataset_metadata(dataset_id)")
    print("  - advanced_search(search_criteria)")
    print("  - get_available_domains()")
    
    # Show cache capabilities
    print(f"\nCache configuration:")
    print(f"  TTL: {connector.cache_ttl} seconds")
    print(f"  Current cache size: {len(connector._cache)} entries")
    
    # Demo cache stats
    stats = connector.get_cache_stats()
    print(f"  Cache stats: {stats}")

def run_complete_demo():
    """Run complete demonstration of all components"""
    print("Interactive Dashboard Components Demo")
    print("=" * 40)
    
    try:
        # Demonstrate each component
        df = demonstrate_data_processing()
        demonstrate_data_validation()
        demonstrate_chart_factory()
        demonstrate_scout_connector()
        
        print("\n=== Integration Example ===")
        print("Complete workflow demonstration:")
        
        # 1. Data processing
        processor = DataProcessor()
        df_processed = processor.clean_data(df)
        print(f"1. Processed dataset: {len(df_processed)} rows")
        
        # 2. Data validation
        validator = DataValidator()
        quality = validator.quick_quality_check(df_processed)
        print(f"2. Quality check: {quality['missing_ratio']:.1%} missing data")
        
        # 3. Chart creation
        factory = ChartFactory()
        fig = factory.create_chart(df_processed, 'scatter', 'value', 'score', 'category')
        print("3. Created scatter plot visualization")
        
        print("\n✓ All components working successfully!")
        print("\nTo run the full dashboard:")
        print("  python app.py")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_complete_demo()