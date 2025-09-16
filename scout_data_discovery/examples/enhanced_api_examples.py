"""
Enhanced API Integration Examples

Comprehensive examples demonstrating the integration between Scout Data Discovery
and the Enhanced NYC Open Data API Client with advanced SoQL capabilities.
"""

import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta
import pandas as pd

# Add the parent directory to Python path to import the package
sys.path.append(str(Path(__file__).parent.parent))

from src.scout_discovery import ScoutDataDiscovery
from src.enhanced_api_client import EnhancedNYCDataClient, ScoutIntegratedClient, SoQLQueryBuilder
from src.exceptions import ScoutDiscoveryError, APIError

# Try to import polars if available
try:
    import polars as pl
    POLARS_AVAILABLE = True
    print("✓ Polars available for high-performance data processing")
except ImportError:
    POLARS_AVAILABLE = False
    print("⚠️ Polars not available. Install with: pip install polars")


def example_basic_enhanced_client():
    """Example: Basic usage of Enhanced NYC Data Client"""
    print("=" * 70)
    print("EXAMPLE 1: Basic Enhanced NYC Data Client Usage")
    print("=" * 70)

    try:
        # Initialize enhanced client
        client = EnhancedNYCDataClient(
            app_token=None,  # Use your app token for production
            default_backend="pandas",
            rate_limit_delay=0.5
        )

        print("📱 Enhanced NYC Data Client initialized")

        # Basic dataset download
        print("\n🔍 Downloading 311 Service Requests sample...")
        basic_data = client.get_dataset("erm2-nwe9", limit=500)
        print(f"   Downloaded: {len(basic_data)} rows, {len(basic_data.columns)} columns")

        # Show some basic statistics
        if not basic_data.empty:
            print(f"   Date range: {basic_data.get('created_date', pd.Series()).min()} to {basic_data.get('created_date', pd.Series()).max()}")
            if 'complaint_type' in basic_data.columns:
                print(f"   Top complaint types: {basic_data['complaint_type'].value_counts().head(3).to_dict()}")

        return client, basic_data

    except Exception as e:
        print(f"❌ Basic enhanced client example failed: {str(e)}")
        return None, None


def example_advanced_soql_queries():
    """Example: Advanced SoQL queries with the enhanced client"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Advanced SoQL Query Building")
    print("=" * 70)

    try:
        client = EnhancedNYCDataClient(default_backend="pandas")

        # Example 1: Recent high-priority 311 requests
        print("\n🚨 Example 1: Recent High-Priority 311 Requests")

        query1 = (client.query()
                 .select("unique_key", "created_date", "complaint_type", "descriptor", "borough")
                 .where_date_range("created_date", datetime.now() - timedelta(days=7))
                 .where_in("complaint_type", ["Emergency", "Noise - Residential", "Heat/Hot Water"])
                 .where_not_null("borough")
                 .order_by("created_date", ascending=False)
                 .limit(100))

        print("   SoQL Query built:")
        for key, value in query1.build().items():
            print(f"     {key}: {value}")

        data1 = client.get_dataset("erm2-nwe9", query1)
        print(f"   ✓ Retrieved {len(data1)} recent high-priority requests")

        # Example 2: Traffic collision analysis
        print("\n🚗 Example 2: Traffic Collisions with Injuries")

        query2 = (client.query()
                 .select("crash_date", "borough", "number_of_persons_injured",
                        "number_of_persons_killed", "contributing_factor_vehicle_1")
                 .where_date_range("crash_date",
                                 datetime.now() - timedelta(days=30))
                 .where_numeric_range("number_of_persons_injured", min_val=1)
                 .where_not_null("borough")
                 .order_by("crash_date", ascending=False)
                 .limit(200))

        print("   Querying Motor Vehicle Collisions dataset...")
        try:
            data2 = client.get_dataset("h9gi-nx95", query2)
            print(f"   ✓ Retrieved {len(data2)} collision records with injuries")

            if not data2.empty and 'borough' in data2.columns:
                borough_stats = data2['borough'].value_counts()
                print(f"   Borough distribution: {borough_stats.head(3).to_dict()}")

        except Exception as e:
            print(f"   ⚠️ Collision data query failed: {str(e)}")

        # Example 3: Housing violations by neighborhood
        print("\n🏠 Example 3: Recent Housing Violations")

        query3 = (client.query()
                 .select("inspectiondate", "class", "violationstatus", "borough", "violationdescription")
                 .where_date_range("inspectiondate", datetime.now() - timedelta(days=14))
                 .where_in("class", ["A", "B", "C"])
                 .where_like("violationdescription", "heat")
                 .order_by("inspectiondate", ascending=False)
                 .limit(150))

        try:
            data3 = client.get_dataset("wvxf-dwi5", query3)
            print(f"   ✓ Retrieved {len(data3)} recent housing violations")

            if not data3.empty and 'class' in data3.columns:
                class_distribution = data3['class'].value_counts()
                print(f"   Violation classes: {class_distribution.to_dict()}")

        except Exception as e:
            print(f"   ⚠️ Housing violations query failed: {str(e)}")

        return [data1, data2 if 'data2' in locals() else None, data3 if 'data3' in locals() else None]

    except Exception as e:
        print(f"❌ Advanced SoQL examples failed: {str(e)}")
        return []


def example_polars_backend():
    """Example: Using Polars backend for high-performance processing"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: High-Performance Processing with Polars")
    print("=" * 70)

    if not POLARS_AVAILABLE:
        print("⚠️ Polars not available, skipping this example")
        return None

    try:
        # Initialize client with Polars backend
        client = EnhancedNYCDataClient(
            default_backend="polars",
            rate_limit_delay=0.3
        )

        print("🚀 Enhanced client initialized with Polars backend")

        # Download data using Polars
        print("\n📊 Downloading dataset as Polars DataFrame...")

        query = (client.query()
                .select("created_date", "complaint_type", "borough", "status")
                .where_date_range("created_date", datetime.now() - timedelta(days=3))
                .where_not_null("borough")
                .limit(2000))

        df_polars = client.get_dataset("erm2-nwe9", query)

        print(f"   ✓ Downloaded: {df_polars.shape[0]} rows, {df_polars.shape[1]} columns")
        print(f"   Memory usage: {df_polars.estimated_size('mb'):.2f} MB")

        # Perform high-speed analytics with Polars
        print("\n⚡ Performing high-speed analytics...")

        # Group by borough and complaint type
        analysis = (df_polars
                   .group_by(["borough", "complaint_type"])
                   .agg([
                       pl.count().alias("count"),
                       pl.col("created_date").min().alias("earliest"),
                       pl.col("created_date").max().alias("latest")
                   ])
                   .sort("count", descending=True)
                   .head(10))

        print("   Top 10 complaint type/borough combinations:")
        print(analysis)

        # Time-based analysis
        time_analysis = (df_polars
                        .with_columns([
                            pl.col("created_date").str.strptime(pl.Date, format="%Y-%m-%dT%H:%M:%S%.3f").alias("date"),
                        ])
                        .group_by("date")
                        .agg([
                            pl.count().alias("daily_count"),
                            pl.col("complaint_type").n_unique().alias("unique_types")
                        ])
                        .sort("date"))

        print(f"\n   Daily request volumes:")
        print(time_analysis)

        return df_polars, analysis

    except Exception as e:
        print(f"❌ Polars example failed: {str(e)}")
        return None, None


def example_streaming_large_datasets():
    """Example: Streaming large datasets for memory efficiency"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Streaming Large Datasets")
    print("=" * 70)

    try:
        client = EnhancedNYCDataClient(rate_limit_delay=0.2)

        print("🌊 Setting up streaming download for large dataset analysis")

        # Create query for recent data
        base_query = (client.query()
                     .select("created_date", "complaint_type", "borough", "location_type")
                     .where_date_range("created_date", datetime.now() - timedelta(days=5))
                     .where_not_null("borough"))

        print("   Streaming 311 requests in chunks of 1000 records...")

        chunk_count = 0
        total_rows = 0
        complaint_types = {}

        # Stream data in chunks
        for chunk_df in client.get_dataset_streaming(
            "erm2-nwe9",
            chunk_size=1000,
            query_builder=base_query
        ):
            chunk_count += 1
            total_rows += len(chunk_df)

            # Process each chunk
            if 'complaint_type' in chunk_df.columns:
                chunk_types = chunk_df['complaint_type'].value_counts()
                for complaint_type, count in chunk_types.items():
                    complaint_types[complaint_type] = complaint_types.get(complaint_type, 0) + count

            print(f"   Processed chunk {chunk_count}: {len(chunk_df)} rows")

            # Stop after a reasonable amount for demo
            if chunk_count >= 5:
                break

        print(f"\n✅ Streaming completed:")
        print(f"   Total chunks processed: {chunk_count}")
        print(f"   Total rows processed: {total_rows}")
        print(f"   Top complaint types: {dict(sorted(complaint_types.items(), key=lambda x: x[1], reverse=True)[:5])}")

        return complaint_types

    except Exception as e:
        print(f"❌ Streaming example failed: {str(e)}")
        return None


def example_scout_integration():
    """Example: Full Scout integration with enhanced client"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Scout Integration with Enhanced Client")
    print("=" * 70)

    try:
        # Initialize Scout with enhanced client
        scout = ScoutDataDiscovery(
            app_token=None,  # Use your token for production
            use_enhanced_client=True,
            default_backend="pandas",
            log_level="INFO"
        )

        print("🔍 Scout Data Discovery initialized with enhanced client")

        # Test the integrated discovery workflow
        print("\n🎯 Running integrated discovery and assessment...")

        results = scout.discover_and_assess_integrated(
            search_terms=["transportation", "traffic"],
            quality_threshold=75,
            max_datasets=8
        )

        print(f"\n📊 Integrated Discovery Results:")
        print(f"   High-quality datasets found: {results['statistics']['datasets_found']}")
        print(f"   Datasets assessed: {results['statistics']['datasets_assessed']}")
        print(f"   Quality threshold: {results['statistics']['quality_threshold']}")
        print(f"   Recommendations generated: {results['statistics']['recommendations_generated']}")

        # Show high-quality datasets
        if not results['high_quality_datasets'].empty:
            print(f"\n🏆 Top High-Quality Datasets:")
            for i, (_, dataset) in enumerate(results['high_quality_datasets'].head(3).iterrows(), 1):
                quality_score = results['quality_assessments'][dataset['id']]['overall_scores']['total_score']
                print(f"   {i}. {dataset['name'][:60]}... (Score: {quality_score:.1f})")

        # Demonstrate advanced search
        print(f"\n🔍 Advanced Search with Multiple Filters:")

        advanced_results = scout.search_datasets_advanced(
            query="housing",
            date_range=(datetime.now() - timedelta(days=180), datetime.now()),
            min_downloads=100,
            limit=20
        )

        print(f"   Found {len(advanced_results)} housing datasets updated in last 6 months with 100+ downloads")

        # Create and use SoQL query
        print(f"\n⚡ Creating Custom SoQL Query:")

        query_builder = scout.create_query_builder()
        if query_builder:
            custom_query = (query_builder
                          .select("created_date", "complaint_type", "borough", "status")
                          .where_date_range("created_date", datetime.now() - timedelta(days=2))
                          .where_in("borough", ["MANHATTAN", "BROOKLYN"])
                          .where_like("complaint_type", "noise")
                          .order_by("created_date", ascending=False)
                          .limit(300))

            print("   Custom query built for recent noise complaints in Manhattan/Brooklyn")

            # Use the query
            sample_data = scout.download_dataset_sample(
                "erm2-nwe9",
                query_builder=custom_query
            )

            print(f"   ✓ Retrieved {len(sample_data)} records using custom SoQL query")

            if not sample_data.empty and 'borough' in sample_data.columns:
                borough_counts = sample_data['borough'].value_counts()
                print(f"   Borough distribution: {borough_counts.to_dict()}")

        # Get API statistics
        stats = scout.get_api_statistics()
        if 'enhanced_client_stats' in stats:
            api_stats = stats['enhanced_client_stats']
            print(f"\n📈 API Usage Statistics:")
            print(f"   Total requests: {api_stats['total_requests']}")
            print(f"   Cache hit rate: {api_stats['cache_hit_rate']:.1f}%")
            print(f"   Default backend: {api_stats['default_backend']}")

        return scout, results

    except Exception as e:
        print(f"❌ Scout integration example failed: {str(e)}")
        return None, None


def example_batch_processing():
    """Example: Batch processing multiple datasets"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Batch Dataset Processing")
    print("=" * 70)

    try:
        client = EnhancedNYCDataClient(rate_limit_delay=0.3)

        print("📦 Setting up batch processing for multiple datasets")

        # Define multiple dataset configurations
        dataset_configs = [
            {
                'id': 'erm2-nwe9',
                'name': '311_recent',
                'query': (client.query()
                         .select("created_date", "complaint_type", "borough")
                         .where_date_range("created_date", datetime.now() - timedelta(days=1))
                         .limit(500))
            },
            {
                'id': 'h9gi-nx95',
                'name': 'collisions_recent',
                'query': (client.query()
                         .select("crash_date", "borough", "number_of_persons_injured")
                         .where_date_range("crash_date", datetime.now() - timedelta(days=2))
                         .where_numeric_range("number_of_persons_injured", min_val=1)
                         .limit(300))
            },
            {
                'id': 'wvxf-dwi5',
                'name': 'violations_recent',
                'query': (client.query()
                         .select("inspectiondate", "class", "borough")
                         .where_date_range("inspectiondate", datetime.now() - timedelta(days=3))
                         .limit(400))
            }
        ]

        print(f"   Batch processing {len(dataset_configs)} datasets...")

        # Execute batch download
        batch_results = client.batch_download(dataset_configs, max_workers=3)

        print(f"\n📊 Batch Processing Results:")
        print(f"   Successful downloads: {batch_results['successful_downloads']}/{len(dataset_configs)}")
        print(f"   Success rate: {batch_results['success_rate']:.1f}%")

        # Analyze each dataset
        print(f"\n🔍 Dataset Analysis:")
        for name, df in batch_results['datasets'].items():
            if df is not None:
                print(f"   {name}: {len(df)} rows, {len(df.columns)} columns")

                # Show sample data characteristics
                if 'borough' in df.columns:
                    borough_dist = df['borough'].value_counts()
                    print(f"     Borough distribution: {dict(borough_dist.head(3))}")

        # Combined analysis
        if batch_results['successful_downloads'] > 1:
            print(f"\n📈 Combined Analysis:")
            all_boroughs = set()
            total_records = 0

            for name, df in batch_results['datasets'].items():
                if df is not None:
                    total_records += len(df)
                    if 'borough' in df.columns:
                        all_boroughs.update(df['borough'].dropna().unique())

            print(f"   Total records across all datasets: {total_records}")
            print(f"   Unique boroughs found: {len(all_boroughs)}")
            print(f"   Borough list: {sorted(all_boroughs)}")

        return batch_results

    except Exception as e:
        print(f"❌ Batch processing example failed: {str(e)}")
        return None


def example_query_templates():
    """Example: Creating and using query templates"""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Query Templates and Reusability")
    print("=" * 70)

    try:
        client = EnhancedNYCDataClient()

        print("📝 Creating reusable query templates")

        # Create a template for recent high-priority requests
        priority_template = (client.query()
                           .select("created_date", "complaint_type", "descriptor", "borough", "status")
                           .where_date_range("created_date", datetime.now() - timedelta(days=1))
                           .where_in("complaint_type", ["Emergency", "Heat/Hot Water", "Water System"])
                           .where_not_null("borough")
                           .order_by("created_date", ascending=False)
                           .limit(200))

        print("   ✓ Priority requests template created")

        # Export template
        template_json = client.export_query_template(priority_template)
        print("   📄 Template exported as JSON:")
        print(f"      {len(template_json)} characters")

        # Save template to file
        template_file = Path("priority_requests_template.json")
        client.export_query_template(priority_template, str(template_file))
        print(f"   💾 Template saved to {template_file}")

        # Use the template
        print("\n🚀 Using the template to fetch data...")
        priority_data = client.get_dataset("erm2-nwe9", priority_template)
        print(f"   ✓ Retrieved {len(priority_data)} priority requests")

        if not priority_data.empty:
            print("   📊 Priority Request Analysis:")
            if 'complaint_type' in priority_data.columns:
                type_counts = priority_data['complaint_type'].value_counts()
                print(f"      Top types: {type_counts.head(3).to_dict()}")

            if 'borough' in priority_data.columns:
                borough_counts = priority_data['borough'].value_counts()
                print(f"      By borough: {borough_counts.to_dict()}")

        # Load and modify template
        print("\n🔄 Loading and modifying template...")

        if template_file.exists():
            loaded_template = client.load_query_template(str(template_file))

            # Modify the loaded template
            modified_template = (loaded_template
                               .where_in("borough", ["MANHATTAN", "BROOKLYN"])
                               .limit(100))

            print("   ✓ Template loaded and modified for Manhattan/Brooklyn only")

            # Use modified template
            modified_data = client.get_dataset("erm2-nwe9", modified_template)
            print(f"   ✓ Retrieved {len(modified_data)} records with modified template")

            # Cleanup
            template_file.unlink()
            print(f"   🗑️ Cleaned up template file")

        return priority_data

    except Exception as e:
        print(f"❌ Query templates example failed: {str(e)}")
        return None


def run_all_enhanced_examples():
    """Run all enhanced API integration examples"""
    print("🚀 Running All Enhanced API Integration Examples")
    print("=" * 80)

    results = {}

    try:
        # Run all examples
        client, basic_data = example_basic_enhanced_client()
        results['basic_client'] = (client, basic_data)

        advanced_data = example_advanced_soql_queries()
        results['advanced_queries'] = advanced_data

        if POLARS_AVAILABLE:
            polars_data, analysis = example_polars_backend()
            results['polars_processing'] = (polars_data, analysis)

        streaming_results = example_streaming_large_datasets()
        results['streaming'] = streaming_results

        scout, scout_results = example_scout_integration()
        results['scout_integration'] = (scout, scout_results)

        batch_results = example_batch_processing()
        results['batch_processing'] = batch_results

        query_templates = example_query_templates()
        results['query_templates'] = query_templates

        # Summary
        print("\n" + "=" * 80)
        print("✅ All Enhanced API Examples Completed Successfully!")
        print("=" * 80)

        print("\n📋 Summary of Examples:")
        print("   1. ✓ Basic Enhanced NYC Data Client usage")
        print("   2. ✓ Advanced SoQL query building and execution")
        print("   3. ✓ High-performance processing with Polars" if POLARS_AVAILABLE else "   3. ⚠️ Polars processing (not available)")
        print("   4. ✓ Streaming large datasets for memory efficiency")
        print("   5. ✓ Full Scout integration with enhanced client")
        print("   6. ✓ Batch processing multiple datasets")
        print("   7. ✓ Query templates and reusability")

        print("\n🎯 Key Features Demonstrated:")
        print("   • Advanced SoQL query building with fluent API")
        print("   • Memory-efficient streaming for large datasets")
        print("   • High-performance Polars backend support" if POLARS_AVAILABLE else "   • Pandas backend with optimization")
        print("   • Seamless Scout Data Discovery integration")
        print("   • Parallel batch processing capabilities")
        print("   • Reusable query templates")
        print("   • Comprehensive error handling and logging")
        print("   • Automated caching and rate limiting")

        return results

    except KeyboardInterrupt:
        print("\n\n⏹️ Examples interrupted by user")
        return results
    except Exception as e:
        print(f"\n\n❌ Examples failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return results


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run all examples
    try:
        results = run_all_enhanced_examples()
    except KeyboardInterrupt:
        print("\n\n⏹️ Examples interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Examples failed: {str(e)}")
        import traceback
        traceback.print_exc()