"""
Scout Data Discovery - Basic Usage Examples

This module demonstrates basic usage patterns for the Scout Data Discovery package.
"""

import sys
from pathlib import Path
import logging

# Add the parent directory to Python path to import the package
sys.path.append(str(Path(__file__).parent.parent))

from src.scout_discovery import ScoutDataDiscovery
from config.config_manager import ConfigManager


def example_basic_search():
    """Example: Basic dataset search"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Dataset Search")
    print("=" * 60)

    # Initialize with default configuration
    scout = ScoutDataDiscovery(log_level="INFO")

    try:
        # Search for transportation-related datasets
        search_terms = ["transportation", "traffic"]
        datasets_df = scout.search_datasets(search_terms, limit=10)

        print(f"\n📋 Found {len(datasets_df)} datasets")

        if not datasets_df.empty:
            print("\nTop 5 datasets by download count:")
            top_datasets = datasets_df.nlargest(5, 'download_count')
            for i, (_, row) in enumerate(top_datasets.iterrows(), 1):
                print(f"{i}. {row['name']} (Downloads: {row['download_count']})")

        return datasets_df

    except Exception as e:
        print(f"❌ Search failed: {str(e)}")
        return None


def example_quality_assessment():
    """Example: Quality assessment of a specific dataset"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Dataset Quality Assessment")
    print("=" * 60)

    scout = ScoutDataDiscovery(log_level="INFO")

    try:
        # Use a known NYC dataset ID (Motor Vehicle Collisions)
        dataset_id = "h9gi-nx95"
        print(f"\n🔍 Assessing quality of dataset: {dataset_id}")

        # Download sample and assess quality
        sample_df = scout.download_dataset_sample(dataset_id, sample_size=500)

        if sample_df.empty:
            print("❌ Dataset is empty or unavailable")
            return None

        print(f"📊 Downloaded sample: {len(sample_df)} rows, {len(sample_df.columns)} columns")

        # Assess quality
        assessment = scout.assess_dataset_quality(dataset_id, sample_df)

        if 'error' not in assessment:
            scores = assessment['overall_scores']
            print(f"\n📈 Quality Assessment Results:")
            print(f"   Overall Score: {scores['total_score']:.1f}/100 (Grade: {scores['grade']})")
            print(f"   Completeness: {scores['completeness_score']:.1f}/100")
            print(f"   Consistency: {scores['consistency_score']:.1f}/100")
            print(f"   Accuracy: {scores['accuracy_score']:.1f}/100")
            print(f"   Timeliness: {scores['timeliness_score']:.1f}/100")
            print(f"   Usability: {scores['usability_score']:.1f}/100")

            # Show some additional insights
            completeness = assessment['completeness']
            print(f"\n🔍 Dataset Insights:")
            print(f"   Missing data: {completeness['missing_percentage']:.1f}%")
            print(f"   Complete columns: {len(completeness['complete_columns'])}")
            print(f"   Empty columns: {len(completeness['empty_columns'])}")

        else:
            print(f"❌ Quality assessment failed: {assessment['error']}")

        return assessment

    except Exception as e:
        print(f"❌ Quality assessment failed: {str(e)}")
        return None


def example_full_pipeline():
    """Example: Complete discovery pipeline"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Complete Discovery Pipeline")
    print("=" * 60)

    # Custom configuration
    custom_config = {
        'data': {
            'quality_threshold': 75,
            'default_sample_size': 300
        },
        'logging': {
            'level': 'INFO'
        }
    }

    scout = ScoutDataDiscovery(config=custom_config, log_level="INFO")

    try:
        # Run complete pipeline
        search_terms = ["housing", "health"]
        results = scout.run_discovery_pipeline(
            search_terms=search_terms,
            max_assessments=5,  # Limit for demo
            include_recommendations=True,
            export_results=True
        )

        # Display results summary
        stats = results['pipeline_stats']
        print(f"\n📊 Pipeline Results Summary:")
        print(f"   Execution Time: {stats['execution_time_seconds']:.1f} seconds")
        print(f"   Datasets Found: {stats['datasets_found']}")
        print(f"   Datasets Assessed: {stats['datasets_assessed']}")
        print(f"   Assessment Success Rate: {stats['assessment_success_rate']:.1f}%")
        print(f"   Recommendations Generated: {stats['recommendations_generated']}")

        if stats.get('quality_statistics'):
            qual_stats = stats['quality_statistics']
            print(f"   Average Quality Score: {qual_stats['mean_score']:.1f}/100")

        # Show top quality datasets
        if results['quality_assessments']:
            print(f"\n🏆 Top Quality Datasets:")
            quality_scores = {
                dataset_id: assessment.get('overall_scores', {}).get('total_score', 0)
                for dataset_id, assessment in results['quality_assessments'].items()
                if 'error' not in assessment
            }

            top_quality = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)
            for i, (dataset_id, score) in enumerate(top_quality[:3], 1):
                # Get dataset name from metadata
                dataset_name = "Unknown"
                metadata_cache = results.get('metadata_cache', {})
                if dataset_id in metadata_cache:
                    dataset_name = metadata_cache[dataset_id].get('name', 'Unknown')[:50]

                print(f"   {i}. {dataset_name}... (Score: {score:.1f})")

        return results

    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        return None


def example_custom_configuration():
    """Example: Using custom configuration"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Custom Configuration")
    print("=" * 60)

    try:
        # Create custom configuration manager
        config_manager = ConfigManager()

        # Show current configuration
        print("📋 Current API Configuration:")
        api_config = config_manager.get_section('api')
        for key, value in api_config.items():
            print(f"   {key}: {value}")

        # Modify configuration
        config_manager.set('data.quality_threshold', 80)
        config_manager.set('api.rate_limit_delay', 1.0)

        print(f"\n🔧 Modified quality threshold to: {config_manager.get('data.quality_threshold')}")
        print(f"🔧 Modified rate limit delay to: {config_manager.get('api.rate_limit_delay')}")

        # Use with ScoutDataDiscovery
        scout = ScoutDataDiscovery(config=config_manager.to_dict())
        print(f"\n✅ ScoutDataDiscovery initialized with custom config")

        return config_manager

    except Exception as e:
        print(f"❌ Configuration example failed: {str(e)}")
        return None


def example_error_handling():
    """Example: Error handling and robustness"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Error Handling and Robustness")
    print("=" * 60)

    scout = ScoutDataDiscovery(log_level="WARNING")  # Reduce logging for cleaner output

    # Test 1: Invalid dataset ID
    print("🧪 Test 1: Invalid Dataset ID")
    try:
        invalid_df = scout.download_dataset_sample("invalid-dataset-id")
        print(f"   Result: {len(invalid_df)} rows (expected: 0)")
    except Exception as e:
        print(f"   Caught expected error: {type(e).__name__}")

    # Test 2: Empty search results
    print("\n🧪 Test 2: Search with No Results")
    try:
        empty_results = scout.search_datasets(["very_unlikely_search_term_123456"])
        print(f"   Result: {len(empty_results)} datasets found")
    except Exception as e:
        print(f"   Unexpected error: {str(e)}")

    # Test 3: Quality assessment on empty dataset
    print("\n🧪 Test 3: Quality Assessment on Empty Data")
    try:
        import pandas as pd
        empty_df = pd.DataFrame()
        assessment = scout.assess_dataset_quality("test-empty", empty_df)
        if 'error' in assessment:
            print(f"   Correctly handled empty dataset: {assessment['error']}")
        else:
            print(f"   Unexpected success on empty dataset")
    except Exception as e:
        print(f"   Caught error: {type(e).__name__}")

    print("\n✅ Error handling tests completed")


def run_all_examples():
    """Run all examples in sequence"""
    print("🚀 Running All Scout Data Discovery Examples")
    print("=" * 80)

    # Run examples
    datasets = example_basic_search()
    assessment = example_quality_assessment()
    pipeline_results = example_full_pipeline()
    config_manager = example_custom_configuration()
    example_error_handling()

    print("\n" + "=" * 80)
    print("✅ All examples completed successfully!")
    print("=" * 80)

    # Summary of what was accomplished
    print("\n📋 Summary of Examples:")
    print("   1. ✓ Basic dataset search")
    print("   2. ✓ Quality assessment of individual dataset")
    print("   3. ✓ Complete discovery pipeline with recommendations")
    print("   4. ✓ Custom configuration management")
    print("   5. ✓ Error handling and robustness testing")

    # Show files created
    exports_dir = Path("exports")
    if exports_dir.exists():
        export_files = list(exports_dir.glob("*.csv")) + list(exports_dir.glob("*.json")) + list(exports_dir.glob("*.txt"))
        if export_files:
            print(f"\n📁 Files created:")
            for file in export_files[:5]:  # Show first 5 files
                print(f"   - {file.name}")
            if len(export_files) > 5:
                print(f"   ... and {len(export_files) - 5} more files")


if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        run_all_examples()
    except KeyboardInterrupt:
        print("\n\n⏹️ Examples interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Examples failed with error: {str(e)}")
        import traceback
        traceback.print_exc()