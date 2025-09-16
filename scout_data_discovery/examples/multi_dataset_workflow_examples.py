"""
Multi-Dataset Workflow Examples

Comprehensive examples demonstrating the complete multi-dataset workflow:
1. Dataset discovery using Scout
2. Column relationship mapping
3. Unified query generation
4. Multi-table data retrieval and joining
"""

import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta
import pandas as pd
import json

# Add the parent directory to Python path to import the package
sys.path.append(str(Path(__file__).parent.parent))

from src.scout_discovery import ScoutDataDiscovery
from src.workflow_orchestrator import MultiDatasetOrchestrator, WorkflowConfig
from src.multi_dataset_workflow import RelationshipGraph, UnifiedQuery
from src.column_relationship_mapper import RelationshipType


def setup_logging():
    """Setup logging for examples"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('workflow_examples.log')
        ]
    )
    return logging.getLogger(__name__)


def example_basic_workflow():
    """Example: Complete workflow from dataset to multi-table query"""
    print("=" * 80)
    print("EXAMPLE 1: Complete Multi-Dataset Workflow")
    print("=" * 80)

    logger = setup_logging()

    try:
        # Initialize Scout with enhanced client
        scout = ScoutDataDiscovery(
            use_enhanced_client=True,
            log_level="INFO"
        )

        # Initialize orchestrator
        config = WorkflowConfig(
            min_relationship_confidence=0.4,
            max_related_datasets=8,
            quality_threshold=75,
            enable_query_optimization=True,
            auto_export_results=True
        )

        orchestrator = MultiDatasetOrchestrator(scout, config, logger)

        print("🚀 Starting complete workflow for NYC 311 Service Requests...")

        # Run complete workflow starting from 311 dataset
        results = orchestrator.run_complete_workflow(
            source_dataset_id="erm2-nwe9",  # 311 Service Requests
            search_terms=["complaints", "violations", "housing", "noise"],
            date_range=(datetime.now() - timedelta(days=30), datetime.now())
        )

        print(f"\n📊 Workflow Results:")
        print(f"   Workflow ID: {results.workflow_id}")
        print(f"   Total execution time: {results.total_execution_time:.2f} seconds")
        print(f"   Steps executed: {len(results.steps_executed)}")
        print(f"   Successful: {results.is_successful()}")

        # Show step results
        print(f"\n📋 Step Execution Summary:")
        for step in results.steps_executed:
            status_icon = "✅" if step.status == "completed" else "❌" if step.status == "failed" else "⏳"
            print(f"   {status_icon} {step.step_name}: {step.status} ({step.duration:.2f}s)")

        # Show relationship analysis
        if results.relationship_graph:
            print(f"\n🔗 Relationship Graph Analysis:")
            analysis = orchestrator.analyze_relationships(results.relationship_graph)
            print(f"   Related datasets found: {analysis['total_related_datasets']}")
            print(f"   High-confidence relationships: {len(analysis['high_confidence_relationships'])}")
            print(f"   Join candidates: {len(analysis['join_candidates'])}")
            print(f"   Integration potential: {analysis['data_integration_potential']}")

            # Show relationship types
            if analysis['relationship_types']:
                print(f"   Relationship types:")
                for rel_type, count in analysis['relationship_types'].items():
                    print(f"     - {rel_type}: {count}")

        # Show query information
        if results.unified_query:
            print(f"\n📝 Unified Query Details:")
            print(f"   Primary dataset: {results.unified_query.primary_dataset}")
            print(f"   Total datasets: {len(results.unified_query.datasets)}")
            print(f"   Joins configured: {len(results.unified_query.joins)}")

            # Show joins
            if results.unified_query.joins:
                print(f"   Join operations:")
                for i, join in enumerate(results.unified_query.joins, 1):
                    print(f"     {i}. {join.left_dataset}.{join.left_column} -> {join.right_dataset}.{join.right_column}")
                    print(f"        Type: {join.join_type}, Confidence: {join.confidence:.3f}")

        # Show execution results
        if results.execution_results:
            print(f"\n💾 Query Execution Results:")
            exec_results = results.execution_results
            print(f"   Datasets processed: {exec_results.datasets_processed}")
            print(f"   Total rows fetched: {exec_results.total_rows_fetched:,}")
            print(f"   Final merged rows: {exec_results.total_rows_final:,}")
            print(f"   Join success rate: {exec_results.join_success_rate:.1%}")

            # Show individual dataset results
            if exec_results.individual_results:
                print(f"   Individual dataset results:")
                for dataset_id, df in exec_results.individual_results.items():
                    print(f"     - {dataset_id}: {len(df):,} rows, {len(df.columns)} columns")

        # Show final dataset info
        if results.final_dataset is not None:
            print(f"\n🎯 Final Integrated Dataset:")
            print(f"   Shape: {results.final_dataset.shape}")
            print(f"   Columns: {list(results.final_dataset.columns)[:10]}...")  # First 10 columns

            # Show sample data
            if not results.final_dataset.empty:
                print(f"   Sample data:")
                print(results.final_dataset.head(3).to_string())

        return results

    except Exception as e:
        print(f"❌ Complete workflow failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def example_step_by_step_workflow():
    """Example: Step-by-step workflow with manual control"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Step-by-Step Workflow with Manual Control")
    print("=" * 80)

    logger = setup_logging()

    try:
        # Initialize components
        scout = ScoutDataDiscovery(use_enhanced_client=True, log_level="INFO")
        orchestrator = MultiDatasetOrchestrator(scout, logger=logger)

        # Step 1: Discovery only
        print("🔍 Step 1: Discovering related datasets...")

        relationship_graph = orchestrator.run_discovery_only(
            source_dataset_id="h9gi-nx95",  # Motor Vehicle Collisions
            search_terms=["traffic", "violations", "parking", "transportation"],
            quality_threshold=70
        )

        print(f"   Found {len(relationship_graph.related_datasets)} related datasets")

        # Analyze relationships
        analysis = orchestrator.analyze_relationships(relationship_graph)
        print(f"   High-confidence relationships: {len(analysis['high_confidence_relationships'])}")

        # Step 2: Create custom query
        print("\n📝 Step 2: Creating unified query...")

        # Select specific datasets (example: pick top 3 related datasets)
        selected_datasets = list(relationship_graph.related_datasets.keys())[:3]
        print(f"   Selected datasets: {selected_datasets}")

        unified_query = orchestrator.create_query_from_graph(
            relationship_graph,
            selected_datasets=selected_datasets,
            join_strategy="high_confidence"
        )

        # Add custom filters
        unified_query.filters["h9gi-nx95"] = {
            "conditions": ["number_of_persons_injured > 0"],
            "date_column": "crash_date"
        }

        # Set date range
        unified_query.date_range = (datetime.now() - timedelta(days=14), datetime.now())

        print(f"   Query created with {len(unified_query.joins)} joins")
        print(f"   Date range: {unified_query.date_range[0].date()} to {unified_query.date_range[1].date()}")

        # Get query optimization suggestions
        suggestions = orchestrator.suggest_query_optimizations(unified_query)
        if suggestions:
            print(f"   Optimization suggestions:")
            for suggestion in suggestions:
                print(f"     - {suggestion['suggestion']} (Impact: {suggestion['impact']})")

        # Step 3: Execute query
        print("\n💾 Step 3: Executing unified query...")

        execution_results = orchestrator.execute_query_only(unified_query)

        print(f"   Execution completed in {execution_results.execution_time:.2f} seconds")
        print(f"   Final result: {execution_results.total_rows_final:,} rows")

        # Step 4: Analyze results
        if execution_results.merged_result is not None:
            print("\n📊 Step 4: Analyzing merged results...")
            merged_df = execution_results.merged_result

            print(f"   Merged dataset shape: {merged_df.shape}")
            print(f"   Column count by source:")

            # Count columns by dataset suffix
            column_sources = {}
            for col in merged_df.columns:
                if '_' in col:
                    source = col.split('_')[-1] if col.count('_') == 1 else 'primary'
                else:
                    source = 'primary'
                column_sources[source] = column_sources.get(source, 0) + 1

            for source, count in column_sources.items():
                print(f"     {source}: {count} columns")

            # Show data quality
            print(f"   Data quality:")
            print(f"     Missing values: {merged_df.isnull().sum().sum():,}")
            print(f"     Duplicate rows: {merged_df.duplicated().sum():,}")

        return {
            'relationship_graph': relationship_graph,
            'unified_query': unified_query,
            'execution_results': execution_results,
            'analysis': analysis
        }

    except Exception as e:
        print(f"❌ Step-by-step workflow failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def example_relationship_analysis():
    """Example: Deep dive into relationship analysis"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Deep Relationship Analysis")
    print("=" * 80)

    logger = setup_logging()

    try:
        # Initialize components
        scout = ScoutDataDiscovery(use_enhanced_client=True, log_level="INFO")
        orchestrator = MultiDatasetOrchestrator(scout, logger=logger)

        print("🔬 Analyzing relationships for Housing dataset...")

        # Discover relationships
        relationship_graph = orchestrator.run_discovery_only(
            source_dataset_id="wvxf-dwi5",  # Housing Maintenance Code Violations
            search_terms=["housing", "buildings", "violations", "inspections"],
            quality_threshold=60
        )

        # Detailed relationship analysis
        print(f"\n📊 Detailed Relationship Analysis:")

        for dataset_id, relationships in relationship_graph.related_datasets.items():
            print(f"\n🏢 Dataset: {dataset_id}")
            print(f"   Total relationships: {len(relationships)}")

            # Group by relationship type
            type_groups = {}
            for rel in relationships:
                rel_type = rel.relationship_type
                if rel_type not in type_groups:
                    type_groups[rel_type] = []
                type_groups[rel_type].append(rel)

            for rel_type, rels in type_groups.items():
                print(f"   {rel_type.value}: {len(rels)} relationships")

                # Show top relationships of this type
                top_rels = sorted(rels, key=lambda r: r.confidence_score, reverse=True)[:3]
                for rel in top_rels:
                    print(f"     - {rel.source_column.name} -> {rel.target_column.name}")
                    print(f"       Confidence: {rel.confidence_score:.3f}, Join potential: {rel.join_potential:.3f}")
                    print(f"       Notes: {rel.notes}")

        # Create visualization data for relationships
        print(f"\n📈 Relationship Network Summary:")

        all_relationships = []
        for dataset_id, relationships in relationship_graph.related_datasets.items():
            all_relationships.extend(relationships)

        # Summary statistics
        confidence_scores = [r.confidence_score for r in all_relationships]
        join_potentials = [r.join_potential for r in all_relationships]

        print(f"   Total relationships found: {len(all_relationships)}")
        print(f"   Average confidence score: {sum(confidence_scores)/len(confidence_scores):.3f}")
        print(f"   Average join potential: {sum(join_potentials)/len(join_potentials):.3f}")
        print(f"   High-confidence relationships (>0.7): {sum(1 for c in confidence_scores if c > 0.7)}")
        print(f"   Strong join candidates (>0.6): {sum(1 for j in join_potentials if j > 0.6)}")

        # Relationship type distribution
        type_counts = {}
        for rel in all_relationships:
            rel_type = rel.relationship_type.value
            type_counts[rel_type] = type_counts.get(rel_type, 0) + 1

        print(f"   Relationship type distribution:")
        for rel_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(all_relationships) * 100
            print(f"     {rel_type}: {count} ({percentage:.1f}%)")

        return relationship_graph, all_relationships

    except Exception as e:
        print(f"❌ Relationship analysis failed: {str(e)}")
        return None, None


def example_custom_query_building():
    """Example: Building custom unified queries"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Custom Unified Query Building")
    print("=" * 80)

    logger = setup_logging()

    try:
        scout = ScoutDataDiscovery(use_enhanced_client=True, log_level="INFO")
        orchestrator = MultiDatasetOrchestrator(scout, logger=logger)

        print("🔧 Building custom unified query...")

        # Start with relationship discovery
        relationship_graph = orchestrator.run_discovery_only(
            source_dataset_id="erm2-nwe9",  # 311 Service Requests
            search_terms=["noise", "violations", "housing"],
            quality_threshold=70
        )

        print(f"Found relationships with {len(relationship_graph.related_datasets)} datasets")

        # Create base query
        unified_query = orchestrator.create_query_from_graph(
            relationship_graph,
            join_strategy="best_match"
        )

        print(f"\n📝 Original Query:")
        print(f"   Datasets: {len(unified_query.datasets)}")
        print(f"   Joins: {len(unified_query.joins)}")

        # Customize the query
        print(f"\n🛠️ Customizing query...")

        # 1. Custom column selection
        unified_query.selected_columns = {
            "erm2-nwe9": ["unique_key", "created_date", "complaint_type", "descriptor", "borough", "status"],
            # Add specific columns for other datasets based on what was discovered
        }

        # Add columns for related datasets
        for dataset_id in unified_query.datasets[1:]:  # Skip primary dataset
            if dataset_id not in unified_query.selected_columns:
                unified_query.selected_columns[dataset_id] = []

            # Add common useful columns
            common_columns = ["date", "type", "category", "borough", "location", "status"]
            # This is simplified - in reality you'd inspect the actual schema
            unified_query.selected_columns[dataset_id].extend(["id", "date_field", "category_field"])

        # 2. Custom filters
        unified_query.filters = {
            "erm2-nwe9": {
                "conditions": [
                    "complaint_type in ('Noise - Residential', 'Heat/Hot Water', 'Blocked Driveway')",
                    "status != 'Closed'"
                ],
                "date_column": "created_date"
            }
        }

        # 3. Date range
        unified_query.date_range = (
            datetime.now() - timedelta(days=7),
            datetime.now()
        )

        # 4. Limit
        unified_query.limit = 5000

        print(f"   Custom column selection applied")
        print(f"   Filters added for primary dataset")
        print(f"   Date range: last 7 days")
        print(f"   Limit: {unified_query.limit:,} rows")

        # Show the final query structure
        print(f"\n📋 Final Query Structure:")
        print(f"   Primary dataset: {unified_query.primary_dataset}")
        print(f"   All datasets: {unified_query.datasets}")

        for dataset_id, columns in unified_query.selected_columns.items():
            print(f"   {dataset_id} columns: {columns[:5]}...")  # First 5 columns

        if unified_query.joins:
            print(f"   Joins:")
            for join in unified_query.joins:
                print(f"     {join.left_dataset}.{join.left_column} = {join.right_dataset}.{join.right_column} ({join.join_type})")

        # Export query as JSON
        query_json = unified_query.to_json()

        # Save to file
        query_file = Path("custom_unified_query.json")
        with open(query_file, 'w') as f:
            f.write(query_json)

        print(f"\n💾 Query exported to: {query_file}")
        print(f"   JSON size: {len(query_json):,} characters")

        # Show part of the JSON structure
        query_dict = json.loads(query_json)
        print(f"   Query ID: {query_dict['query_id']}")
        print(f"   Created at: {query_dict['created_at']}")

        return unified_query, query_json

    except Exception as e:
        print(f"❌ Custom query building failed: {str(e)}")
        return None, None


def example_workflow_callbacks():
    """Example: Using workflow callbacks for monitoring"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Workflow Monitoring with Callbacks")
    print("=" * 80)

    logger = setup_logging()

    try:
        scout = ScoutDataDiscovery(use_enhanced_client=True, log_level="INFO")
        orchestrator = MultiDatasetOrchestrator(scout, logger=logger)

        # Setup callbacks
        step_times = {}

        def on_step_start(step):
            step_times[step.step_name] = {'start': step.start_time}
            print(f"🟡 Starting: {step.step_name} at {step.start_time.strftime('%H:%M:%S')}")

        def on_step_complete(step):
            step_times[step.step_name]['end'] = step.end_time
            step_times[step.step_name]['duration'] = step.duration
            print(f"✅ Completed: {step.step_name} in {step.duration:.2f}s")

        def on_step_fail(step):
            print(f"❌ Failed: {step.step_name} - {step.error}")

        def on_workflow_complete(results):
            print(f"🎉 Workflow {results.workflow_id} completed successfully!")
            print(f"   Total time: {results.total_execution_time:.2f}s")

            if results.final_dataset is not None:
                print(f"   Final dataset: {results.final_dataset.shape}")

        # Register callbacks
        orchestrator.register_callback('step_start', on_step_start)
        orchestrator.register_callback('step_complete', on_step_complete)
        orchestrator.register_callback('step_fail', on_step_fail)
        orchestrator.register_callback('workflow_complete', on_workflow_complete)

        print("📊 Running monitored workflow...")

        # Run workflow with monitoring
        results = orchestrator.run_complete_workflow(
            source_dataset_id="qgea-i56i",  # NYPD Complaint Data
            search_terms=["crime", "violations", "complaints"],
            custom_config=WorkflowConfig(
                max_related_datasets=5,  # Smaller for faster demo
                quality_threshold=75
            )
        )

        # Show detailed step timing
        print(f"\n⏱️ Detailed Step Timing:")
        for step_name, timing in step_times.items():
            if 'duration' in timing:
                print(f"   {step_name}: {timing['duration']:.2f}s")

        # Show execution statistics
        stats = orchestrator.get_execution_statistics()
        print(f"\n📊 Orchestrator Statistics:")
        print(f"   Total workflows: {stats['total_workflows']}")
        print(f"   Successful workflows: {stats['successful_workflows']}")
        print(f"   Average execution time: {stats['avg_execution_time']:.2f}s")

        return results, step_times

    except Exception as e:
        print(f"❌ Monitored workflow failed: {str(e)}")
        return None, None


def run_all_workflow_examples():
    """Run all multi-dataset workflow examples"""
    print("🚀 Running All Multi-Dataset Workflow Examples")
    print("=" * 100)

    results = {}

    try:
        # Example 1: Complete workflow
        results['complete_workflow'] = example_basic_workflow()

        # Example 2: Step-by-step
        results['step_by_step'] = example_step_by_step_workflow()

        # Example 3: Relationship analysis
        results['relationship_analysis'] = example_relationship_analysis()

        # Example 4: Custom query building
        results['custom_query'] = example_custom_query_building()

        # Example 5: Workflow callbacks
        results['monitored_workflow'] = example_workflow_callbacks()

        # Summary
        print("\n" + "=" * 100)
        print("✅ All Multi-Dataset Workflow Examples Completed!")
        print("=" * 100)

        print("\n📋 Examples Summary:")
        print("   1. ✓ Complete workflow - End-to-end multi-dataset integration")
        print("   2. ✓ Step-by-step - Manual control over each phase")
        print("   3. ✓ Relationship analysis - Deep dive into column relationships")
        print("   4. ✓ Custom query building - Advanced query customization")
        print("   5. ✓ Workflow monitoring - Callbacks and progress tracking")

        print("\n🎯 Key Capabilities Demonstrated:")
        print("   • Automated dataset discovery using Scout intelligence")
        print("   • Advanced column relationship mapping across datasets")
        print("   • Intelligent join strategy selection")
        print("   • Optimized multi-dataset query execution")
        print("   • Comprehensive workflow orchestration")
        print("   • JSON-based query serialization for reusability")
        print("   • Real-time monitoring and callback system")
        print("   • Automatic result export and analysis")

        # Show files created
        files_created = []
        current_dir = Path(".")

        # Look for created files
        for pattern in ["*workflow*", "*query*", "*relationship*"]:
            files_created.extend(current_dir.glob(pattern))

        if files_created:
            print(f"\n📁 Files Created:")
            for file in files_created[:10]:  # Show first 10 files
                print(f"   - {file.name}")
            if len(files_created) > 10:
                print(f"   ... and {len(files_created) - 10} more files")

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
    try:
        results = run_all_workflow_examples()
    except KeyboardInterrupt:
        print("\n\n⏹️ Examples interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Examples failed: {str(e)}")
        import traceback
        traceback.print_exc()