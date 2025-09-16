#!/usr/bin/env python3
"""
Debug version of the 311 workflow to identify and fix issues
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import json
import traceback

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("🚀 Starting Scout 311 Workflow Debug")
print("=" * 50)

def safe_import(module_name, from_module=None):
    """Safely import modules with error handling"""
    try:
        if from_module:
            module = __import__(from_module, fromlist=[module_name])
            return getattr(module, module_name)
        else:
            return __import__(module_name)
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error importing {module_name}: {e}")
        return None

# Test imports
print("\n📚 Testing imports...")

# Core imports
ScoutDataDiscovery = safe_import('ScoutDataDiscovery', 'src.scout_discovery')
DatasetRelationshipGraph = safe_import('DatasetRelationshipGraph', 'src.dataset_relationship_graph')
SoQLQueryBuilder = safe_import('SoQLQueryBuilder', 'src.enhanced_api_client')

# Check availability
available_modules = {
    'ScoutDataDiscovery': ScoutDataDiscovery is not None,
    'DatasetRelationshipGraph': DatasetRelationshipGraph is not None,
    'SoQLQueryBuilder': SoQLQueryBuilder is not None
}

print("Module availability:")
for module, available in available_modules.items():
    status = "✅" if available else "❌"
    print(f"  {status} {module}")

if not all(available_modules.values()):
    print("\n❌ Some required modules are not available. Cannot proceed.")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("\n🔧 Step 1: Initialize Scout Data Discovery")
print("-" * 40)

try:
    # Simple configuration
    config = {
        'api': {
            'rate_limit_delay': 1.0,
            'request_timeout': 30,
            'retry_attempts': 2
        },
        'data': {
            'quality_threshold': 60,
            'default_sample_size': 500  # Smaller sample for testing
        }
    }

    # Initialize Scout
    scout = ScoutDataDiscovery(
        config=config,
        log_level="WARNING",  # Reduce log noise
        use_enhanced_client=True,
        max_workers=2
    )

    print("✅ Scout Data Discovery initialized successfully!")

except Exception as e:
    print(f"❌ Failed to initialize Scout: {e}")
    traceback.print_exc()
    sys.exit(1)

print(f"\n🔍 Step 2: Search for 311 datasets")
print("-" * 40)

try:
    # Simple search terms first
    search_terms = ["311", "service requests"]

    print(f"Searching for: {', '.join(search_terms)}")

    datasets_311 = scout.search_datasets(
        search_terms=search_terms,
        limit=10  # Smaller limit for testing
    )

    print(f"Found {len(datasets_311)} datasets")

    if not datasets_311.empty:
        print("\nTop 3 datasets by popularity:")
        top_3 = datasets_311.nlargest(3, 'download_count')
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            name = row['name'][:50] + '...' if len(row['name']) > 50 else row['name']
            print(f"  {i}. {name}")
            print(f"     Downloads: {row['download_count']:,}")
            print(f"     ID: {row['id']}")
            print()
    else:
        print("⚠️ No 311 datasets found. This might be expected if none exist.")

except Exception as e:
    print(f"❌ Search failed: {e}")
    traceback.print_exc()
    datasets_311 = pd.DataFrame()

print(f"\n📊 Step 3: Analyze dataset (if available)")
print("-" * 40)

if not datasets_311.empty:
    try:
        # Get the most popular dataset
        primary_dataset = datasets_311.nlargest(1, 'download_count').iloc[0]
        primary_id = primary_dataset['id']

        print(f"Analyzing dataset: {primary_dataset['name'][:60]}...")
        print(f"Dataset ID: {primary_id}")

        # Try to download a small sample
        print("Downloading sample...")
        sample_df = scout.download_dataset_sample(primary_id, sample_size=100)

        if not sample_df.empty:
            print(f"✅ Downloaded sample: {len(sample_df)} rows × {len(sample_df.columns)} columns")

            # Show basic info
            print(f"Shape: {sample_df.shape}")
            print(f"Columns: {list(sample_df.columns[:5])}{'...' if len(sample_df.columns) > 5 else ''}")

        else:
            print("⚠️ Dataset appears to be empty or inaccessible")
            sample_df = pd.DataFrame()

    except Exception as e:
        print(f"❌ Dataset analysis failed: {e}")
        traceback.print_exc()
        sample_df = pd.DataFrame()
        primary_id = None
else:
    print("⚠️ No datasets to analyze")
    sample_df = pd.DataFrame()
    primary_id = None

print(f"\n🔍 Step 4: Quality assessment (if data available)")
print("-" * 40)

if not sample_df.empty and primary_id:
    try:
        print(f"Assessing quality of: {primary_dataset['name'][:40]}...")

        quality_assessment = scout.assess_dataset_quality(
            dataset_id=primary_id,
            df=sample_df
        )

        if 'error' not in quality_assessment:
            scores = quality_assessment['overall_scores']
            print(f"✅ Quality Assessment Results:")
            print(f"   Overall Score: {scores['total_score']:.1f}/100 (Grade: {scores['grade']})")
            print(f"   Completeness:  {scores['completeness_score']:.1f}/100")
            print(f"   Consistency:   {scores['consistency_score']:.1f}/100")
            print(f"   Accuracy:      {scores['accuracy_score']:.1f}/100")
            print(f"   Timeliness:    {scores['timeliness_score']:.1f}/100")
            print(f"   Usability:     {scores['usability_score']:.1f}/100")
        else:
            print(f"❌ Quality assessment failed: {quality_assessment['error']}")

    except Exception as e:
        print(f"❌ Quality assessment error: {e}")
        traceback.print_exc()
else:
    print("⚠️ Skipping quality assessment - no data available")

print(f"\n🌐 Step 5: Search for related datasets")
print("-" * 40)

try:
    # Broader search for related datasets
    related_terms = ["complaints", "city services", "public works", "violations"]

    print(f"Searching for related datasets: {', '.join(related_terms)}")

    related_datasets = scout.search_datasets(
        search_terms=related_terms,
        limit=8  # Small limit for testing
    )

    print(f"Found {len(related_datasets)} related datasets")

    # Combine datasets if both exist
    if not datasets_311.empty and not related_datasets.empty:
        all_datasets = pd.concat([datasets_311, related_datasets], ignore_index=True)
        all_datasets = all_datasets.drop_duplicates(subset=['id']).reset_index(drop=True)
        print(f"Combined total: {len(all_datasets)} unique datasets")
    elif not datasets_311.empty:
        all_datasets = datasets_311.copy()
        print(f"Using only 311 datasets: {len(all_datasets)}")
    elif not related_datasets.empty:
        all_datasets = related_datasets.copy()
        print(f"Using only related datasets: {len(all_datasets)}")
    else:
        all_datasets = pd.DataFrame()
        print("⚠️ No datasets found for relationship analysis")

except Exception as e:
    print(f"❌ Related dataset search failed: {e}")
    traceback.print_exc()
    all_datasets = pd.DataFrame()

print(f"\n🔗 Step 6: Relationship analysis")
print("-" * 40)

if not all_datasets.empty and len(all_datasets) >= 2:
    try:
        print(f"Analyzing relationships between {len(all_datasets)} datasets...")

        # Initialize relationship graph
        relationship_graph = DatasetRelationshipGraph()
        relationship_graph.add_datasets(all_datasets)

        print("Calculating relationships...")

        # Calculate with lower threshold for testing
        stats = relationship_graph.calculate_relationships(
            content_weight=0.4,
            structural_weight=0.2,
            metadata_weight=0.2,
            tag_weight=0.15,
            category_weight=0.05,
            similarity_threshold=0.15  # Lower threshold to find more relationships
        )

        print(f"✅ Relationship Analysis Results:")
        print(f"   Datasets Analyzed: {stats['datasets_analyzed']}")
        print(f"   Relationships Found: {stats['relationships_found']}")
        print(f"   Graph Density: {stats['graph_density']:.3f}")
        print(f"   Connected Components: {stats['connected_components']}")

        # Find relationships for primary dataset if available
        if primary_id and stats['relationships_found'] > 0:
            print(f"\nDatasets related to primary 311 dataset:")
            related_to_311 = relationship_graph.get_related_datasets(
                dataset_id=primary_id,
                top_n=3,
                min_similarity=0.1
            )

            if related_to_311:
                for i, rel in enumerate(related_to_311, 1):
                    print(f"  {i}. {rel['name'][:40]}...")
                    print(f"     Similarity: {rel['similarity_score']:.3f}")
                    print(f"     Reasons: {', '.join(rel['relationship_reasons'])}")
            else:
                print("  No related datasets found above threshold")

    except Exception as e:
        print(f"❌ Relationship analysis failed: {e}")
        traceback.print_exc()
        relationship_graph = None
        stats = None
else:
    print("⚠️ Not enough datasets for relationship analysis")
    relationship_graph = None
    stats = None

print(f"\n🎨 Step 7: Create simple visualization")
print("-" * 40)

if relationship_graph and stats and stats['relationships_found'] > 0:
    try:
        print("Creating network visualization...")

        # Try to create a simple visualization
        viz_path = relationship_graph.visualize_graph(
            output_path="debug_network.png",
            layout='spring',
            node_size_by='download_count',
            show_labels=False,  # Turn off labels for simpler viz
            figsize=(10, 8)
        )

        if viz_path:
            print(f"✅ Visualization saved to: {viz_path}")
        else:
            print("⚠️ Visualization created but not saved")

    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        print("This is often due to missing display or font issues")

else:
    print("⚠️ No relationships to visualize")

print(f"\n📄 Step 8: Generate summary")
print("-" * 40)

# Create a simple summary
summary = []
summary.append("SCOUT 311 WORKFLOW DEBUG SUMMARY")
summary.append("=" * 40)
summary.append(f"Run completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
summary.append("")

summary.append("RESULTS:")
if not datasets_311.empty:
    summary.append(f"✅ Found {len(datasets_311)} 311 datasets")
else:
    summary.append("❌ No 311 datasets found")

if not all_datasets.empty:
    summary.append(f"✅ Found {len(all_datasets)} total datasets")
else:
    summary.append("❌ No datasets found for analysis")

if stats:
    summary.append(f"✅ Found {stats['relationships_found']} dataset relationships")
else:
    summary.append("❌ No relationship analysis completed")

summary.append("")
summary.append("NEXT STEPS:")
summary.append("1. If successful, you can now run the full Jupyter notebook")
summary.append("2. If errors occurred, check the specific error messages above")
summary.append("3. Common issues: API rate limits, network connectivity, missing dependencies")

print("\n".join(summary))

# Save summary
try:
    with open("debug_workflow_summary.txt", "w") as f:
        f.write("\n".join(summary))
    print(f"\n📝 Debug summary saved to: debug_workflow_summary.txt")
except Exception as e:
    print(f"❌ Could not save summary: {e}")

print(f"\n🎉 Debug workflow completed!")
print("If this ran successfully, the Jupyter notebook should work too.")
print("Check the output above for any error messages.")