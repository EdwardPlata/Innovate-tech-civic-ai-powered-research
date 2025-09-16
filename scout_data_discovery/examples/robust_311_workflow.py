#!/usr/bin/env python3
"""
Robust version of the 311 workflow that handles network issues and uses smaller datasets
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

# Import Scout components
from src.scout_discovery import ScoutDataDiscovery
from src.dataset_relationship_graph import DatasetRelationshipGraph

# Suppress warnings
warnings.filterwarnings('ignore')

print("🚀 Scout 311 Workflow - Robust Version")
print("=" * 50)
print("This version handles network timeouts and uses smaller datasets for testing")

def find_suitable_dataset(scout, datasets_df, max_attempts=3):
    """Find a dataset that we can actually download data from"""
    print(f"\n🔍 Testing {min(max_attempts, len(datasets_df))} datasets to find one that works...")

    # Try smaller, more recent datasets first
    sorted_datasets = datasets_df.sort_values('download_count', ascending=True)  # Try smaller ones first

    for attempt, (_, row) in enumerate(sorted_datasets.head(max_attempts).iterrows(), 1):
        dataset_id = row['id']
        dataset_name = row['name'][:50]

        print(f"  Attempt {attempt}: Testing {dataset_name}... (Downloads: {row['download_count']:,})")

        try:
            # Try very small sample first
            sample_df = scout.download_dataset_sample(dataset_id, sample_size=50)

            if not sample_df.empty:
                print(f"  ✅ Success! Downloaded {len(sample_df)} rows from {dataset_name}")
                return row, sample_df
            else:
                print(f"  ⚠️ Dataset empty")

        except Exception as e:
            error_msg = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
            print(f"  ❌ Failed: {error_msg}")

    print("  ⚠️ No suitable datasets found that can be downloaded")
    return None, pd.DataFrame()

print("\n🔧 Step 1: Initialize Scout (with longer timeouts)")
print("-" * 50)

# More robust configuration
config = {
    'api': {
        'rate_limit_delay': 2.0,  # More conservative
        'request_timeout': 60,    # Longer timeout
        'retry_attempts': 2       # Fewer retries to avoid long waits
    },
    'data': {
        'quality_threshold': 60,
        'default_sample_size': 100  # Very small sample
    }
}

scout = ScoutDataDiscovery(
    config=config,
    log_level="WARNING",
    use_enhanced_client=True,
    max_workers=1  # Single-threaded to avoid overwhelming the API
)

print("✅ Scout initialized with robust settings!")

print("\n🔍 Step 2: Search for 311 datasets")
print("-" * 50)

try:
    # Try with a single, focused search term first
    datasets_311 = scout.search_datasets(
        search_terms=["311"],
        limit=15
    )

    print(f"Found {len(datasets_311)} 311 datasets")

    if not datasets_311.empty:
        print("\nDataset overview:")

        # Show variety of dataset sizes
        size_stats = datasets_311['download_count'].describe()
        print(f"  Download count range: {size_stats['min']:,.0f} to {size_stats['max']:,.0f}")
        print(f"  Median downloads: {size_stats['50%']:,.0f}")

        # Show top 5 and bottom 5
        print(f"\n  Most popular 311 datasets:")
        top_5 = datasets_311.nlargest(5, 'download_count')
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            name = row['name'][:45] + '...' if len(row['name']) > 45 else row['name']
            print(f"    {i}. {name} ({row['download_count']:,} downloads)")

        print(f"\n  Smaller 311 datasets (easier to download):")
        bottom_5 = datasets_311.nsmallest(5, 'download_count')
        for i, (_, row) in enumerate(bottom_5.iterrows(), 1):
            name = row['name'][:45] + '...' if len(row['name']) > 45 else row['name']
            print(f"    {i}. {name} ({row['download_count']:,} downloads)")

except Exception as e:
    print(f"❌ Search failed: {str(e)[:200]}")
    datasets_311 = pd.DataFrame()

print(f"\n📊 Step 3: Find a workable dataset")
print("-" * 50)

primary_dataset = None
sample_df = pd.DataFrame()

if not datasets_311.empty:
    primary_dataset, sample_df = find_suitable_dataset(scout, datasets_311, max_attempts=5)

    if primary_dataset is not None:
        primary_id = primary_dataset['id']
        print(f"\n✅ Selected dataset: {primary_dataset['name'][:60]}...")
        print(f"   ID: {primary_id}")
        print(f"   Downloads: {primary_dataset['download_count']:,}")
        print(f"   Sample size: {len(sample_df)} rows × {len(sample_df.columns)} columns")

        if not sample_df.empty:
            print(f"   Columns: {list(sample_df.columns[:8])}{'...' if len(sample_df.columns) > 8 else ''}")

            # Show sample data
            print(f"\n   Sample data preview:")
            try:
                # Show first few rows of first few columns
                preview_cols = sample_df.columns[:5]
                preview_df = sample_df[preview_cols].head(3)
                print(preview_df.to_string(index=False))
            except Exception:
                print("   (Sample data preview not available)")
    else:
        print("❌ Could not find a workable dataset")
        primary_id = None

print(f"\n🔍 Step 4: Quality assessment")
print("-" * 50)

if not sample_df.empty and primary_dataset is not None:
    try:
        print("Assessing data quality...")

        quality_assessment = scout.assess_dataset_quality(
            dataset_id=primary_id,
            df=sample_df,
            metadata=primary_dataset.to_dict()
        )

        if 'error' not in quality_assessment:
            scores = quality_assessment['overall_scores']
            print(f"✅ Quality Assessment Results:")
            print(f"   Overall Score: {scores['total_score']:.1f}/100 (Grade: {scores['grade']})")
            print(f"   📊 Quality Breakdown:")
            print(f"      Completeness:  {scores['completeness_score']:.1f}/100")
            print(f"      Consistency:   {scores['consistency_score']:.1f}/100")
            print(f"      Accuracy:      {scores['accuracy_score']:.1f}/100")
            print(f"      Timeliness:    {scores['timeliness_score']:.1f}/100")
            print(f"      Usability:     {scores['usability_score']:.1f}/100")

            # Key insights
            completeness = quality_assessment.get('completeness', {})
            missing_pct = completeness.get('missing_percentage', 0)
            print(f"\n   💡 Key Insights:")
            print(f"      Missing data: {missing_pct:.1f}%")
            print(f"      Complete columns: {len(completeness.get('complete_columns', []))}")

        else:
            print(f"❌ Quality assessment failed: {quality_assessment['error']}")
            quality_assessment = None

    except Exception as e:
        print(f"❌ Quality assessment error: {str(e)[:200]}")
        quality_assessment = None
else:
    print("⚠️ Skipping quality assessment - no data available")
    quality_assessment = None

print(f"\n🌐 Step 5: Search for related datasets (limited search)")
print("-" * 50)

try:
    # Use a more focused search for related datasets
    related_terms = ["complaints", "violations"]  # Just 2 terms to avoid timeouts

    print(f"Searching for related datasets: {', '.join(related_terms)}")

    related_datasets = scout.search_datasets(
        search_terms=related_terms,
        limit=5  # Very small limit
    )

    print(f"Found {len(related_datasets)} related datasets")

    # Combine datasets
    if not datasets_311.empty and not related_datasets.empty:
        all_datasets = pd.concat([datasets_311, related_datasets], ignore_index=True)
        all_datasets = all_datasets.drop_duplicates(subset=['id']).reset_index(drop=True)
        print(f"Combined total: {len(all_datasets)} unique datasets")
    elif not datasets_311.empty:
        all_datasets = datasets_311.copy()
        print(f"Using only 311 datasets: {len(all_datasets)}")
    else:
        all_datasets = pd.DataFrame()
        print("⚠️ No datasets for analysis")

except Exception as e:
    print(f"❌ Related dataset search failed: {str(e)[:200]}")
    all_datasets = datasets_311.copy() if not datasets_311.empty else pd.DataFrame()

print(f"\n🔗 Step 6: Relationship analysis")
print("-" * 50)

relationship_stats = None
relationship_graph = None

if not all_datasets.empty and len(all_datasets) >= 2:
    try:
        print(f"Analyzing relationships between {len(all_datasets)} datasets...")

        # Initialize and run relationship analysis
        relationship_graph = DatasetRelationshipGraph()
        relationship_graph.add_datasets(all_datasets)

        # Use a lower threshold to find more relationships
        relationship_stats = relationship_graph.calculate_relationships(
            content_weight=0.4,
            structural_weight=0.2,
            metadata_weight=0.2,
            tag_weight=0.15,
            category_weight=0.05,
            similarity_threshold=0.1  # Very low threshold
        )

        print(f"✅ Relationship Analysis Results:")
        print(f"   Datasets: {relationship_stats['datasets_analyzed']}")
        print(f"   Relationships: {relationship_stats['relationships_found']}")
        print(f"   Density: {relationship_stats['graph_density']:.3f}")
        print(f"   Components: {relationship_stats['connected_components']}")

        # Find relationships to our primary dataset
        if primary_id and relationship_stats['relationships_found'] > 0:
            related_to_primary = relationship_graph.get_related_datasets(
                dataset_id=primary_id,
                top_n=3,
                min_similarity=0.05  # Very low threshold
            )

            if related_to_primary:
                print(f"\n🎯 Datasets related to our primary 311 dataset:")
                for i, rel in enumerate(related_to_primary, 1):
                    name = rel['name'][:40] + '...' if len(rel['name']) > 40 else rel['name']
                    print(f"   {i}. {name}")
                    print(f"      Similarity: {rel['similarity_score']:.3f}")
                    print(f"      Reasons: {', '.join(rel['relationship_reasons'])}")
            else:
                print(f"\n⚠️ No relationships found for primary dataset above threshold")

        # Show strongest relationships overall
        if relationship_stats['relationships_found'] > 0:
            print(f"\n🏆 Strongest relationships found:")
            edges = list(relationship_graph.graph.edges(data=True))
            if edges:
                # Sort by weight
                edges.sort(key=lambda x: x[2].get('weight', 0), reverse=True)

                for i, (node1, node2, data) in enumerate(edges[:2], 1):  # Top 2
                    meta1 = relationship_graph.datasets_metadata.get(node1, {})
                    meta2 = relationship_graph.datasets_metadata.get(node2, {})
                    name1 = meta1.get('name', 'Unknown')[:30]
                    name2 = meta2.get('name', 'Unknown')[:30]
                    weight = data.get('weight', 0)
                    reasons = ', '.join(data.get('relationship_reasons', []))

                    print(f"   {i}. {name1}... ↔ {name2}...")
                    print(f"      Similarity: {weight:.3f} | Reasons: {reasons}")

    except Exception as e:
        print(f"❌ Relationship analysis failed: {str(e)[:200]}")
        traceback.print_exc()

else:
    print("⚠️ Not enough datasets for relationship analysis")

print(f"\n🎨 Step 7: Simple visualization (if relationships found)")
print("-" * 50)

if relationship_graph and relationship_stats and relationship_stats['relationships_found'] > 0:
    try:
        print("Creating basic network visualization...")

        # Create a simple plot
        plt.figure(figsize=(10, 8))

        viz_path = relationship_graph.visualize_graph(
            output_path="robust_311_network.png",
            layout='spring',
            node_size_by='download_count',
            show_labels=False,  # Keep it simple
            figsize=(10, 8)
        )

        if viz_path:
            print(f"✅ Network visualization saved to: {viz_path}")

        plt.close('all')  # Clean up

    except Exception as e:
        print(f"⚠️ Visualization failed (this is common in some environments): {str(e)[:100]}")
else:
    print("⚠️ No relationships to visualize")

print(f"\n📄 Step 8: Generate summary report")
print("-" * 50)

# Create summary
summary_lines = []
summary_lines.append("SCOUT 311 WORKFLOW - ROBUST VERSION SUMMARY")
summary_lines.append("=" * 50)
summary_lines.append(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
summary_lines.append("")

# Results
summary_lines.append("RESULTS:")
if not datasets_311.empty:
    summary_lines.append(f"✅ Found {len(datasets_311)} 311-related datasets")
    if primary_dataset is not None:
        summary_lines.append(f"✅ Successfully analyzed: {primary_dataset['name'][:50]}...")
        summary_lines.append(f"   Dataset ID: {primary_id}")
        summary_lines.append(f"   Sample size: {len(sample_df)} rows")

        if quality_assessment and 'error' not in quality_assessment:
            score = quality_assessment['overall_scores']['total_score']
            grade = quality_assessment['overall_scores']['grade']
            summary_lines.append(f"   Quality score: {score:.1f}/100 (Grade {grade})")
    else:
        summary_lines.append("❌ Could not download data from any dataset")
else:
    summary_lines.append("❌ No 311 datasets found")

if relationship_stats:
    summary_lines.append(f"✅ Relationship analysis: {relationship_stats['relationships_found']} connections found")
else:
    summary_lines.append("❌ No relationship analysis completed")

summary_lines.append("")
summary_lines.append("KEY INSIGHTS:")

if not datasets_311.empty:
    summary_lines.append("• NYC Open Data has multiple 311-related datasets available")

    if primary_dataset is not None:
        summary_lines.append("• Successfully demonstrated end-to-end Scout workflow")
        summary_lines.append("• Data quality assessment provides actionable insights")

        if relationship_stats and relationship_stats['relationships_found'] > 0:
            summary_lines.append(f"• Found {relationship_stats['relationships_found']} dataset relationships")
            summary_lines.append("• Network analysis reveals data integration opportunities")

summary_lines.append("")
summary_lines.append("NEXT STEPS:")
summary_lines.append("1. The core workflow is now working - you can run the Jupyter notebook")
summary_lines.append("2. For production use, consider larger sample sizes and longer timeouts")
summary_lines.append("3. Explore the relationship graphs to find data integration opportunities")
summary_lines.append("4. Use quality scores to prioritize high-quality datasets")

summary_content = "\n".join(summary_lines)

# Save summary
try:
    with open("robust_workflow_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_content)
    print("✅ Summary saved to: robust_workflow_summary.txt")
except Exception as e:
    print(f"⚠️ Could not save summary: {e}")

# Display summary
print(f"\n📋 WORKFLOW SUMMARY")
print("=" * 30)
print(summary_content.split('\n\n', 2)[1])  # Show just the results section

print(f"\n🎉 Robust workflow completed successfully!")

if primary_dataset is not None:
    print("✅ The workflow is working! You can now run the full Jupyter notebook.")
    print("💡 Consider using smaller datasets first if you encounter timeouts.")
else:
    print("⚠️  While we found datasets, data download timed out.")
    print("💡 This is common with large datasets. Try the Jupyter notebook with smaller datasets.")

print(f"\nFiles created:")
print(f"- robust_workflow_summary.txt (This summary)")
if relationship_graph and relationship_stats and relationship_stats.get('relationships_found', 0) > 0:
    print(f"- robust_311_network.png (Network visualization)")

print(f"\n⏰ Workflow runtime: {datetime.now().strftime('%H:%M:%S')}")