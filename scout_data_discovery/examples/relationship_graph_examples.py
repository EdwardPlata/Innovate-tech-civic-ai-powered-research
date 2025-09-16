"""
Dataset Relationship Graph Examples

Demonstrates how to use the Dataset Relationship Graph functionality
to analyze and visualize relationships between datasets.
"""

import sys
from pathlib import Path
import logging
import pandas as pd

# Add the parent directory to Python path to import the package
sys.path.append(str(Path(__file__).parent.parent))

from src.scout_discovery import ScoutDataDiscovery
from src.dataset_relationship_graph import DatasetRelationshipGraph


def example_basic_relationship_analysis():
    """Example: Basic relationship analysis between datasets"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Relationship Analysis")
    print("=" * 60)

    # Initialize Scout and discover datasets
    scout = ScoutDataDiscovery(log_level="INFO")

    try:
        # Search for datasets in related domains
        search_terms = ["health", "transportation", "housing"]
        datasets_df = scout.search_datasets(search_terms, limit=15)

        if datasets_df.empty:
            print("❌ No datasets found for analysis")
            return None

        print(f"\n📋 Found {len(datasets_df)} datasets for relationship analysis")

        # Initialize relationship graph
        relationship_graph = DatasetRelationshipGraph()

        # Add datasets to the graph
        relationship_graph.add_datasets(datasets_df)

        # Calculate relationships
        print("\n🔍 Calculating dataset relationships...")
        stats = relationship_graph.calculate_relationships(
            similarity_threshold=0.2  # Lower threshold to find more relationships
        )

        if stats:
            print(f"\n📊 Relationship Analysis Results:")
            print(f"   Datasets Analyzed: {stats['datasets_analyzed']}")
            print(f"   Relationships Found: {stats['relationships_found']}")
            print(f"   Graph Density: {stats['graph_density']:.3f}")
            print(f"   Connected Components: {stats['connected_components']}")

        # Find related datasets for the first dataset
        if not datasets_df.empty:
            target_dataset = datasets_df.iloc[0]['id']
            related = relationship_graph.get_related_datasets(target_dataset, top_n=5)

            print(f"\n🎯 Datasets related to '{datasets_df.iloc[0]['name']}':")
            for i, rel in enumerate(related, 1):
                print(f"   {i}. {rel['name'][:50]}...")
                print(f"      Similarity: {rel['similarity_score']:.3f}")
                print(f"      Reasons: {', '.join(rel['relationship_reasons'])}")
                print()

        return relationship_graph

    except Exception as e:
        print(f"❌ Relationship analysis failed: {str(e)}")
        return None


def example_comprehensive_graph_analysis():
    """Example: Comprehensive graph analysis with visualizations"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Comprehensive Graph Analysis with Visualizations")
    print("=" * 60)

    scout = ScoutDataDiscovery(log_level="INFO")

    try:
        # Search for a larger dataset collection
        search_terms = ["crime", "safety", "emergency", "fire", "police"]
        datasets_df = scout.search_datasets(search_terms, limit=25)

        if datasets_df.empty:
            print("❌ No datasets found for comprehensive analysis")
            return None

        print(f"\n📋 Analyzing {len(datasets_df)} datasets comprehensively")

        # Initialize relationship graph
        relationship_graph = DatasetRelationshipGraph()
        relationship_graph.add_datasets(datasets_df)

        # Calculate relationships with custom weights
        print("\n🔍 Calculating relationships with custom weights...")
        stats = relationship_graph.calculate_relationships(
            content_weight=0.4,      # Emphasize content similarity
            structural_weight=0.2,   # Moderate structural weight
            metadata_weight=0.2,     # Moderate metadata weight
            tag_weight=0.15,         # Tag similarity
            category_weight=0.05,    # Low category weight
            similarity_threshold=0.25
        )

        print(f"\n📊 Comprehensive Analysis Results:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key.replace('_', ' ').title()}: {value:.3f}")
            else:
                print(f"   {key.replace('_', ' ').title()}: {value}")

        # Create visualizations
        print("\n🎨 Creating visualizations...")

        # Network visualization
        viz_path = Path("exports/relationship_network.png")
        viz_path.parent.mkdir(exist_ok=True)

        network_file = relationship_graph.visualize_graph(
            output_path=str(viz_path),
            layout='spring',
            node_size_by='download_count',
            show_labels=True,
            figsize=(16, 12)
        )

        if network_file:
            print(f"   📊 Network visualization saved to: {network_file}")

        # Interactive visualization
        interactive_path = Path("exports/interactive_network.html")
        interactive_file = relationship_graph.create_interactive_graph(
            output_path=str(interactive_path),
            height=800
        )

        if interactive_file:
            print(f"   🌐 Interactive visualization saved to: {interactive_file}")

        # Generate comprehensive report
        report_path = Path("exports/relationship_report.txt")
        report_content = relationship_graph.generate_relationship_report(
            output_path=str(report_path)
        )

        print(f"   📄 Relationship report saved to: {report_path}")

        # Export graph data
        json_path = Path("exports/graph_data.json")
        relationship_graph.export_graph_data(str(json_path), format='json')
        print(f"   💾 Graph data exported to: {json_path}")

        # Show sample from report
        print(f"\n📋 Sample from Relationship Report:")
        print("-" * 40)
        report_lines = report_content.split('\n')
        for line in report_lines[:15]:  # Show first 15 lines
            print(line)
        print("...")

        return relationship_graph

    except Exception as e:
        print(f"❌ Comprehensive analysis failed: {str(e)}")
        return None


def example_targeted_relationship_discovery():
    """Example: Find relationships for specific datasets of interest"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Targeted Relationship Discovery")
    print("=" * 60)

    scout = ScoutDataDiscovery(log_level="INFO")

    try:
        # Search for datasets in specific domain
        search_terms = ["311", "complaints", "service requests"]
        datasets_df = scout.search_datasets(search_terms, limit=20)

        if len(datasets_df) < 2:
            print("❌ Need at least 2 datasets for relationship analysis")
            return None

        print(f"\n📋 Analyzing relationships in 311/Service Request domain")
        print(f"Found {len(datasets_df)} datasets to analyze")

        # Initialize relationship graph
        relationship_graph = DatasetRelationshipGraph()
        relationship_graph.add_datasets(datasets_df)

        # Calculate relationships with emphasis on structural similarity
        print("\n🔍 Focusing on structural and content similarities...")
        stats = relationship_graph.calculate_relationships(
            content_weight=0.35,
            structural_weight=0.35,  # High structural weight
            metadata_weight=0.15,
            tag_weight=0.1,
            category_weight=0.05,
            similarity_threshold=0.3  # Higher threshold for quality relationships
        )

        print(f"\n📊 Targeted Analysis Results:")
        print(f"   High-quality relationships found: {stats.get('relationships_found', 0)}")
        print(f"   Graph connectivity: {stats.get('graph_density', 0):.3f}")

        # Analyze each dataset's relationships
        print(f"\n🎯 Individual Dataset Relationship Analysis:")
        print("-" * 50)

        for idx, (_, dataset) in enumerate(datasets_df.head(5).iterrows()):
            dataset_id = dataset['id']
            dataset_name = dataset['name'][:60]

            related = relationship_graph.get_related_datasets(
                dataset_id, top_n=3, min_similarity=0.3
            )

            print(f"\n{idx + 1}. {dataset_name}")
            print(f"   Dataset ID: {dataset_id}")

            if related:
                print(f"   Top Related Datasets:")
                for i, rel in enumerate(related):
                    print(f"      {i+1}. {rel['name'][:50]}... (Similarity: {rel['similarity_score']:.3f})")
                    print(f"         Reasons: {', '.join(rel['relationship_reasons'])}")
            else:
                print(f"   No strong relationships found (threshold: 0.3)")

        # Get overall statistics
        graph_stats = relationship_graph.get_statistics()
        print(f"\n📈 Graph Statistics:")
        for key, value in graph_stats.items():
            if isinstance(value, dict):
                print(f"   {key.replace('_', ' ').title()}:")
                for sub_key, sub_value in value.items():
                    print(f"      {sub_key}: {sub_value}")
            elif isinstance(value, float):
                print(f"   {key.replace('_', ' ').title()}: {value:.3f}")
            else:
                print(f"   {key.replace('_', ' ').title()}: {value}")

        return relationship_graph

    except Exception as e:
        print(f"❌ Targeted relationship discovery failed: {str(e)}")
        return None


def example_cross_domain_analysis():
    """Example: Analyze relationships across different data domains"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Cross-Domain Relationship Analysis")
    print("=" * 60)

    scout = ScoutDataDiscovery(log_level="INFO")

    try:
        # Search across multiple domains
        domains = {
            "environment": ["air quality", "pollution", "environmental"],
            "health": ["health", "disease", "medical"],
            "demographics": ["population", "census", "demographics"],
            "economics": ["business", "economic", "employment"]
        }

        all_datasets = []

        for domain_name, search_terms in domains.items():
            print(f"\n🔍 Searching {domain_name} domain...")
            domain_datasets = scout.search_datasets(search_terms, limit=8)

            if not domain_datasets.empty:
                # Add domain label
                domain_datasets['analysis_domain'] = domain_name
                all_datasets.append(domain_datasets)
                print(f"   Found {len(domain_datasets)} datasets")

        if not all_datasets:
            print("❌ No datasets found across domains")
            return None

        # Combine all datasets
        combined_df = pd.concat(all_datasets, ignore_index=True).drop_duplicates(subset=['id'])
        print(f"\n📋 Total datasets for cross-domain analysis: {len(combined_df)}")

        # Initialize relationship graph
        relationship_graph = DatasetRelationshipGraph()
        relationship_graph.add_datasets(combined_df)

        # Calculate relationships with balanced weights
        print("\n🔗 Calculating cross-domain relationships...")
        stats = relationship_graph.calculate_relationships(
            content_weight=0.3,
            structural_weight=0.3,
            metadata_weight=0.2,
            tag_weight=0.15,
            category_weight=0.05,
            similarity_threshold=0.25
        )

        print(f"\n📊 Cross-Domain Analysis Results:")
        print(f"   Total Relationships: {stats.get('relationships_found', 0)}")
        print(f"   Cross-domain connections: {stats.get('connected_components', 1) - 1}")

        # Analyze cross-domain relationships
        print(f"\n🌐 Cross-Domain Relationship Patterns:")
        print("-" * 40)

        domain_relationships = defaultdict(lambda: defaultdict(int))

        # Count relationships between domains
        for edge in relationship_graph.graph.edges(data=True):
            node1, node2, data = edge
            domain1 = relationship_graph.datasets_metadata[node1].get('analysis_domain', 'unknown')
            domain2 = relationship_graph.datasets_metadata[node2].get('analysis_domain', 'unknown')

            if domain1 != domain2:  # Cross-domain relationship
                domain_relationships[domain1][domain2] += 1
                domain_relationships[domain2][domain1] += 1

        # Display cross-domain connections
        for domain1, connections in domain_relationships.items():
            total_cross_connections = sum(connections.values())
            if total_cross_connections > 0:
                print(f"\n{domain1.title()} domain connections:")
                for domain2, count in connections.items():
                    if count > 0:
                        print(f"   ↔ {domain2.title()}: {count} relationships")

        # Find most connected datasets across domains
        print(f"\n🏆 Most Connected Datasets (Cross-Domain):")
        print("-" * 45)

        cross_domain_scores = {}
        for dataset_id in relationship_graph.graph.nodes():
            dataset_domain = relationship_graph.datasets_metadata[dataset_id].get('analysis_domain', 'unknown')
            cross_domain_count = 0

            for neighbor in relationship_graph.graph.neighbors(dataset_id):
                neighbor_domain = relationship_graph.datasets_metadata[neighbor].get('analysis_domain', 'unknown')
                if neighbor_domain != dataset_domain:
                    cross_domain_count += 1

            if cross_domain_count > 0:
                cross_domain_scores[dataset_id] = cross_domain_count

        # Show top cross-domain connectors
        sorted_cross_domain = sorted(cross_domain_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (dataset_id, score) in enumerate(sorted_cross_domain[:5]):
            meta = relationship_graph.datasets_metadata[dataset_id]
            name = meta.get('name', 'Unknown')[:50]
            domain = meta.get('analysis_domain', 'unknown')

            print(f"{i+1}. {name}...")
            print(f"   Domain: {domain.title()}")
            print(f"   Cross-domain connections: {score}")
            print()

        return relationship_graph

    except Exception as e:
        print(f"❌ Cross-domain analysis failed: {str(e)}")
        return None


def run_all_relationship_examples():
    """Run all relationship graph examples"""
    print("🚀 Running All Dataset Relationship Graph Examples")
    print("=" * 80)

    # Set up logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce log noise for examples
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create exports directory
    exports_dir = Path("exports")
    exports_dir.mkdir(exist_ok=True)

    # Run examples
    try:
        graph1 = example_basic_relationship_analysis()
        graph2 = example_comprehensive_graph_analysis()
        graph3 = example_targeted_relationship_discovery()
        graph4 = example_cross_domain_analysis()

        print("\n" + "=" * 80)
        print("✅ All relationship graph examples completed successfully!")
        print("=" * 80)

        # Summary
        print("\n📋 Summary of Examples:")
        print("   1. ✓ Basic relationship analysis between datasets")
        print("   2. ✓ Comprehensive analysis with visualizations")
        print("   3. ✓ Targeted relationship discovery in specific domain")
        print("   4. ✓ Cross-domain relationship analysis")

        # Show files created
        if exports_dir.exists():
            export_files = list(exports_dir.glob("*"))
            if export_files:
                print(f"\n📁 Files created in exports/:")
                for file in export_files:
                    file_size = file.stat().st_size / 1024  # KB
                    print(f"   - {file.name} ({file_size:.1f} KB)")

        print(f"\n🎯 Key Insights:")
        print("   • Dataset relationships can be quantified using multiple similarity metrics")
        print("   • Network analysis reveals clusters and connection patterns")
        print("   • Cross-domain relationships help identify unexpected data connections")
        print("   • Interactive visualizations make complex relationships explorable")

        return [graph1, graph2, graph3, graph4]

    except Exception as e:
        print(f"\n❌ Examples failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    try:
        from collections import defaultdict  # Import needed for cross-domain example
        run_all_relationship_examples()
    except KeyboardInterrupt:
        print("\n\n⏹️ Examples interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Examples failed with error: {str(e)}")
        import traceback
        traceback.print_exc()