"""
Dataset Relationship Graph Module

Creates visual and analytical graphs showing relationships between datasets
based on content similarity, shared metadata, and data structures.
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import json
import logging
from pathlib import Path
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False


class DatasetRelationshipGraph:
    """
    Analyzes and visualizes relationships between datasets based on:
    - Content similarity (tags, categories, descriptions)
    - Structural similarity (column names, data types)
    - Metadata similarity (update patterns, usage patterns)
    - Semantic relationships (topic modeling)
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the relationship graph analyzer.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.graph = nx.Graph()
        self.datasets_metadata = {}
        self.similarity_matrices = {}
        self.clusters = {}
        self.relationship_types = {
            'content_similarity': 'Content Similar',
            'structural_similarity': 'Structure Similar',
            'metadata_similarity': 'Metadata Similar',
            'tag_overlap': 'Shared Tags',
            'category_match': 'Same Category',
            'column_overlap': 'Similar Columns'
        }

    def add_datasets(self, datasets_df: pd.DataFrame):
        """
        Add datasets to the relationship analysis.

        Args:
            datasets_df: DataFrame containing dataset metadata
        """
        try:
            self.logger.info(f"Adding {len(datasets_df)} datasets to relationship graph")

            # Store metadata
            for _, row in datasets_df.iterrows():
                dataset_id = row['id']
                self.datasets_metadata[dataset_id] = row.to_dict()

                # Add node to graph with attributes
                self.graph.add_node(dataset_id, **row.to_dict())

            self.logger.info(f"Graph now contains {self.graph.number_of_nodes()} datasets")

        except Exception as e:
            self.logger.error(f"Failed to add datasets: {str(e)}")
            raise

    def calculate_relationships(self,
                              content_weight: float = 0.3,
                              structural_weight: float = 0.25,
                              metadata_weight: float = 0.2,
                              tag_weight: float = 0.15,
                              category_weight: float = 0.1,
                              similarity_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Calculate relationships between all datasets.

        Args:
            content_weight: Weight for content similarity
            structural_weight: Weight for structural similarity
            metadata_weight: Weight for metadata similarity
            tag_weight: Weight for tag overlap
            category_weight: Weight for category matching
            similarity_threshold: Minimum similarity to create edge

        Returns:
            Dictionary with relationship statistics
        """
        try:
            if len(self.datasets_metadata) < 2:
                self.logger.warning("Need at least 2 datasets to calculate relationships")
                return {}

            self.logger.info("Calculating dataset relationships...")

            dataset_ids = list(self.datasets_metadata.keys())
            n_datasets = len(dataset_ids)

            # Initialize similarity matrices
            similarity_types = ['content', 'structural', 'metadata', 'tag', 'category']
            for sim_type in similarity_types:
                self.similarity_matrices[sim_type] = np.zeros((n_datasets, n_datasets))

            # Calculate pairwise similarities
            relationships_added = 0

            for i, dataset_id1 in enumerate(dataset_ids):
                for j, dataset_id2 in enumerate(dataset_ids[i+1:], i+1):

                    # Get metadata for both datasets
                    meta1 = self.datasets_metadata[dataset_id1]
                    meta2 = self.datasets_metadata[dataset_id2]

                    # Calculate different similarity types
                    similarities = {}

                    # Content similarity (descriptions, names)
                    similarities['content'] = self._calculate_content_similarity(meta1, meta2)

                    # Structural similarity (columns, data types)
                    similarities['structural'] = self._calculate_structural_similarity(meta1, meta2)

                    # Metadata similarity (update patterns, usage)
                    similarities['metadata'] = self._calculate_metadata_similarity(meta1, meta2)

                    # Tag overlap
                    similarities['tag'] = self._calculate_tag_overlap(meta1, meta2)

                    # Category matching
                    similarities['category'] = self._calculate_category_similarity(meta1, meta2)

                    # Store in matrices
                    for sim_type, score in similarities.items():
                        self.similarity_matrices[sim_type][i, j] = score
                        self.similarity_matrices[sim_type][j, i] = score

                    # Calculate composite similarity
                    composite_score = (
                        similarities['content'] * content_weight +
                        similarities['structural'] * structural_weight +
                        similarities['metadata'] * metadata_weight +
                        similarities['tag'] * tag_weight +
                        similarities['category'] * category_weight
                    )

                    # Add edge if above threshold
                    if composite_score >= similarity_threshold:
                        edge_data = {
                            'weight': composite_score,
                            'similarity_breakdown': similarities,
                            'relationship_reasons': self._generate_relationship_reasons(similarities)
                        }

                        self.graph.add_edge(dataset_id1, dataset_id2, **edge_data)
                        relationships_added += 1

            # Calculate clusters
            self._calculate_clusters()

            stats = {
                'datasets_analyzed': n_datasets,
                'relationships_found': relationships_added,
                'graph_density': nx.density(self.graph),
                'connected_components': nx.number_connected_components(self.graph),
                'average_clustering': nx.average_clustering(self.graph),
                'similarity_threshold': similarity_threshold
            }

            self.logger.info(f"Found {relationships_added} relationships between {n_datasets} datasets")

            return stats

        except Exception as e:
            self.logger.error(f"Failed to calculate relationships: {str(e)}")
            raise

    def _calculate_content_similarity(self, meta1: Dict, meta2: Dict) -> float:
        """Calculate similarity based on textual content (descriptions, names)"""
        try:
            # Combine text fields
            text1 = ' '.join([
                str(meta1.get('name', '')),
                str(meta1.get('description', '')),
                str(meta1.get('attribution', ''))
            ]).lower()

            text2 = ' '.join([
                str(meta2.get('name', '')),
                str(meta2.get('description', '')),
                str(meta2.get('attribution', ''))
            ]).lower()

            if not text1.strip() or not text2.strip():
                return 0.0

            # Use TF-IDF for semantic similarity
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            try:
                tfidf_matrix = vectorizer.fit_transform([text1, text2])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                return float(similarity)
            except:
                # Fallback to simple word overlap
                words1 = set(text1.split())
                words2 = set(text2.split())
                if not words1 or not words2:
                    return 0.0
                overlap = len(words1.intersection(words2))
                union = len(words1.union(words2))
                return overlap / union if union > 0 else 0.0

        except Exception as e:
            self.logger.debug(f"Content similarity calculation failed: {str(e)}")
            return 0.0

    def _calculate_structural_similarity(self, meta1: Dict, meta2: Dict) -> float:
        """Calculate similarity based on data structure (columns, types)"""
        try:
            # Get column information
            cols1 = set(meta1.get('columns_names', []) or meta1.get('columns', []))
            cols2 = set(meta2.get('columns_names', []) or meta2.get('columns', []))

            if not cols1 or not cols2:
                return 0.0

            # Jaccard similarity for column names
            intersection = len(cols1.intersection(cols2))
            union = len(cols1.union(cols2))
            jaccard = intersection / union if union > 0 else 0.0

            # Consider data types if available
            types1 = set(meta1.get('columns_datatypes', []))
            types2 = set(meta2.get('columns_datatypes', []))

            type_similarity = 0.0
            if types1 and types2:
                type_intersection = len(types1.intersection(types2))
                type_union = len(types1.union(types2))
                type_similarity = type_intersection / type_union if type_union > 0 else 0.0

            # Combine column and type similarities
            return (jaccard * 0.7 + type_similarity * 0.3)

        except Exception as e:
            self.logger.debug(f"Structural similarity calculation failed: {str(e)}")
            return 0.0

    def _calculate_metadata_similarity(self, meta1: Dict, meta2: Dict) -> float:
        """Calculate similarity based on metadata patterns"""
        try:
            similarity_score = 0.0
            components = 0

            # Update frequency similarity
            try:
                update1 = pd.to_datetime(meta1.get('updatedAt', meta1.get('updated_at')))
                update2 = pd.to_datetime(meta2.get('updatedAt', meta2.get('updated_at')))

                if pd.notna(update1) and pd.notna(update2):
                    # Similar update times suggest related maintenance
                    days_diff = abs((update1 - update2).days)
                    update_similarity = max(0, 1 - days_diff / 365)  # Normalize by year
                    similarity_score += update_similarity
                    components += 1
            except:
                pass

            # Usage pattern similarity (downloads, views)
            download1 = meta1.get('download_count', 0) or 0
            download2 = meta2.get('download_count', 0) or 0
            views1 = meta1.get('page_views_total', 0) or meta1.get('page_views', 0) or 0
            views2 = meta2.get('page_views_total', 0) or meta2.get('page_views', 0) or 0

            if download1 > 0 and download2 > 0:
                # Log-scale similarity for download counts
                download_sim = 1 - abs(np.log10(download1 + 1) - np.log10(download2 + 1)) / 10
                similarity_score += max(0, download_sim)
                components += 1

            if views1 > 0 and views2 > 0:
                # Log-scale similarity for view counts
                view_sim = 1 - abs(np.log10(views1 + 1) - np.log10(views2 + 1)) / 10
                similarity_score += max(0, view_sim)
                components += 1

            # Size similarity (column count)
            cols1 = meta1.get('columns_count', len(meta1.get('columns_names', [])))
            cols2 = meta2.get('columns_count', len(meta2.get('columns_names', [])))

            if cols1 > 0 and cols2 > 0:
                size_sim = 1 - abs(cols1 - cols2) / max(cols1, cols2)
                similarity_score += size_sim
                components += 1

            return similarity_score / components if components > 0 else 0.0

        except Exception as e:
            self.logger.debug(f"Metadata similarity calculation failed: {str(e)}")
            return 0.0

    def _calculate_tag_overlap(self, meta1: Dict, meta2: Dict) -> float:
        """Calculate similarity based on tag overlap"""
        try:
            # Get all tags
            tags1 = set()
            tags2 = set()

            # From various tag fields
            for tag_field in ['tags', 'domain_tags', 'categories']:
                if tag_field in meta1:
                    tag_data = meta1[tag_field]
                    if isinstance(tag_data, list):
                        tags1.update([str(tag).lower() for tag in tag_data])
                    elif isinstance(tag_data, str):
                        tags1.add(tag_data.lower())

                if tag_field in meta2:
                    tag_data = meta2[tag_field]
                    if isinstance(tag_data, list):
                        tags2.update([str(tag).lower() for tag in tag_data])
                    elif isinstance(tag_data, str):
                        tags2.add(tag_data.lower())

            if not tags1 or not tags2:
                return 0.0

            # Jaccard similarity
            intersection = len(tags1.intersection(tags2))
            union = len(tags1.union(tags2))

            return intersection / union if union > 0 else 0.0

        except Exception as e:
            self.logger.debug(f"Tag overlap calculation failed: {str(e)}")
            return 0.0

    def _calculate_category_similarity(self, meta1: Dict, meta2: Dict) -> float:
        """Calculate similarity based on category matching"""
        try:
            cat1 = str(meta1.get('domain_category', meta1.get('category', ''))).lower().strip()
            cat2 = str(meta2.get('domain_category', meta2.get('category', ''))).lower().strip()

            if not cat1 or not cat2:
                return 0.0

            # Exact match
            if cat1 == cat2:
                return 1.0

            # Partial match (for hierarchical categories)
            if cat1 in cat2 or cat2 in cat1:
                return 0.5

            return 0.0

        except Exception as e:
            self.logger.debug(f"Category similarity calculation failed: {str(e)}")
            return 0.0

    def _generate_relationship_reasons(self, similarities: Dict[str, float]) -> List[str]:
        """Generate human-readable reasons for the relationship"""
        reasons = []

        for sim_type, score in similarities.items():
            if score > 0.5:
                if sim_type == 'content':
                    reasons.append("Similar content and descriptions")
                elif sim_type == 'structural':
                    reasons.append("Similar data structure and columns")
                elif sim_type == 'metadata':
                    reasons.append("Similar usage patterns and metadata")
                elif sim_type == 'tag':
                    reasons.append("Shared tags and categories")
                elif sim_type == 'category':
                    reasons.append("Same domain category")

        return reasons if reasons else ["General similarity"]

    def _calculate_clusters(self, n_clusters: Optional[int] = None):
        """Calculate dataset clusters based on similarity"""
        try:
            if len(self.datasets_metadata) < 3:
                self.logger.info("Too few datasets for clustering")
                return

            # Create feature matrix for clustering
            dataset_ids = list(self.datasets_metadata.keys())
            n_datasets = len(dataset_ids)

            # Use average similarity as features
            features = []
            for i, dataset_id in enumerate(dataset_ids):
                feature_vector = []
                for sim_type in ['content', 'structural', 'metadata', 'tag', 'category']:
                    if sim_type in self.similarity_matrices:
                        avg_similarity = np.mean(self.similarity_matrices[sim_type][i, :])
                        feature_vector.append(avg_similarity)
                    else:
                        feature_vector.append(0.0)
                features.append(feature_vector)

            features_array = np.array(features)

            # Determine number of clusters
            if n_clusters is None:
                n_clusters = min(max(2, n_datasets // 3), 8)

            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_array)

            # Store cluster assignments
            self.clusters = {}
            for i, dataset_id in enumerate(dataset_ids):
                cluster_id = int(cluster_labels[i])
                if cluster_id not in self.clusters:
                    self.clusters[cluster_id] = []
                self.clusters[cluster_id].append(dataset_id)

                # Add cluster info to graph node
                self.graph.nodes[dataset_id]['cluster'] = cluster_id

            self.logger.info(f"Identified {len(self.clusters)} dataset clusters")

        except Exception as e:
            self.logger.error(f"Clustering failed: {str(e)}")

    def get_related_datasets(self,
                           dataset_id: str,
                           top_n: int = 5,
                           min_similarity: float = 0.3) -> List[Dict[str, Any]]:
        """
        Get datasets most related to the specified dataset.

        Args:
            dataset_id: Target dataset ID
            top_n: Number of top related datasets to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of related datasets with similarity scores
        """
        try:
            if dataset_id not in self.graph:
                self.logger.warning(f"Dataset {dataset_id} not found in graph")
                return []

            # Get neighbors from graph
            neighbors = list(self.graph.neighbors(dataset_id))

            if not neighbors:
                return []

            # Get edge data and sort by similarity
            related = []
            for neighbor_id in neighbors:
                edge_data = self.graph.edges[dataset_id, neighbor_id]
                similarity = edge_data.get('weight', 0.0)

                if similarity >= min_similarity:
                    neighbor_meta = self.datasets_metadata.get(neighbor_id, {})
                    related.append({
                        'dataset_id': neighbor_id,
                        'name': neighbor_meta.get('name', 'Unknown'),
                        'similarity_score': similarity,
                        'relationship_reasons': edge_data.get('relationship_reasons', []),
                        'similarity_breakdown': edge_data.get('similarity_breakdown', {}),
                        'category': neighbor_meta.get('domain_category', neighbor_meta.get('category', ''))
                    })

            # Sort by similarity and return top N
            related.sort(key=lambda x: x['similarity_score'], reverse=True)
            return related[:top_n]

        except Exception as e:
            self.logger.error(f"Failed to get related datasets for {dataset_id}: {str(e)}")
            return []

    def visualize_graph(self,
                       output_path: Optional[str] = None,
                       layout: str = 'spring',
                       node_size_by: str = 'download_count',
                       show_labels: bool = True,
                       figsize: Tuple[int, int] = (15, 12)) -> Optional[str]:
        """
        Create a network visualization of dataset relationships.

        Args:
            output_path: Path to save the visualization
            layout: Network layout algorithm ('spring', 'kamada_kawai', 'circular')
            node_size_by: Attribute to determine node sizes
            show_labels: Whether to show dataset names as labels
            figsize: Figure size (width, height)

        Returns:
            Path to saved visualization or None
        """
        try:
            if self.graph.number_of_nodes() == 0:
                self.logger.warning("No datasets in graph to visualize")
                return None

            fig, ax = plt.subplots(1, 1, figsize=figsize)

            # Choose layout
            if layout == 'spring':
                pos = nx.spring_layout(self.graph, k=1, iterations=50)
            elif layout == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(self.graph)
            elif layout == 'circular':
                pos = nx.circular_layout(self.graph)
            else:
                pos = nx.spring_layout(self.graph)

            # Calculate node sizes
            node_sizes = []
            for node in self.graph.nodes():
                node_data = self.graph.nodes[node]
                size_value = node_data.get(node_size_by, 1)
                # Normalize size (min 100, max 2000)
                if isinstance(size_value, (int, float)) and size_value > 0:
                    normalized_size = min(2000, max(100, size_value / 10))
                else:
                    normalized_size = 300
                node_sizes.append(normalized_size)

            # Color nodes by cluster if available
            node_colors = []
            if self.clusters:
                color_map = plt.cm.Set3(np.linspace(0, 1, len(self.clusters)))
                for node in self.graph.nodes():
                    cluster_id = self.graph.nodes[node].get('cluster', 0)
                    node_colors.append(color_map[cluster_id % len(color_map)])
            else:
                node_colors = ['lightblue'] * len(self.graph.nodes())

            # Draw edges with varying thickness based on similarity
            edges = self.graph.edges(data=True)
            edge_weights = [edge[2].get('weight', 0.5) * 5 for edge in edges]

            nx.draw_networkx_edges(self.graph, pos, width=edge_weights,
                                 alpha=0.6, edge_color='gray', ax=ax)

            # Draw nodes
            nx.draw_networkx_nodes(self.graph, pos, node_size=node_sizes,
                                 node_color=node_colors, alpha=0.8, ax=ax)

            # Add labels if requested
            if show_labels:
                labels = {}
                for node in self.graph.nodes():
                    node_data = self.graph.nodes[node]
                    name = node_data.get('name', node)
                    # Truncate long names
                    if len(name) > 30:
                        name = name[:27] + '...'
                    labels[node] = name

                nx.draw_networkx_labels(self.graph, pos, labels, font_size=8, ax=ax)

            # Set title and styling
            ax.set_title('Dataset Relationship Network', fontsize=16, fontweight='bold')
            ax.axis('off')

            # Add legend for clusters if available
            if self.clusters:
                cluster_labels = [f'Cluster {i+1} ({len(datasets)} datasets)'
                                for i, datasets in self.clusters.items()]
                ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=color_map[i % len(color_map)],
                                            markersize=10) for i in range(len(self.clusters))],
                         labels=cluster_labels, loc='upper right', bbox_to_anchor=(1, 1))

            plt.tight_layout()

            # Save if path provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Graph visualization saved to {output_path}")

                plt.close()
                return output_path
            else:
                plt.show()
                return None

        except Exception as e:
            self.logger.error(f"Failed to create graph visualization: {str(e)}")
            return None

    def create_interactive_graph(self,
                               output_path: Optional[str] = None,
                               height: int = 800) -> Optional[str]:
        """
        Create an interactive network visualization using Plotly.

        Args:
            output_path: Path to save the HTML file
            height: Height of the visualization

        Returns:
            Path to saved HTML file or None
        """
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available for interactive graphs")
            return None

        try:
            if self.graph.number_of_nodes() == 0:
                self.logger.warning("No datasets in graph to visualize")
                return None

            # Create layout
            pos = nx.spring_layout(self.graph, k=1, iterations=50)

            # Prepare edge traces
            edge_x = []
            edge_y = []
            edge_info = []

            for edge in self.graph.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

                similarity = edge[2].get('weight', 0)
                reasons = ', '.join(edge[2].get('relationship_reasons', []))
                edge_info.append(f"Similarity: {similarity:.3f}<br>Reasons: {reasons}")

            # Create edge trace
            edge_trace = go.Scatter(x=edge_x, y=edge_y,
                                  line=dict(width=2, color='#888'),
                                  hoverinfo='none',
                                  mode='lines')

            # Prepare node traces
            node_x = []
            node_y = []
            node_text = []
            node_info = []
            node_colors = []
            node_sizes = []

            for node in self.graph.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

                node_data = self.graph.nodes[node]
                name = node_data.get('name', 'Unknown')
                category = node_data.get('domain_category', node_data.get('category', 'Uncategorized'))
                download_count = node_data.get('download_count', 0)

                node_text.append(name[:50] + '...' if len(name) > 50 else name)

                hover_info = f"<b>{name}</b><br>"
                hover_info += f"ID: {node}<br>"
                hover_info += f"Category: {category}<br>"
                hover_info += f"Downloads: {download_count:,}<br>"
                hover_info += f"Connections: {self.graph.degree(node)}"
                node_info.append(hover_info)

                # Color by cluster
                cluster_id = node_data.get('cluster', 0)
                node_colors.append(cluster_id)

                # Size by download count
                size = min(50, max(10, np.log10(download_count + 1) * 5))
                node_sizes.append(size)

            # Create node trace
            node_trace = go.Scatter(x=node_x, y=node_y,
                                  mode='markers+text',
                                  hoverinfo='text',
                                  text=node_text,
                                  textposition="middle center",
                                  hovertext=node_info,
                                  marker=dict(
                                      size=node_sizes,
                                      color=node_colors,
                                      colorscale='Viridis',
                                      showscale=True,
                                      colorbar=dict(
                                          title="Cluster",
                                          titleside="right",
                                          tickmode="linear"
                                      ),
                                      line=dict(width=2, color='white')
                                  ))

            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title={
                                  'text': 'Interactive Dataset Relationship Network',
                                  'font': {'size': 16}
                              },
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=40),
                              annotations=[ dict(
                                  text="Hover over nodes for details. Node size = downloads, color = cluster",
                                  showarrow=False,
                                  xref="paper", yref="paper",
                                  x=0.005, y=-0.002,
                                  xanchor='left', yanchor='bottom',
                                  font=dict(size=12, color="gray")
                              )],
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              height=height))

            # Save if path provided
            if output_path:
                fig.write_html(output_path)
                self.logger.info(f"Interactive graph saved to {output_path}")
                return output_path
            else:
                fig.show()
                return None

        except Exception as e:
            self.logger.error(f"Failed to create interactive graph: {str(e)}")
            return None

    def generate_relationship_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive report of dataset relationships.

        Args:
            output_path: Path to save the report

        Returns:
            Report content as string
        """
        try:
            report_lines = []
            report_lines.append("DATASET RELATIONSHIP ANALYSIS REPORT")
            report_lines.append("=" * 50)
            report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")

            # Overall statistics
            n_datasets = len(self.datasets_metadata)
            n_relationships = self.graph.number_of_edges()

            report_lines.append("OVERVIEW")
            report_lines.append("-" * 20)
            report_lines.append(f"Total Datasets: {n_datasets}")
            report_lines.append(f"Total Relationships: {n_relationships}")
            report_lines.append(f"Graph Density: {nx.density(self.graph):.3f}")
            report_lines.append(f"Connected Components: {nx.number_connected_components(self.graph)}")

            if n_relationships > 0:
                report_lines.append(f"Average Clustering Coefficient: {nx.average_clustering(self.graph):.3f}")

            report_lines.append("")

            # Cluster analysis
            if self.clusters:
                report_lines.append("CLUSTER ANALYSIS")
                report_lines.append("-" * 20)

                for cluster_id, dataset_ids in self.clusters.items():
                    report_lines.append(f"\nCluster {cluster_id + 1}: {len(dataset_ids)} datasets")

                    # Get representative datasets
                    for i, dataset_id in enumerate(dataset_ids[:3]):  # Show first 3
                        meta = self.datasets_metadata.get(dataset_id, {})
                        name = meta.get('name', 'Unknown')[:60]
                        category = meta.get('domain_category', meta.get('category', 'Unknown'))
                        report_lines.append(f"  - {name} ({category})")

                    if len(dataset_ids) > 3:
                        report_lines.append(f"  ... and {len(dataset_ids) - 3} more")

                report_lines.append("")

            # Top relationships
            if n_relationships > 0:
                report_lines.append("STRONGEST RELATIONSHIPS")
                report_lines.append("-" * 30)

                # Get edges sorted by weight
                edges_with_weight = [(u, v, data['weight'], data)
                                   for u, v, data in self.graph.edges(data=True)]
                edges_with_weight.sort(key=lambda x: x[2], reverse=True)

                for i, (dataset1, dataset2, weight, data) in enumerate(edges_with_weight[:10]):
                    meta1 = self.datasets_metadata.get(dataset1, {})
                    meta2 = self.datasets_metadata.get(dataset2, {})

                    name1 = meta1.get('name', 'Unknown')[:40]
                    name2 = meta2.get('name', 'Unknown')[:40]
                    reasons = ', '.join(data.get('relationship_reasons', []))

                    report_lines.append(f"{i+1}. {name1} ↔ {name2}")
                    report_lines.append(f"   Similarity: {weight:.3f} | Reasons: {reasons}")
                    report_lines.append("")

            # Category analysis
            category_counts = defaultdict(int)
            for meta in self.datasets_metadata.values():
                category = meta.get('domain_category', meta.get('category', 'Unknown'))
                category_counts[category] += 1

            if category_counts:
                report_lines.append("CATEGORY DISTRIBUTION")
                report_lines.append("-" * 25)

                sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
                for category, count in sorted_categories[:10]:
                    report_lines.append(f"{category}: {count} datasets")

                report_lines.append("")

            # Most connected datasets
            if n_relationships > 0:
                report_lines.append("MOST CONNECTED DATASETS")
                report_lines.append("-" * 30)

                degree_centrality = nx.degree_centrality(self.graph)
                sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)

                for i, (dataset_id, centrality) in enumerate(sorted_nodes[:10]):
                    meta = self.datasets_metadata.get(dataset_id, {})
                    name = meta.get('name', 'Unknown')[:50]
                    connections = self.graph.degree(dataset_id)

                    report_lines.append(f"{i+1}. {name}")
                    report_lines.append(f"   Connections: {connections} | Centrality: {centrality:.3f}")
                    report_lines.append("")

            report_content = '\n'.join(report_lines)

            # Save to file if path provided
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                self.logger.info(f"Relationship report saved to {output_path}")

            return report_content

        except Exception as e:
            self.logger.error(f"Failed to generate relationship report: {str(e)}")
            return f"Error generating report: {str(e)}"

    def export_graph_data(self, output_path: str, format: str = 'json') -> str:
        """
        Export graph data in various formats.

        Args:
            output_path: Path to save the exported data
            format: Export format ('json', 'gml', 'graphml', 'csv')

        Returns:
            Path to exported file
        """
        try:
            output_path = Path(output_path)

            if format.lower() == 'json':
                # Export as JSON with detailed data
                graph_data = {
                    'nodes': [
                        {
                            'id': node,
                            'attributes': data
                        }
                        for node, data in self.graph.nodes(data=True)
                    ],
                    'edges': [
                        {
                            'source': u,
                            'target': v,
                            'attributes': data
                        }
                        for u, v, data in self.graph.edges(data=True)
                    ],
                    'clusters': self.clusters,
                    'metadata': {
                        'generated_at': datetime.now().isoformat(),
                        'total_nodes': self.graph.number_of_nodes(),
                        'total_edges': self.graph.number_of_edges()
                    }
                }

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(graph_data, f, indent=2, default=str)

            elif format.lower() == 'gml':
                nx.write_gml(self.graph, output_path)

            elif format.lower() == 'graphml':
                nx.write_graphml(self.graph, output_path)

            elif format.lower() == 'csv':
                # Export edges as CSV
                edges_df = pd.DataFrame([
                    {
                        'source': u,
                        'target': v,
                        'weight': data.get('weight', 0),
                        'reasons': '; '.join(data.get('relationship_reasons', []))
                    }
                    for u, v, data in self.graph.edges(data=True)
                ])
                edges_df.to_csv(output_path, index=False)

                # Also save nodes
                nodes_path = output_path.parent / (output_path.stem + '_nodes.csv')
                nodes_df = pd.DataFrame([
                    {
                        'id': node,
                        'name': data.get('name', ''),
                        'category': data.get('domain_category', data.get('category', '')),
                        'cluster': data.get('cluster', -1),
                        'download_count': data.get('download_count', 0)
                    }
                    for node, data in self.graph.nodes(data=True)
                ])
                nodes_df.to_csv(nodes_path, index=False)

            else:
                raise ValueError(f"Unsupported export format: {format}")

            self.logger.info(f"Graph data exported to {output_path} in {format} format")
            return str(output_path)

        except Exception as e:
            self.logger.error(f"Failed to export graph data: {str(e)}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the relationship graph"""
        try:
            stats = {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph),
                'connected_components': nx.number_connected_components(self.graph),
                'clusters': len(self.clusters) if self.clusters else 0
            }

            if self.graph.number_of_edges() > 0:
                stats['average_clustering'] = nx.average_clustering(self.graph)
                stats['average_degree'] = sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()

                # Centrality measures for largest component
                largest_cc = max(nx.connected_components(self.graph), key=len)
                if len(largest_cc) > 1:
                    subgraph = self.graph.subgraph(largest_cc)
                    centralities = nx.degree_centrality(subgraph)
                    stats['max_degree_centrality'] = max(centralities.values())
                    stats['avg_degree_centrality'] = sum(centralities.values()) / len(centralities)

            if self.similarity_matrices:
                stats['similarity_matrices'] = {
                    sim_type: {
                        'mean': float(np.mean(matrix[np.nonzero(matrix)])) if np.any(matrix) else 0.0,
                        'max': float(np.max(matrix)),
                        'std': float(np.std(matrix[np.nonzero(matrix)])) if np.any(matrix) else 0.0
                    }
                    for sim_type, matrix in self.similarity_matrices.items()
                }

            return stats

        except Exception as e:
            self.logger.error(f"Failed to calculate statistics: {str(e)}")
            return {}