"""
NYC Homelessness Analytics Workflow

A comprehensive workflow for analyzing homelessness data from NYC Open Data
using Scout data discovery, NVIDIA AI analysis, and interactive visualizations.

Features:
- Automated discovery of homelessness-related datasets
- AI-powered analysis using NVIDIA models
- Interactive analytics and visualizations with pandas/plotly
- Scout integration for data quality assessment
- Relationship mapping between different datasets
"""

import asyncio
import os
import sys
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
from pathlib import Path

# Add project paths
current_dir = Path(__file__).parent
sys.path.extend([
    str(current_dir / "scout_data_discovery"),
])

from scout_data_discovery.src.scout_discovery import ScoutDataDiscovery
from scout_data_discovery.src.dataset_relationship_graph import DatasetRelationshipGraph

# Simple NVIDIA AI provider
class SimpleNvidiaProvider:
    """Simplified NVIDIA AI provider for this workflow"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://integrate.api.nvidia.com/v1"
        
    async def analyze_dataset(self, dataset_info: Dict, sample_df: pd.DataFrame) -> Dict:
        """Analyze dataset using NVIDIA API"""
        try:
            import httpx
            
            # Prepare dataset context
            context = f"""
Dataset Name: {dataset_info.get('name', 'Unknown')}
Description: {dataset_info.get('description', 'No description')}
Category: {dataset_info.get('category', 'Unknown')}
Rows: {len(sample_df)}
Columns: {list(sample_df.columns)}

Sample Data:
{sample_df.head().to_string()}

Data Types:
{sample_df.dtypes.to_string()}
"""
            
            prompt = f"""Based on this homelessness-related dataset from NYC Open Data, provide insights about:

1. Key patterns and trends in the data
2. Policy implications for homelessness
3. Data quality observations
4. Recommendations for further analysis

Dataset Information:
{context}

Please provide a comprehensive analysis focusing on homelessness research and policy implications."""

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "nvidia/llama-3-70b-instruct",
                        "messages": [
                            {"role": "system", "content": "You are an expert data analyst specializing in homelessness research and policy analysis."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.4,
                        "max_tokens": 1500
                    },
                    timeout=120.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        'analysis': result["choices"][0]["message"]["content"],
                        'provider': 'NVIDIA',
                        'model': 'llama-3-70b-instruct',
                        'success': True
                    }
                else:
                    return {
                        'analysis': f"API Error: {response.status_code} - {response.text}",
                        'provider': 'NVIDIA',
                        'model': 'llama-3-70b-instruct', 
                        'success': False
                    }
                    
        except Exception as e:
            return {
                'analysis': f"Analysis failed: {str(e)}",
                'provider': 'NVIDIA',
                'model': 'llama-3-70b-instruct',
                'success': False
            }


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HomelessnessAnalyticsWorkflow:
    """
    Comprehensive workflow for homelessness data analysis
    """
    
    def __init__(self, nvidia_api_key: str = None, cache_dir: str = "./homelessness_cache"):
        """
        Initialize the workflow
        
        Args:
            nvidia_api_key: NVIDIA API key for AI analysis
            cache_dir: Directory for caching results
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize Scout data discovery
        self.scout = ScoutDataDiscovery(
            cache_dir=str(self.cache_dir / "scout"),
            log_level="INFO"
        )
        
        # Initialize AI provider with NVIDIA
        self.ai_provider = None
        if nvidia_api_key:
            self.ai_provider = SimpleNvidiaProvider(nvidia_api_key)
        elif os.getenv('NVIDIA_API_KEY'):
            self.ai_provider = SimpleNvidiaProvider(os.getenv('NVIDIA_API_KEY'))
        else:
            logger.warning("No NVIDIA API key provided. AI analysis will be limited.")
        
        # Initialize relationship graph
        self.relationship_graph = DatasetRelationshipGraph()
        
        # Storage for datasets and analyses
        self.homelessness_datasets = None
        self.related_datasets = None
        self.ai_analyses = {}
        self.quality_assessments = {}
        
        logger.info("HomelessnessAnalyticsWorkflow initialized")

    async def run_complete_workflow(self, output_dir: str = "./homelessness_analysis_results"):
        """
        Run the complete homelessness analytics workflow
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info("🏠 Starting Comprehensive Homelessness Analytics Workflow")
        
        # Step 1: Discover homelessness datasets
        logger.info("📊 Step 1: Discovering homelessness datasets...")
        await self.discover_homelessness_datasets()
        
        # Step 2: Find related datasets
        logger.info("🔍 Step 2: Finding related datasets...")
        await self.find_related_datasets()
        
        # Step 3: Quality assessment
        logger.info("✅ Step 3: Assessing data quality...")
        await self.assess_data_quality()
        
        # Step 4: AI analysis of each dataset
        if self.ai_provider:
            logger.info("🤖 Step 4: Running AI analysis on datasets...")
            await self.run_ai_analysis()
        else:
            logger.info("⚠️ Step 4: Skipping AI analysis (no API key provided)")
        logger.info("📈 Step 5: Creating analytics and visualizations...")
        await self.create_analytics_dashboard(str(output_path))
        
        # Step 6: Generate comprehensive report
        logger.info("📋 Step 6: Generating comprehensive report...")
        await self.generate_final_report(str(output_path))
        
        logger.info(f"✨ Workflow complete! Results saved to: {output_path}")
        return str(output_path)

    async def discover_homelessness_datasets(self) -> pd.DataFrame:
        """
        Discover homelessness-related datasets from NYC Open Data
        """
        # Search terms related to homelessness
        search_terms = [
            "homeless", "homelessness", "shelter", "supportive housing",
            "affordable housing", "housing assistance", "dhs", "human services",
            "temporary housing", "transitional housing", "street outreach"
        ]
        
        logger.info(f"Searching for datasets with terms: {search_terms}")
        
        # Search for datasets
        self.homelessness_datasets = self.scout.search_datasets(
            search_terms=search_terms,
            domains=['data.cityofnewyork.us'],
            limit=50
        )
        
        logger.info(f"Found {len(self.homelessness_datasets)} homelessness-related datasets")
        
        # Display top datasets
        if not self.homelessness_datasets.empty:
            print("\n🏠 Top Homelessness Datasets Found:")
            print("=" * 80)
            for idx, row in self.homelessness_datasets.head(10).iterrows():
                print(f"\n{idx + 1}. {row['name']}")
                print(f"   ID: {row['id']}")
                print(f"   Downloads: {row.get('download_count', 'N/A')}")
                print(f"   Category: {row.get('category', 'N/A')}")
                print(f"   Description: {row.get('description', 'No description')[:100]}...")
        
        return self.homelessness_datasets

    async def find_related_datasets(self) -> pd.DataFrame:
        """
        Find datasets related to homelessness using Scout's relationship graph
        """
        if self.homelessness_datasets is None or self.homelessness_datasets.empty:
            logger.warning("No homelessness datasets found. Cannot find related datasets.")
            return pd.DataFrame()
        
        # Search for broader social services datasets
        broader_terms = [
            "social services", "mental health", "substance abuse", "unemployment",
            "poverty", "welfare", "food assistance", "medicaid", "healthcare",
            "education", "crime", "police", "arrest", "domestic violence"
        ]
        
        logger.info(f"Searching for related datasets with terms: {broader_terms}")
        
        self.related_datasets = self.scout.search_datasets(
            search_terms=broader_terms,
            domains=['data.cityofnewyork.us'],
            limit=100
        )
        
        # Combine datasets for relationship analysis
        all_datasets = pd.concat([
            self.homelessness_datasets,
            self.related_datasets
        ]).drop_duplicates(subset=['id'])
        
        # Add to relationship graph
        self.relationship_graph.add_datasets(all_datasets)
        
        # Calculate relationships
        stats = self.relationship_graph.calculate_relationships(
            similarity_threshold=0.25,
            content_weight=0.3,
            structural_weight=0.2,
            metadata_weight=0.2,
            tag_weight=0.15,
            category_weight=0.15
        )
        
        logger.info(f"Found {len(self.related_datasets)} related datasets")
        logger.info(f"Relationship analysis: {stats['relationships_found']} connections found")
        
        return self.related_datasets

    async def assess_data_quality(self):
        """
        Assess quality of key homelessness datasets
        """
        if self.homelessness_datasets is None or self.homelessness_datasets.empty:
            logger.warning("No datasets to assess")
            return
        
        # Assess top datasets
        top_datasets = self.homelessness_datasets.head(10)
        
        for idx, row in top_datasets.iterrows():
            dataset_id = row['id']
            dataset_name = row['name']
            
            try:
                logger.info(f"Assessing quality for: {dataset_name}")
                
                assessment = self.scout.assess_dataset_quality(dataset_id)
                self.quality_assessments[dataset_id] = {
                    'assessment': assessment,
                    'dataset_info': row.to_dict()
                }
                
                score = assessment['overall_scores']['total_score']
                grade = assessment['overall_scores']['letter_grade']
                
                logger.info(f"  Quality Score: {score:.1f}/100 (Grade: {grade})")
                
            except Exception as e:
                logger.error(f"Failed to assess {dataset_name}: {e}")

    async def run_ai_analysis(self):
        """
        Run AI analysis on each dataset using NVIDIA models
        """
        if not self.ai_provider:
            logger.warning("AI Provider not available. Skipping AI analysis.")
            return
        
        # Analyze top quality datasets
        top_datasets = self.homelessness_datasets.head(10)
        
        for idx, row in top_datasets.iterrows():
            dataset_id = row['id']
            dataset_name = row['name']
            dataset_info = row.to_dict()
            
            try:
                logger.info(f"Running AI analysis for: {dataset_name}")
                
                # Download a sample of the dataset
                sample_df = self.scout.download_dataset_sample(
                    dataset_id, 
                    sample_size=1000
                )
                
                if sample_df is not None and not sample_df.empty:
                    # Generate AI analysis
                    analysis = await self.ai_provider.analyze_dataset(
                        dataset_info, sample_df
                    )
                    
                    self.ai_analyses[dataset_id] = {
                        'analysis': analysis,
                        'dataset_info': dataset_info,
                        'sample_data': sample_df.head().to_dict()
                    }
                    
                    logger.info(f"  ✅ AI analysis completed for {dataset_name}")
                else:
                    logger.warning(f"  ⚠️ Could not download sample for {dataset_name}")
                    
            except Exception as e:
                logger.error(f"Failed AI analysis for {dataset_name}: {e}")

    async def create_analytics_dashboard(self, output_dir: str):
        """
        Create comprehensive analytics dashboard with visualizations
        """
        output_path = Path(output_dir)
        
        # 1. Dataset Overview Dashboard
        await self._create_dataset_overview_dashboard(output_path)
        
        # 2. Quality Assessment Visualization
        await self._create_quality_dashboard(output_path)
        
        # 3. Relationship Network Visualization
        await self._create_relationship_visualization(output_path)
        
        # 4. Data Analysis Dashboard (if we have sample data)
        if self.ai_analyses:
            await self._create_data_analysis_dashboard(output_path)

    async def _create_dataset_overview_dashboard(self, output_path: Path):
        """Create overview dashboard of discovered datasets"""
        if self.homelessness_datasets is None or self.homelessness_datasets.empty:
            return
            
        # Prepare data for visualization
        df = self.homelessness_datasets.copy()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Datasets by Category',
                'Download Counts Distribution',
                'Dataset Creation Timeline',
                'Top 10 Most Downloaded Datasets'
            ],
            specs=[[{'type': 'pie'}, {'type': 'histogram'}],
                   [{'type': 'scatter'}, {'type': 'bar'}]]
        )
        
        # 1. Category distribution
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=category_counts.index,
                    values=category_counts.values,
                    name="Categories"
                ),
                row=1, col=1
            )
        
        # 2. Download counts histogram
        if 'download_count' in df.columns:
            download_counts = pd.to_numeric(df['download_count'], errors='coerce').dropna()
            fig.add_trace(
                go.Histogram(
                    x=download_counts,
                    name="Downloads",
                    nbinsx=20
                ),
                row=1, col=2
            )
        
        # 3. Creation timeline
        if 'createdAt' in df.columns:
            df['created_date'] = pd.to_datetime(df['createdAt'], errors='coerce')
            timeline_data = df.dropna(subset=['created_date'])
            if not timeline_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=timeline_data['created_date'],
                        y=timeline_data.index,
                        mode='markers',
                        name="Creation Date"
                    ),
                    row=2, col=1
                )
        
        # 4. Top downloads
        if 'download_count' in df.columns:
            top_downloads = df.nlargest(10, 'download_count')
            fig.add_trace(
                go.Bar(
                    x=top_downloads['download_count'],
                    y=top_downloads['name'],
                    orientation='h',
                    name="Top Downloads"
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="NYC Homelessness Datasets - Overview Dashboard",
            height=800,
            showlegend=False
        )
        
        # Save dashboard
        dashboard_path = output_path / "01_dataset_overview_dashboard.html"
        fig.write_html(str(dashboard_path))
        logger.info(f"Saved dataset overview dashboard: {dashboard_path}")

    async def _create_quality_dashboard(self, output_path: Path):
        """Create quality assessment dashboard"""
        if not self.quality_assessments:
            return
            
        # Prepare quality data
        quality_data = []
        for dataset_id, data in self.quality_assessments.items():
            assessment = data['assessment']
            dataset_info = data['dataset_info']
            
            scores = assessment['overall_scores']
            quality_data.append({
                'dataset_id': dataset_id,
                'name': dataset_info['name'],
                'total_score': scores['total_score'],
                'letter_grade': scores['letter_grade'],
                'completeness': assessment['detailed_scores']['completeness']['score'],
                'consistency': assessment['detailed_scores']['consistency']['score'],
                'accuracy': assessment['detailed_scores']['accuracy']['score'],
                'timeliness': assessment['detailed_scores']['timeliness']['score'],
                'usability': assessment['detailed_scores']['usability']['score'],
            })
        
        quality_df = pd.DataFrame(quality_data)
        
        # Create quality dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Overall Quality Scores',
                'Quality Dimensions Comparison',
                'Grade Distribution',
                'Quality vs Popularity'
            ]
        )
        
        # 1. Overall scores
        fig.add_trace(
            go.Bar(
                x=quality_df['name'],
                y=quality_df['total_score'],
                name="Quality Score",
                text=quality_df['letter_grade'],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 2. Quality dimensions radar chart
        dimensions = ['completeness', 'consistency', 'accuracy', 'timeliness', 'usability']
        avg_scores = [quality_df[dim].mean() for dim in dimensions]
        
        fig.add_trace(
            go.Scatterpolar(
                r=avg_scores + [avg_scores[0]],  # Close the polygon
                theta=dimensions + [dimensions[0]],
                fill='toself',
                name="Average Quality"
            ),
            row=1, col=2
        )
        
        # 3. Grade distribution
        grade_counts = quality_df['letter_grade'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=grade_counts.index,
                values=grade_counts.values,
                name="Grades"
            ),
            row=2, col=1
        )
        
        # 4. Quality vs downloads (if available)
        if 'download_count' in self.homelessness_datasets.columns:
            # Merge with download data
            merged = quality_df.merge(
                self.homelessness_datasets[['id', 'download_count']],
                left_on='dataset_id',
                right_on='id',
                how='left'
            )
            
            fig.add_trace(
                go.Scatter(
                    x=pd.to_numeric(merged['download_count'], errors='coerce'),
                    y=merged['total_score'],
                    mode='markers',
                    text=merged['name'],
                    name="Quality vs Downloads"
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Data Quality Assessment Dashboard",
            height=800,
            showlegend=False
        )
        
        # Save dashboard
        dashboard_path = output_path / "02_quality_assessment_dashboard.html"
        fig.write_html(str(dashboard_path))
        logger.info(f"Saved quality dashboard: {dashboard_path}")

    async def _create_relationship_visualization(self, output_path: Path):
        """Create dataset relationship network visualization"""
        try:
            # Create interactive relationship graph
            graph_path = output_path / "03_dataset_relationships.html"
            self.relationship_graph.create_interactive_graph(
                output_path=str(graph_path),
                height=800
            )
            logger.info(f"Saved relationship visualization: {graph_path}")
            
            # Generate relationship report
            report_path = output_path / "dataset_relationships_report.txt"
            self.relationship_graph.generate_relationship_report(
                output_path=str(report_path)
            )
            logger.info(f"Saved relationship report: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to create relationship visualization: {e}")

    async def _create_data_analysis_dashboard(self, output_path: Path):
        """Create data analysis dashboard from AI insights"""
        if not self.ai_analyses:
            return
            
        # Create AI insights summary dashboard
        insights_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Analysis Insights - NYC Homelessness Data</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { text-align: center; color: #2c3e50; margin-bottom: 40px; }
                .dataset-card { 
                    background: white; 
                    margin: 20px 0; 
                    padding: 20px; 
                    border-radius: 8px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .dataset-title { color: #3498db; font-size: 20px; font-weight: bold; margin-bottom: 10px; }
                .question { color: #2980b9; font-weight: bold; margin-top: 20px; }
                .answer { margin: 10px 0; line-height: 1.6; }
                .metadata { color: #7f8c8d; font-size: 12px; margin-top: 10px; }
                .nav { position: sticky; top: 0; background: white; padding: 10px; text-align: center; border-bottom: 1px solid #eee; }
                .nav a { margin: 0 10px; color: #3498db; text-decoration: none; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 AI Analysis Insights</h1>
                    <h2>NYC Homelessness Data Analysis</h2>
                    <p>Generated using NVIDIA AI Models</p>
                </div>
                
                <div class="nav">
                    <strong>Datasets Analyzed:</strong>
        """
        
        # Add navigation
        for i, (dataset_id, data) in enumerate(self.ai_analyses.items()):
            dataset_name = data['dataset_info']['name'][:50]
            insights_html += f'<a href="#dataset-{i}">{dataset_name}...</a>'
        
        insights_html += "</div>"
        
        # Add dataset analyses
        for i, (dataset_id, data) in enumerate(self.ai_analyses.items()):
            analysis = data['analysis']
            dataset_info = data['dataset_info']
            
            insights_html += f"""
                <div class="dataset-card" id="dataset-{i}">
                    <div class="dataset-title">{dataset_info['name']}</div>
                    <p><strong>Dataset ID:</strong> {dataset_id}</p>
                    <p><strong>Category:</strong> {dataset_info.get('category', 'N/A')}</p>
                    <p><strong>Description:</strong> {dataset_info.get('description', 'No description')[:200]}...</p>
            """
            
            # Add AI analysis results
            if analysis.get('success', False):
                insights_html += f"""
                    <div class="question">AI Analysis Results</div>
                    <div class="answer">{analysis['analysis']}</div>
                    <div class="metadata">
                        Provider: {analysis.get('provider', 'N/A')} | 
                        Model: {analysis.get('model', 'N/A')}
                    </div>
                """
            else:
                insights_html += f"""
                    <div class="question">AI Analysis Results</div>
                    <div class="answer" style="color: #e74c3c;">Analysis failed: {analysis.get('analysis', 'Unknown error')}</div>
                """
            
            insights_html += "</div>"
        
        insights_html += """
            </div>
        </body>
        </html>
        """
        
        # Save AI insights dashboard
        insights_path = output_path / "04_ai_insights_dashboard.html"
        with open(insights_path, 'w', encoding='utf-8') as f:
            f.write(insights_html)
        
        logger.info(f"Saved AI insights dashboard: {insights_path}")

    async def generate_final_report(self, output_dir: str):
        """
        Generate comprehensive final report
        """
        output_path = Path(output_dir)
        
        # Generate summary statistics
        stats = {
            'total_datasets_found': len(self.homelessness_datasets) if self.homelessness_datasets is not None else 0,
            'related_datasets_found': len(self.related_datasets) if self.related_datasets is not None else 0,
            'datasets_quality_assessed': len(self.quality_assessments),
            'datasets_ai_analyzed': len(self.ai_analyses),
            'analysis_timestamp': datetime.now().isoformat(),
        }
        
        # Quality statistics
        if self.quality_assessments:
            quality_scores = [
                data['assessment']['overall_scores']['total_score'] 
                for data in self.quality_assessments.values()
            ]
            stats['quality_stats'] = {
                'average_quality_score': np.mean(quality_scores),
                'highest_quality_score': np.max(quality_scores),
                'lowest_quality_score': np.min(quality_scores),
                'quality_std': np.std(quality_scores),
            }
        
        # Relationship statistics
        if hasattr(self.relationship_graph, 'graph_stats'):
            stats['relationship_stats'] = self.relationship_graph.graph_stats
        
        # Create markdown report
        report_content = f"""# NYC Homelessness Data Analysis Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents a comprehensive analysis of homelessness-related datasets from NYC Open Data, utilizing automated data discovery, AI-powered analysis, and quality assessment methodologies.

## Key Findings

### Dataset Discovery
- **Total Homelessness Datasets Found:** {stats['total_datasets_found']}
- **Related Datasets Identified:** {stats['related_datasets_found']}
- **Datasets Quality Assessed:** {stats['datasets_quality_assessed']}
- **Datasets AI Analyzed:** {stats['datasets_ai_analyzed']}

### Data Quality Assessment
"""
        
        if 'quality_stats' in stats:
            qs = stats['quality_stats']
            report_content += f"""
- **Average Quality Score:** {qs['average_quality_score']:.1f}/100
- **Highest Quality Score:** {qs['highest_quality_score']:.1f}/100
- **Lowest Quality Score:** {qs['lowest_quality_score']:.1f}/100
- **Quality Standard Deviation:** {qs['quality_std']:.1f}
"""
        
        report_content += """
## Methodology

### 1. Data Discovery
The analysis utilized Scout data discovery framework to systematically search NYC Open Data for homelessness-related datasets using comprehensive search terms including:
- Direct terms: homeless, homelessness, shelter
- Related terms: supportive housing, affordable housing, housing assistance
- Agency terms: DHS, human services
- Program terms: transitional housing, street outreach

### 2. Quality Assessment
Each dataset was evaluated across five dimensions:
- **Completeness (25%):** Missing data assessment
- **Consistency (20%):** Data type and format consistency  
- **Accuracy (20%):** Outlier detection and validation
- **Timeliness (15%):** Data freshness and update frequency
- **Usability (20%):** Structure and accessibility

### 3. AI-Powered Analysis
NVIDIA AI models were employed to provide deeper insights into each dataset, analyzing:
- Key patterns and trends
- Policy implications
- Data limitations
- Complementary data recommendations

### 4. Relationship Mapping
Dataset relationship analysis identified connections between homelessness data and related social services datasets through:
- Content similarity analysis
- Structural pattern matching
- Metadata correlation
- Tag and category overlap

## Dataset Highlights

"""
        
        # Add top datasets
        if self.homelessness_datasets is not None and not self.homelessness_datasets.empty:
            top_datasets = self.homelessness_datasets.head(5)
            for idx, row in top_datasets.iterrows():
                quality_info = ""
                if row['id'] in self.quality_assessments:
                    score = self.quality_assessments[row['id']]['assessment']['overall_scores']['total_score']
                    grade = self.quality_assessments[row['id']]['assessment']['overall_scores']['letter_grade']
                    quality_info = f" (Quality: {score:.1f}/100, Grade: {grade})"
                
                report_content += f"""
### {row['name']}{quality_info}
- **Dataset ID:** {row['id']}
- **Category:** {row.get('category', 'N/A')}
- **Downloads:** {row.get('download_count', 'N/A')}
- **Description:** {row.get('description', 'No description')[:200]}...
"""
        
        report_content += """
## Technical Implementation

### Tools and Technologies Used
- **Scout Data Discovery:** Automated dataset discovery and quality assessment
- **NVIDIA AI Models:** Advanced natural language analysis
- **Pandas & NumPy:** Data processing and statistical analysis
- **Plotly:** Interactive visualizations and dashboards
- **Network Analysis:** Dataset relationship mapping

### Architecture
The workflow implements a modular architecture combining:
1. **Data Discovery Layer:** Scout-based automated search and cataloging
2. **Quality Assessment Layer:** Multi-dimensional quality scoring
3. **AI Analysis Layer:** NVIDIA-powered content analysis
4. **Visualization Layer:** Interactive dashboards and reports
5. **Relationship Layer:** Network analysis of dataset connections

## Recommendations

### Data Quality Improvements
Based on the quality assessment, recommendations include:
1. Standardize missing data handling across datasets
2. Implement regular data validation checks
3. Improve metadata documentation
4. Establish update frequency guidelines

### Policy Implications
The analysis reveals opportunities for:
1. Enhanced data integration across agencies
2. Improved tracking of housing outcomes
3. Better coordination of social services
4. Evidence-based policy development

### Future Analysis
Suggested extensions to this work:
1. Temporal analysis of homelessness trends
2. Geographic distribution analysis
3. Cross-agency data integration
4. Predictive modeling development

## Files Generated

1. `01_dataset_overview_dashboard.html` - Interactive overview of all discovered datasets
2. `02_quality_assessment_dashboard.html` - Comprehensive quality analysis visualizations
3. `03_dataset_relationships.html` - Network visualization of dataset relationships
4. `04_ai_insights_dashboard.html` - AI-generated insights and analysis
5. `dataset_relationships_report.txt` - Detailed relationship analysis report
6. `homelessness_analysis_report.md` - This comprehensive report

## Conclusion

This comprehensive analysis demonstrates the power of combining automated data discovery, AI-powered analysis, and systematic quality assessment for understanding complex social issues like homelessness. The workflow provides a scalable framework for ongoing data analysis and policy support.

---

*Report generated by HomelessnessAnalyticsWorkflow*
*Powered by Scout Data Discovery and NVIDIA AI*
"""
        
        # Save report
        report_path = output_path / "homelessness_analysis_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Save statistics as JSON
        stats_path = output_path / "analysis_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Saved final report: {report_path}")
        logger.info(f"Saved statistics: {stats_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("🏠 NYC HOMELESSNESS ANALYTICS WORKFLOW COMPLETE")
        print("="*80)
        print(f"📊 Datasets Found: {stats['total_datasets_found']}")
        print(f"🔍 Related Datasets: {stats['related_datasets_found']}")
        print(f"✅ Quality Assessed: {stats['datasets_quality_assessed']}")
        print(f"🤖 AI Analyzed: {stats['datasets_ai_analyzed']}")
        if 'quality_stats' in stats:
            print(f"📈 Avg Quality Score: {stats['quality_stats']['average_quality_score']:.1f}/100")
        print(f"📁 Results saved to: {output_dir}")
        print("="*80)


# Convenience functions for interactive use
async def quick_homelessness_analysis(nvidia_api_key: str = None, output_dir: str = "./homelessness_results"):
    """
    Quick function to run the complete homelessness analysis workflow
    
    Args:
        nvidia_api_key: NVIDIA API key (or set NVIDIA_API_KEY environment variable)
        output_dir: Directory to save results
    
    Returns:
        Path to results directory
    """
    workflow = HomelessnessAnalyticsWorkflow(nvidia_api_key=nvidia_api_key)
    return await workflow.run_complete_workflow(output_dir=output_dir)


# Main execution function
async def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NYC Homelessness Analytics Workflow')
    parser.add_argument('--nvidia-api-key', help='NVIDIA API key')
    parser.add_argument('--output-dir', default='./homelessness_analysis_results', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Use API key from args or environment
    api_key = args.nvidia_api_key or os.getenv('NVIDIA_API_KEY')
    
    if not api_key:
        print("⚠️  No NVIDIA API key provided. Set NVIDIA_API_KEY environment variable or use --nvidia-api-key")
        print("   AI analysis will be limited without this key.")
    
    # Run workflow
    workflow = HomelessnessAnalyticsWorkflow(nvidia_api_key=api_key)
    results_path = await workflow.run_complete_workflow(output_dir=args.output_dir)
    
    print(f"\n✨ Analysis complete! Open {results_path}/01_dataset_overview_dashboard.html to explore results.")


if __name__ == "__main__":
    asyncio.run(main())