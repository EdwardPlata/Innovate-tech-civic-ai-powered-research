"""
NYC Homelessness Geographic Analytics

Memory-efficient workflow for analyzing homelessness location data and creating 
map overlays with accident/collision clustering for hotspot analysis.

Features:
- Targeted search for location-based datasets
- Memory-efficient data processing
- Geographic clustering analysis
- Interactive map overlays
- Accident correlation analysis
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import sys
import logging

# Add Scout path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "scout_data_discovery"))

from scout_data_discovery.src.scout_discovery import ScoutDataDiscovery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HomelessnessGeoAnalytics:
    """
    Memory-efficient geographic analysis of homelessness and related incident data
    """
    
    def __init__(self, cache_dir: str = "./geo_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize Scout with minimal config
        self.scout = ScoutDataDiscovery(
            cache_dir=str(self.cache_dir / "scout"),
            log_level="INFO"
        )
        
        # Storage for key datasets
        self.location_datasets = {}
        self.processed_data = {}
        
        logger.info("HomelessnessGeoAnalytics initialized")

    def find_location_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Find datasets with location/geographic information
        """
        logger.info("🗺️ Searching for location-based datasets...")
        
        # Search for specific location-based datasets
        search_categories = {
            'homelessness': ['homeless shelter', 'dhs shelter', 'supportive housing'],
            'accidents': ['motor vehicle collision', 'traffic accident', 'crash'],
            'services': ['social services facility', 'health facility', 'mental health'],
            'housing': ['affordable housing', 'public housing', 'housing development']
        }
        
        results = {}
        
        for category, terms in search_categories.items():
            logger.info(f"Searching for {category} datasets...")
            
            try:
                datasets = self.scout.search_datasets(
                    search_terms=terms,
                    domains=['data.cityofnewyork.us'],
                    limit=10  # Keep small to save memory
                )
                
                if not datasets.empty:
                    # Filter for datasets likely to have location data
                    location_datasets = self._filter_location_datasets(datasets)
                    if not location_datasets.empty:
                        results[category] = location_datasets
                        logger.info(f"Found {len(location_datasets)} {category} datasets with location data")
                
            except Exception as e:
                logger.error(f"Error searching {category}: {e}")
        
        self.location_datasets = results
        return results

    def _filter_location_datasets(self, datasets: pd.DataFrame) -> pd.DataFrame:
        """
        Filter datasets likely to contain location data
        """
        location_keywords = [
            'latitude', 'longitude', 'lat', 'lon', 'coordinate', 'address',
            'location', 'borough', 'zip', 'geocoded', 'point', 'geom'
        ]
        
        # Check names and descriptions for location keywords
        location_datasets = []
        
        for idx, row in datasets.iterrows():
            name = str(row.get('name', '')).lower()
            desc = str(row.get('description', '')).lower()
            
            # Check if likely to have location data
            has_location_keywords = any(
                keyword in name or keyword in desc 
                for keyword in location_keywords
            )
            
            if has_location_keywords:
                location_datasets.append(row)
        
        return pd.DataFrame(location_datasets) if location_datasets else pd.DataFrame()

    def analyze_key_datasets(self, max_sample_size: int = 5000) -> Dict[str, Dict]:
        """
        Analyze key datasets for location data with memory efficiency
        """
        logger.info("📊 Analyzing key location datasets...")
        
        analysis_results = {}
        
        for category, datasets in self.location_datasets.items():
            logger.info(f"Analyzing {category} datasets...")
            category_results = {}
            
            # Analyze top 3 datasets per category to save memory
            top_datasets = datasets.head(3)
            
            for idx, row in top_datasets.iterrows():
                dataset_id = row['id']
                dataset_name = row['name']
                
                try:
                    logger.info(f"  Sampling dataset: {dataset_name}")
                    
                    # Download small sample
                    sample_df = self.scout.download_dataset_sample(
                        dataset_id, 
                        sample_size=max_sample_size
                    )
                    
                    if sample_df is not None and not sample_df.empty:
                        # Analyze location columns
                        location_analysis = self._analyze_location_columns(sample_df)
                        
                        if location_analysis['has_location_data']:
                            category_results[dataset_id] = {
                                'name': dataset_name,
                                'data': sample_df,
                                'location_info': location_analysis,
                                'metadata': row.to_dict()
                            }
                            
                            logger.info(f"    ✅ Found location data: {location_analysis['location_columns']}")
                        else:
                            logger.info(f"    ⚠️ No clear location data found")
                
                except Exception as e:
                    logger.error(f"    ❌ Failed to analyze {dataset_name}: {e}")
            
            if category_results:
                analysis_results[category] = category_results
        
        self.processed_data = analysis_results
        return analysis_results

    def _analyze_location_columns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze DataFrame for location-related columns
        """
        location_patterns = {
            'latitude': ['lat', 'latitude', 'y_coord', 'y'],
            'longitude': ['lon', 'lng', 'longitude', 'x_coord', 'x'],
            'address': ['address', 'location', 'street'],
            'borough': ['borough', 'boro'],
            'zipcode': ['zip', 'zipcode', 'postal'],
            'coordinates': ['coordinates', 'point', 'geom', 'the_geom']
        }
        
        found_columns = {}
        location_columns = []
        
        # Check column names (case insensitive)
        column_names_lower = [col.lower() for col in df.columns]
        
        for location_type, patterns in location_patterns.items():
            for pattern in patterns:
                matching_cols = [
                    col for col in df.columns 
                    if pattern in col.lower()
                ]
                if matching_cols:
                    found_columns[location_type] = matching_cols
                    location_columns.extend(matching_cols)
        
        # Check for coordinate pairs
        has_lat_lon = ('latitude' in found_columns and 'longitude' in found_columns)
        
        # Sample data to check for coordinate-like values
        coordinate_samples = {}
        if location_columns:
            for col in location_columns[:5]:  # Check first 5 location columns
                sample_values = df[col].dropna().head(10).tolist()
                coordinate_samples[col] = sample_values
        
        return {
            'has_location_data': len(location_columns) > 0,
            'location_columns': location_columns,
            'found_patterns': found_columns,
            'has_lat_lon_pair': has_lat_lon,
            'coordinate_samples': coordinate_samples,
            'total_rows': len(df),
            'non_null_location_rows': len(df.dropna(subset=location_columns)) if location_columns else 0
        }

    def create_homelessness_clustering_map(self, output_dir: str = "./geo_results"):
        """
        Create clustering maps for homelessness and accident data
        """
        logger.info("🗺️ Creating geographic clustering maps...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Process each category
        maps_created = []
        
        for category, datasets in self.processed_data.items():
            if not datasets:
                continue
                
            logger.info(f"Creating map for {category}...")
            
            try:
                map_file = self._create_category_map(category, datasets, output_path)
                if map_file:
                    maps_created.append(map_file)
                    
            except Exception as e:
                logger.error(f"Failed to create map for {category}: {e}")
        
        # Create combined overlay map if we have multiple categories
        if len(maps_created) >= 2:
            try:
                combined_map = self._create_combined_overlay_map(output_path)
                if combined_map:
                    maps_created.append(combined_map)
            except Exception as e:
                logger.error(f"Failed to create combined map: {e}")
        
        return maps_created

    def _create_category_map(self, category: str, datasets: Dict, output_path: Path) -> Optional[str]:
        """
        Create map for a specific category of data
        """
        all_points = []
        
        # Collect all valid coordinate points
        for dataset_id, data in datasets.items():
            df = data['data']
            location_info = data['location_info']
            name = data['name']
            
            # Extract coordinates
            points = self._extract_coordinates(df, location_info)
            if points:
                for point in points:
                    point['dataset'] = name
                    point['category'] = category
                all_points.append(point)
        
        if not all_points:
            logger.warning(f"No valid coordinates found for {category}")
            return None
        
        # Convert to DataFrame
        points_df = pd.DataFrame(all_points)
        
        # Create map
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scattermapbox(
            lat=points_df['lat'],
            lon=points_df['lon'],
            mode='markers',
            marker=dict(
                size=8,
                color=px.colors.qualitative.Set1[hash(category) % len(px.colors.qualitative.Set1)],
                opacity=0.7
            ),
            text=points_df.apply(lambda row: f"{row['category']}: {row['dataset']}<br>Location: {row.get('address', 'N/A')}", axis=1),
            name=category.title()
        ))
        
        # Update layout
        fig.update_layout(
            title=f"NYC {category.title()} Locations",
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=40.7128, lon=-74.0060),  # NYC center
                zoom=10
            ),
            height=600,
            margin=dict(t=50, b=0, l=0, r=0)
        )
        
        # Save map
        map_filename = f"{category}_locations_map.html"
        map_path = output_path / map_filename
        fig.write_html(str(map_path))
        
        logger.info(f"Saved {category} map: {map_path}")
        return str(map_path)

    def _create_combined_overlay_map(self, output_path: Path) -> Optional[str]:
        """
        Create combined overlay map showing all categories
        """
        logger.info("Creating combined overlay map...")
        
        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        
        color_idx = 0
        total_points = 0
        
        for category, datasets in self.processed_data.items():
            if not datasets:
                continue
                
            # Collect points for this category
            category_points = []
            
            for dataset_id, data in datasets.items():
                df = data['data']
                location_info = data['location_info']
                name = data['name']
                
                points = self._extract_coordinates(df, location_info)
                if points:
                    for point in points:
                        point['dataset'] = name
                        point['category'] = category
                    category_points.extend(points)
            
            if category_points:
                points_df = pd.DataFrame(category_points)
                total_points += len(points_df)
                
                # Add to map
                fig.add_trace(go.Scattermapbox(
                    lat=points_df['lat'],
                    lon=points_df['lon'],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=colors[color_idx % len(colors)],
                        opacity=0.6
                    ),
                    text=points_df.apply(lambda row: f"{row['category']}: {row['dataset']}", axis=1),
                    name=category.title()
                ))
                
                color_idx += 1
        
        if total_points == 0:
            logger.warning("No valid coordinates found for combined map")
            return None
        
        # Update layout
        fig.update_layout(
            title=f"NYC Homelessness & Related Services - Geographic Analysis ({total_points} locations)",
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=40.7128, lon=-74.0060),
                zoom=10
            ),
            height=700,
            margin=dict(t=50, b=0, l=0, r=0),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            )
        )
        
        # Save combined map
        combined_path = output_path / "combined_homelessness_services_map.html"
        fig.write_html(str(combined_path))
        
        logger.info(f"Saved combined map: {combined_path}")
        return str(combined_path)

    def _extract_coordinates(self, df: pd.DataFrame, location_info: Dict) -> List[Dict]:
        """
        Extract coordinate points from DataFrame
        """
        points = []
        
        # Find latitude and longitude columns
        lat_cols = []
        lon_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in ['lat', 'y_coord']):
                lat_cols.append(col)
            elif any(pattern in col_lower for pattern in ['lon', 'lng', 'x_coord']):
                lon_cols.append(col)
        
        # Extract coordinates
        if lat_cols and lon_cols:
            lat_col = lat_cols[0]
            lon_col = lon_cols[0]
            
            # Filter valid coordinates
            valid_data = df.dropna(subset=[lat_col, lon_col])
            
            for _, row in valid_data.iterrows():
                try:
                    lat = float(row[lat_col])
                    lon = float(row[lon_col])
                    
                    # Basic validation for NYC area
                    if 40.4 <= lat <= 40.9 and -74.3 <= lon <= -73.7:
                        point = {
                            'lat': lat,
                            'lon': lon,
                            'address': row.get('address', row.get('location', 'N/A'))
                        }
                        points.append(point)
                        
                except (ValueError, TypeError):
                    continue
        
        return points

    def generate_analysis_report(self, output_dir: str = "./geo_results"):
        """
        Generate analysis summary report
        """
        output_path = Path(output_dir)
        
        # Count totals
        total_datasets = sum(len(datasets) for datasets in self.location_datasets.values())
        total_processed = sum(len(datasets) for datasets in self.processed_data.values())
        
        # Create summary
        report = f"""# NYC Homelessness Geographic Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Location Datasets Found:** {total_datasets}
- **Datasets Successfully Processed:** {total_processed}
- **Categories Analyzed:** {list(self.location_datasets.keys())}

## Dataset Categories

"""
        
        for category, datasets in self.processed_data.items():
            report += f"\n### {category.title()}\n"
            
            for dataset_id, data in datasets.items():
                location_info = data['location_info']
                report += f"""
- **{data['name']}**
  - Dataset ID: {dataset_id}
  - Rows: {location_info['total_rows']:,}
  - Location Columns: {', '.join(location_info['location_columns'])}
  - Valid Location Records: {location_info['non_null_location_rows']:,}
"""
        
        report += """
## Files Generated

- Individual category maps: `{category}_locations_map.html`
- Combined overlay map: `combined_homelessness_services_map.html`
- This report: `geographic_analysis_report.md`

## Analysis Notes

This analysis focused on memory-efficient processing of location-based datasets related to homelessness, accidents, and social services in NYC. The maps provide geographic clustering visualization to identify hotspots and service gaps.
"""
        
        # Save report
        report_path = output_path / "geographic_analysis_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved: {report_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("🗺️ NYC HOMELESSNESS GEOGRAPHIC ANALYSIS COMPLETE")
        print("="*80)
        print(f"📊 Total Datasets Found: {total_datasets}")
        print(f"✅ Datasets Processed: {total_processed}")
        print(f"🗂️ Categories: {', '.join(self.location_datasets.keys())}")
        print(f"📁 Results saved to: {output_dir}")
        print("="*80)


async def run_geographic_analysis(output_dir: str = "./homelessness_geo_results"):
    """
    Run the complete geographic analysis workflow
    """
    analyzer = HomelessnessGeoAnalytics()
    
    # Step 1: Find location datasets
    location_datasets = analyzer.find_location_datasets()
    
    if not location_datasets:
        print("No location datasets found!")
        return
    
    # Step 2: Analyze key datasets
    processed_data = analyzer.analyze_key_datasets(max_sample_size=3000)
    
    if not processed_data:
        print("No datasets with valid location data found!")
        return
    
    # Step 3: Create maps
    maps = analyzer.create_homelessness_clustering_map(output_dir)
    
    # Step 4: Generate report
    analyzer.generate_analysis_report(output_dir)
    
    return output_dir


if __name__ == "__main__":
    import asyncio
    import argparse
    
    parser = argparse.ArgumentParser(description='NYC Homelessness Geographic Analytics')
    parser.add_argument('--output-dir', default='./homelessness_geo_results', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run analysis
    asyncio.run(run_geographic_analysis(args.output_dir))