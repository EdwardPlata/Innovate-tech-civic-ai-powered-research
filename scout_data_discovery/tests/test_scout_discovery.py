"""
Unit Tests for Scout Data Discovery

Test suite for the ScoutDataDiscovery class and related functionality.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.scout_discovery import ScoutDataDiscovery
from src.data_quality import DataQualityAssessor
from src.exceptions import (
    ScoutDiscoveryError, APIError, DataDownloadError,
    SearchError, ConfigurationError
)
from config.config_manager import ConfigManager


class TestScoutDataDiscovery(unittest.TestCase):
    """Test cases for ScoutDataDiscovery class"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_config = {
            'api': {'rate_limit_delay': 0.1, 'retry_attempts': 2},
            'data': {'default_sample_size': 100, 'quality_threshold': 70},
            'cache': {'duration_hours': 1}
        }

        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.scout = ScoutDataDiscovery(
            config=self.test_config,
            cache_dir=self.temp_dir,
            log_level="ERROR"  # Suppress logging during tests
        )

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test ScoutDataDiscovery initialization"""
        self.assertIsInstance(self.scout, ScoutDataDiscovery)
        self.assertIsInstance(self.scout.quality_assessor, DataQualityAssessor)
        self.assertEqual(self.scout.config['api']['retry_attempts'], 2)

    @patch('src.scout_discovery.requests.Session.get')
    def test_search_datasets_success(self, mock_get):
        """Test successful dataset search"""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [{
                'resource': {
                    'id': 'test-123',
                    'name': 'Test Dataset',
                    'description': 'A test dataset',
                    'download_count': 100,
                    'columns_name': ['col1', 'col2']
                },
                'classification': {
                    'domain_category': 'Transportation',
                    'tags': ['test', 'data']
                }
            }]
        }
        mock_get.return_value = mock_response

        # Test search
        results = self.scout.search_datasets("test")

        self.assertIsInstance(results, pd.DataFrame)
        self.assertGreater(len(results), 0)
        self.assertIn('id', results.columns)
        self.assertIn('name', results.columns)
        self.assertEqual(results.iloc[0]['id'], 'test-123')

    @patch('src.scout_discovery.requests.Session.get')
    def test_search_datasets_api_error(self, mock_get):
        """Test search with API error"""
        # Mock API error
        mock_get.side_effect = Exception("API Error")

        # Test that SearchError is raised
        with self.assertRaises(SearchError):
            self.scout.search_datasets("test")

    def test_extract_dataset_metadata(self):
        """Test dataset metadata extraction"""
        # Sample dataset from API
        dataset = {
            'resource': {
                'id': 'test-456',
                'name': 'Sample Dataset',
                'description': 'Sample description',
                'download_count': 500,
                'columns_name': ['col1', 'col2', 'col3']
            },
            'classification': {
                'domain_category': 'Health',
                'tags': ['health', 'public']
            }
        }

        metadata = self.scout._extract_dataset_metadata(dataset)

        self.assertEqual(metadata['id'], 'test-456')
        self.assertEqual(metadata['name'], 'Sample Dataset')
        self.assertEqual(metadata['download_count'], 500)
        self.assertEqual(len(metadata['columns_names']), 3)
        self.assertIn('metadata_quality', metadata)

    @patch('src.scout_discovery.requests.Session.get')
    def test_download_dataset_sample_success(self, mock_get):
        """Test successful dataset download"""
        # Mock successful download
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {'col1': 'value1', 'col2': 100},
            {'col1': 'value2', 'col2': 200}
        ]
        mock_get.return_value = mock_response

        df = self.scout.download_dataset_sample('test-dataset')

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn('col1', df.columns)
        self.assertIn('col2', df.columns)

    @patch('src.scout_discovery.requests.Session.get')
    def test_download_dataset_sample_error(self, mock_get):
        """Test dataset download with error"""
        # Mock API error
        mock_get.side_effect = Exception("Download failed")

        with self.assertRaises(DataDownloadError):
            self.scout.download_dataset_sample('invalid-dataset')

    def test_assess_metadata_quality(self):
        """Test metadata quality assessment"""
        # Complete metadata
        resource = {
            'name': 'Test Dataset',
            'description': 'A comprehensive test dataset',
            'attribution': 'Test Agency',
            'columns_name': ['col1', 'col2'],
            'download_count': 1000
        }
        classification = {
            'domain_category': 'Transportation',
            'tags': ['transport', 'public'],
            'domain_tags': ['traffic']
        }

        quality = self.scout._assess_metadata_quality(resource, classification)

        self.assertIn('score', quality)
        self.assertIn('percentage', quality)
        self.assertGreater(quality['score'], 5)  # Should have good score
        self.assertEqual(quality['quality_level'], 'High')

    def test_calculate_similarity(self):
        """Test dataset similarity calculation"""
        target = pd.Series({
            'domain_category': 'Transportation',
            'domain_tags': ['traffic', 'roads'],
            'tags': ['public'],
            'columns_names': ['street', 'count'],
            'description': 'traffic data on city streets'
        })

        candidate = pd.Series({
            'domain_category': 'Transportation',
            'domain_tags': ['traffic', 'vehicles'],
            'tags': ['public'],
            'columns_names': ['street', 'volume'],
            'description': 'vehicle traffic on city streets'
        })

        similarity = self.scout._calculate_similarity(target, candidate)

        self.assertIsInstance(similarity, float)
        self.assertGreater(similarity, 0.5)  # Should be quite similar
        self.assertLessEqual(similarity, 1.0)

    def test_generate_recommendations(self):
        """Test recommendation generation"""
        # Create sample catalog
        catalog_data = [
            {
                'id': 'dataset-1',
                'name': 'Traffic Dataset 1',
                'domain_category': 'Transportation',
                'domain_tags': ['traffic'],
                'tags': ['public'],
                'columns_names': ['street', 'count']
            },
            {
                'id': 'dataset-2',
                'name': 'Housing Dataset',
                'domain_category': 'Housing',
                'domain_tags': ['residential'],
                'tags': ['permits'],
                'columns_names': ['address', 'value']
            },
            {
                'id': 'dataset-3',
                'name': 'Traffic Dataset 2',
                'domain_category': 'Transportation',
                'domain_tags': ['traffic'],
                'tags': ['public'],
                'columns_names': ['road', 'volume']
            }
        ]

        catalog_df = pd.DataFrame(catalog_data)
        recommendations = self.scout.generate_recommendations('dataset-1', catalog_df, top_n=2)

        self.assertIsInstance(recommendations, pd.DataFrame)
        self.assertLessEqual(len(recommendations), 2)
        if not recommendations.empty:
            # First recommendation should be dataset-3 (more similar)
            self.assertEqual(recommendations.iloc[0]['id'], 'dataset-3')
            self.assertIn('similarity_score', recommendations.columns)

    def test_select_assessment_candidates(self):
        """Test candidate selection for assessment"""
        # Create sample dataset DataFrame
        data = [
            {'id': 'high-quality', 'download_count': 10000, 'page_views_total': 5000, 'metadata_quality': {'score': 9}},
            {'id': 'medium-quality', 'download_count': 1000, 'page_views_total': 500, 'metadata_quality': {'score': 6}},
            {'id': 'low-quality', 'download_count': 10, 'page_views_total': 5, 'metadata_quality': {'score': 3}},
        ]
        df = pd.DataFrame(data)

        candidates = self.scout._select_assessment_candidates(df, max_count=2)

        self.assertEqual(len(candidates), 2)
        # Should prioritize high-quality dataset
        self.assertEqual(candidates.iloc[0]['id'], 'high-quality')

    def test_empty_search_results(self):
        """Test handling of empty search results"""
        with patch('src.scout_discovery.requests.Session.get') as mock_get:
            # Mock empty results
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'results': []}
            mock_get.return_value = mock_response

            results = self.scout.search_datasets("nonexistent_term")

            self.assertIsInstance(results, pd.DataFrame)
            self.assertTrue(results.empty)


class TestDataQualityAssessor(unittest.TestCase):
    """Test cases for DataQualityAssessor class"""

    def setUp(self):
        """Set up test fixtures"""
        self.assessor = DataQualityAssessor()

    def test_assess_dataset_quality_complete_data(self):
        """Test quality assessment on complete, high-quality data"""
        # Create high-quality sample data
        data = {
            'id': range(1, 101),
            'name': [f'Name_{i}' for i in range(1, 101)],
            'value': np.random.normal(100, 15, 100),
            'category': ['A', 'B', 'C'] * 33 + ['A']
        }
        df = pd.DataFrame(data)

        assessment = self.assessor.assess_dataset_quality('test-dataset', df)

        # Check assessment structure
        self.assertIn('overall_scores', assessment)
        self.assertIn('completeness', assessment)
        self.assertIn('consistency', assessment)
        self.assertIn('accuracy', assessment)

        # Check scores
        scores = assessment['overall_scores']
        self.assertGreater(scores['completeness_score'], 95)  # Should be very complete
        self.assertGreater(scores['total_score'], 80)  # Should have good overall score
        self.assertIn(scores['grade'], ['A', 'B'])

    def test_assess_dataset_quality_missing_data(self):
        """Test quality assessment with missing data"""
        # Create data with missing values
        data = {
            'complete_col': range(100),
            'partial_missing': [i if i % 3 != 0 else None for i in range(100)],
            'mostly_missing': [i if i < 10 else None for i in range(100)],
            'empty_col': [None] * 100
        }
        df = pd.DataFrame(data)

        assessment = self.assessor.assess_dataset_quality('test-missing', df)

        completeness = assessment['completeness']
        self.assertEqual(len(completeness['complete_columns']), 1)
        self.assertEqual(len(completeness['empty_columns']), 1)
        self.assertLess(completeness['completeness_score'], 80)

    def test_assess_dataset_quality_outliers(self):
        """Test accuracy assessment with outliers"""
        # Create data with outliers
        normal_data = np.random.normal(50, 10, 95)
        outliers = [200, -100, 1000, -500, 800]  # Extreme outliers
        data = {
            'values': np.concatenate([normal_data, outliers])
        }
        df = pd.DataFrame(data)

        assessment = self.assessor.assess_dataset_quality('test-outliers', df)

        accuracy = assessment['accuracy']
        self.assertGreater(len(accuracy['accuracy_flags']), 0)
        self.assertLess(accuracy['accuracy_score'], 95)  # Should detect outliers

    def test_empty_dataset_handling(self):
        """Test handling of empty dataset"""
        empty_df = pd.DataFrame()

        with self.assertRaises(Exception):  # Should raise ValidationError
            self.assessor.assess_dataset_quality('empty-dataset', empty_df)

    def test_generate_quality_report(self):
        """Test quality report generation"""
        # Create sample assessments
        assessments = {
            'dataset-1': {
                'overall_scores': {
                    'total_score': 85.5,
                    'grade': 'B',
                    'completeness_score': 90,
                    'consistency_score': 80,
                    'accuracy_score': 85,
                    'timeliness_score': 75,
                    'usability_score': 88
                },
                'basic_stats': {
                    'row_count': 1000,
                    'column_count': 5,
                    'size_category': 'Medium'
                }
            },
            'dataset-2': {
                'error': 'Assessment failed'  # Should be excluded
            }
        }

        report = self.assessor.generate_quality_report(assessments)

        self.assertIsInstance(report, pd.DataFrame)
        self.assertEqual(len(report), 1)  # Only successful assessment
        self.assertEqual(report.iloc[0]['dataset_id'], 'dataset-1')
        self.assertEqual(report.iloc[0]['total_score'], 85.5)


class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create temporary config file
        self.temp_config = {
            'api': {'rate_limit_delay': 2.0},
            'custom_section': {'test_value': 'user_override'}
        }

        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        import yaml
        yaml.dump(self.temp_config, self.temp_file)
        self.temp_file.close()

    def tearDown(self):
        """Clean up test fixtures"""
        Path(self.temp_file.name).unlink(missing_ok=True)

    def test_load_default_config(self):
        """Test loading default configuration"""
        config_manager = ConfigManager()

        self.assertIn('api', config_manager.config)
        self.assertIn('data', config_manager.config)
        self.assertIn('quality', config_manager.config)

    def test_load_user_config(self):
        """Test loading user configuration"""
        config_manager = ConfigManager(config_path=self.temp_file.name)

        # Should have merged user config
        self.assertEqual(config_manager.get('api.rate_limit_delay'), 2.0)
        self.assertEqual(config_manager.get('custom_section.test_value'), 'user_override')

    def test_get_set_config(self):
        """Test getting and setting configuration values"""
        config_manager = ConfigManager()

        # Test get with dot notation
        original_delay = config_manager.get('api.rate_limit_delay')
        self.assertIsNotNone(original_delay)

        # Test set
        config_manager.set('api.rate_limit_delay', 1.5)
        self.assertEqual(config_manager.get('api.rate_limit_delay'), 1.5)

        # Test get with default
        self.assertEqual(config_manager.get('nonexistent.key', 'default'), 'default')

    def test_get_section(self):
        """Test getting entire configuration sections"""
        config_manager = ConfigManager()

        api_section = config_manager.get_section('api')
        self.assertIn('rate_limit_delay', api_section)
        self.assertIn('request_timeout', api_section)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow"""

    def setUp(self):
        """Set up integration test fixtures"""
        self.test_config = {
            'api': {'rate_limit_delay': 0.1, 'retry_attempts': 1},
            'data': {'default_sample_size': 50}
        }

    @patch('src.scout_discovery.requests.Session.get')
    def test_end_to_end_workflow(self, mock_get):
        """Test complete end-to-end workflow"""
        # Mock search response
        search_response = Mock()
        search_response.status_code = 200
        search_response.json.return_value = {
            'results': [{
                'resource': {
                    'id': 'integration-test',
                    'name': 'Integration Test Dataset',
                    'description': 'A dataset for integration testing',
                    'download_count': 1000,
                    'columns_name': ['col1', 'col2']
                },
                'classification': {
                    'domain_category': 'Testing',
                    'tags': ['test']
                }
            }]
        }

        # Mock download response
        download_response = Mock()
        download_response.status_code = 200
        download_response.json.return_value = [
            {'col1': 'test1', 'col2': 10},
            {'col1': 'test2', 'col2': 20},
            {'col1': 'test3', 'col2': 30}
        ]

        # Configure mock to return different responses based on URL
        def side_effect(url, **kwargs):
            if 'catalog' in url:
                return search_response
            else:
                return download_response

        mock_get.side_effect = side_effect

        # Run workflow
        with tempfile.TemporaryDirectory() as temp_dir:
            scout = ScoutDataDiscovery(
                config=self.test_config,
                cache_dir=temp_dir,
                log_level="ERROR"
            )

            # Search datasets
            datasets = scout.search_datasets("test")
            self.assertEqual(len(datasets), 1)

            # Assess quality
            assessment = scout.assess_dataset_quality(
                datasets.iloc[0]['id'],
                metadata=datasets.iloc[0].to_dict()
            )
            self.assertNotIn('error', assessment)

            # Generate recommendations (should return empty since only one dataset)
            recommendations = scout.generate_recommendations(
                datasets.iloc[0]['id'],
                datasets
            )
            self.assertTrue(recommendations.empty)  # Only one dataset, no recommendations


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestScoutDataDiscovery,
        TestDataQualityAssessor,
        TestConfigManager,
        TestIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)