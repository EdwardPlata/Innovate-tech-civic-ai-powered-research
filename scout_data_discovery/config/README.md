# Configuration System Documentation

## Overview

The `config/` directory contains the comprehensive configuration management system for Scout Data Discovery. This system provides hierarchical configuration loading, validation, and environment-specific customization capabilities.

## Configuration Architecture

### File Structure
```
config/
├── README.md              # This documentation
├── default_config.yaml    # Production-ready default settings
└── config_manager.py      # Configuration loading and management engine
```

### Configuration Hierarchy

The system loads configuration in the following priority order (later sources override earlier ones):

1. **Default Configuration** (`default_config.yaml`)
2. **User Configuration File** (optional, specified at runtime)
3. **Environment Variables** (runtime overrides with `SCOUT_` prefix)

---

## Configuration Files

### default_config.yaml
**Production-Ready Default Configuration**

This file contains comprehensive default settings optimized for production use. All configuration sections are documented with comments explaining their purpose and recommended values.

#### Configuration Sections:

**API Settings**
```yaml
api:
  rate_limit_delay: 0.5    # Seconds between requests
  request_timeout: 30      # Request timeout in seconds
  retry_attempts: 3        # Number of retry attempts
  max_concurrent_requests: 5  # Parallel request limit
```

**Caching Configuration**
```yaml
cache:
  duration_hours: 24       # Cache TTL in hours
  directory: "cache"       # Cache directory path
  enable_file_cache: true  # Enable file-based caching
  enable_memory_cache: true # Enable in-memory caching
```

**Data Processing Settings**
```yaml
data:
  default_sample_size: 1000    # Default sample size for analysis
  max_search_results: 100      # Maximum search results
  quality_threshold: 70        # Minimum quality score (0-100)
  max_parallel_assessments: 20 # Parallel quality assessments
```

**Supported Data Portals**
```yaml
portals:
  default_domains:
    - "data.cityofnewyork.us"
    - "data.cityofchicago.org"
    - "data.seattle.gov"
    - "data.sfgov.org"

  socrata:
    discovery_api_url: "http://api.us.socrata.com/api/catalog/v1"
    resource_base_urls:
      "data.cityofnewyork.us": "https://data.cityofnewyork.us/resource"
```

**Quality Assessment Configuration**
```yaml
quality:
  weights:
    completeness: 0.25    # Completeness dimension weight
    consistency: 0.20     # Consistency dimension weight
    accuracy: 0.20        # Accuracy dimension weight
    timeliness: 0.15      # Timeliness dimension weight
    usability: 0.20       # Usability dimension weight

  thresholds:
    high_quality: 90      # High quality threshold
    medium_quality: 70    # Medium quality threshold
    low_quality: 50       # Low quality threshold
```

**Recommendation System**
```yaml
recommendations:
  similarity_weights:
    category_match: 0.4        # Category matching weight
    tag_similarity: 0.3        # Tag similarity weight
    column_similarity: 0.2     # Column structure similarity
    description_similarity: 0.1 # Description similarity weight

  max_recommendations_per_dataset: 10
  min_similarity_score: 0.1
```

**Logging Configuration**
```yaml
logging:
  level: "INFO"              # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  enable_file_logging: false
  log_file: "logs/scout_discovery.log"
```

**Export Settings**
```yaml
export:
  default_directory: "exports"
  formats:
    - "csv"
    - "json"
    - "parquet"
  include_metadata: true
  include_summary_report: true
```

**Performance Optimization**
```yaml
performance:
  enable_parallel_processing: true
  max_memory_usage_mb: 1000
  chunk_size_for_large_datasets: 10000
```

---

## Configuration Manager

### config_manager.py
**Advanced Configuration Loading and Management**

The `ConfigManager` class provides sophisticated configuration management with validation, environment variable support, and hierarchical merging.

#### Key Features:

**Hierarchical Loading**
```python
config_manager = ConfigManager('user_config.yaml')
# Loads: default_config.yaml → user_config.yaml → environment variables
```

**Environment Variable Overrides**
Environment variables use the format: `SCOUT_<SECTION>_<KEY>`
```bash
export SCOUT_API_RATE_LIMIT_DELAY=1.0
export SCOUT_DATA_QUALITY_THRESHOLD=80
export SCOUT_CACHE_DURATION_HOURS=48
```

**Configuration Validation**
- Required section validation
- Value range checking
- Type validation
- Consistency checks (e.g., quality weights sum to 1.0)

**Dynamic Configuration**
```python
# Get configuration values with dot notation
delay = config.get('api.rate_limit_delay', default=0.5)

# Set configuration values
config.set('data.quality_threshold', 85)

# Get entire sections
api_config = config.get_section('api')
```

#### Configuration Profiles

Create environment-specific configurations:

```python
# Development profile
dev_config = config.create_profile('development', {
    'api': {'rate_limit_delay': 0.1},
    'data': {'default_sample_size': 100},
    'logging': {'level': 'DEBUG'}
})

# Production profile
prod_config = config.create_profile('production', {
    'api': {'rate_limit_delay': 2.0},
    'data': {'default_sample_size': 5000},
    'cache': {'duration_hours': 72}
})
```

#### Configuration Persistence
```python
# Save current configuration
config.save_user_config('my_config.yaml', format='yaml')
config.save_user_config('my_config.json', format='json')

# Load saved configuration
config = ConfigManager('my_config.yaml')
```

---

## Usage Examples

### Basic Configuration
```python
from scout_data_discovery.config import ConfigManager

# Use default configuration
config = ConfigManager()

# Access configuration values
rate_limit = config.get('api.rate_limit_delay')
quality_threshold = config.get('data.quality_threshold')
```

### Custom Configuration
```python
# Load custom configuration file
config = ConfigManager('path/to/custom_config.yaml')

# Use with Scout
from scout_data_discovery import ScoutDataDiscovery

scout = ScoutDataDiscovery(config=config.to_dict())
```

### Environment-Specific Setup
```bash
# Set environment variables
export SCOUT_API_RATE_LIMIT_DELAY=2.0
export SCOUT_DATA_QUALITY_THRESHOLD=85
export SCOUT_LOGGING_LEVEL=DEBUG

# Configuration automatically includes environment overrides
python your_script.py
```

### Runtime Configuration Updates
```python
config = ConfigManager()

# Update configuration at runtime
config.update_from_dict({
    'api': {'rate_limit_delay': 1.5},
    'data': {'quality_threshold': 80}
})

# Get performance-specific configuration
perf_config = config.get_performance_config()
quality_config = config.get_quality_config()
```

---

## Configuration Validation

The system performs comprehensive validation on all configuration values:

### Required Sections
- `api`: API client configuration
- `cache`: Caching system settings
- `data`: Data processing parameters
- `portals`: Supported data portal configurations
- `quality`: Quality assessment settings

### Value Validation
- **Rate Limits**: Must be non-negative
- **Timeouts**: Must be at least 1 second
- **Quality Thresholds**: Must be between 0-100
- **Quality Weights**: Must sum to 1.0
- **Memory Limits**: Must be positive integers

### Type Validation
- Automatic type conversion from environment variables
- Boolean parsing for true/false strings
- JSON parsing for complex structures
- Numeric validation with range checking

---

## Best Practices

### Development Configuration
```yaml
# development_config.yaml
api:
  rate_limit_delay: 0.1      # Faster for development
  retry_attempts: 1          # Fail fast

data:
  default_sample_size: 100   # Smaller samples for speed

logging:
  level: "DEBUG"             # Detailed logging
  enable_file_logging: true
```

### Production Configuration
```yaml
# production_config.yaml
api:
  rate_limit_delay: 2.0      # Respectful API usage
  retry_attempts: 5          # Robust error handling

data:
  default_sample_size: 10000 # Comprehensive analysis

cache:
  duration_hours: 72         # Longer caching

logging:
  level: "INFO"              # Appropriate verbosity
  enable_file_logging: true
  log_file: "/var/log/scout/discovery.log"
```

### Security Configuration
```yaml
# security_config.yaml
api:
  user_agent: "YourApp/1.0 (contact@yourcompany.com)"

security:
  respect_robots_txt: true
  max_request_rate_per_minute: 60
```

---

## Configuration Schema

### Complete Schema Reference

```yaml
# API Configuration
api:
  rate_limit_delay: float      # Seconds between requests (≥0)
  request_timeout: int         # Request timeout in seconds (≥1)
  retry_attempts: int          # Number of retry attempts (≥1)
  max_concurrent_requests: int # Parallel request limit (≥1)

# Caching Configuration
cache:
  duration_hours: int          # Cache TTL in hours (≥0)
  directory: str              # Cache directory path
  enable_file_cache: bool     # Enable file-based caching
  enable_memory_cache: bool   # Enable in-memory caching

# Data Processing Configuration
data:
  default_sample_size: int     # Default sample size (≥1)
  max_search_results: int      # Maximum search results (≥1)
  quality_threshold: float     # Quality threshold 0-100
  max_parallel_assessments: int # Parallel assessments (≥1)

# Portal Configuration
portals:
  default_domains: [str]       # List of default domains
  socrata:
    discovery_api_url: str     # Socrata Discovery API URL
    resource_base_urls: {str: str} # Domain to base URL mapping

# Quality Assessment Configuration
quality:
  weights:
    completeness: float        # Completeness weight (sum to 1.0)
    consistency: float         # Consistency weight
    accuracy: float           # Accuracy weight
    timeliness: float         # Timeliness weight
    usability: float          # Usability weight
  thresholds:
    high_quality: float       # High quality threshold
    medium_quality: float     # Medium quality threshold
    low_quality: float        # Low quality threshold

# Recommendation Configuration
recommendations:
  similarity_weights:
    category_match: float      # Category matching weight
    tag_similarity: float      # Tag similarity weight
    column_similarity: float   # Column similarity weight
    description_similarity: float # Description similarity weight
  max_recommendations_per_dataset: int
  min_similarity_score: float

# Logging Configuration
logging:
  level: str                  # Log level: DEBUG, INFO, WARNING, ERROR
  format: str                 # Log format string
  enable_file_logging: bool   # Enable file logging
  log_file: str              # Log file path

# Export Configuration
export:
  default_directory: str      # Default export directory
  formats: [str]             # Supported export formats
  include_metadata: bool      # Include metadata in exports
  include_summary_report: bool # Include summary reports

# Performance Configuration
performance:
  enable_parallel_processing: bool  # Enable parallel processing
  max_memory_usage_mb: int         # Memory usage limit in MB
  chunk_size_for_large_datasets: int # Chunk size for large datasets

# Security Configuration
security:
  user_agent: str            # User agent string
  respect_robots_txt: bool   # Respect robots.txt
  max_request_rate_per_minute: int # Rate limiting
```

---

## Troubleshooting

### Common Configuration Issues

**Configuration File Not Found**
```python
# Solution: Use absolute paths or check working directory
config = ConfigManager('/absolute/path/to/config.yaml')
```

**Environment Variables Not Working**
```bash
# Ensure correct format: SCOUT_<SECTION>_<KEY>
export SCOUT_API_RATE_LIMIT_DELAY=1.0  # Correct
export API_RATE_LIMIT_DELAY=1.0         # Incorrect
```

**Configuration Validation Errors**
```
ConfigurationError: Quality weights must sum to 1.0, got 1.2
```
Solution: Check that quality assessment weights sum exactly to 1.0

**Performance Issues**
- Increase `max_concurrent_requests` for faster processing
- Adjust `chunk_size_for_large_datasets` based on available memory
- Enable `enable_parallel_processing` for multi-core systems

This configuration system provides robust, flexible, and production-ready configuration management for all aspects of the Scout Data Discovery system.