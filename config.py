"""
Configuration file for NYC Open Data analytics application
"""

# NYC Open Data datasets to collect
DATASETS = {
    # Traffic and Transportation
    "motor_vehicle_collisions": {
        "dataset_id": "h9gi-nx95",
        "name": "Motor Vehicle Collisions - Crashes",
        "update_frequency": "daily"
    },
    "traffic_volume": {
        "dataset_id": "btm5-ppia",
        "name": "Traffic Volume Counts",
        "update_frequency": "monthly"
    },

    # Crime and Safety
    "nypd_complaints": {
        "dataset_id": "qgea-i56i",
        "name": "NYPD Complaint Data Current (Year To Date)",
        "update_frequency": "daily"
    },
    "nypd_arrests": {
        "dataset_id": "uip8-fykc",
        "name": "NYPD Arrests Data (Year to Date)",
        "update_frequency": "daily"
    },

    # Housing and Development
    "housing_violations": {
        "dataset_id": "wvxf-dwi5",
        "name": "Housing Maintenance Code Violations",
        "update_frequency": "daily"
    },
    "building_permits": {
        "dataset_id": "ipu4-2q9a",
        "name": "DOB NOW: Build – Approved Applications",
        "update_frequency": "daily"
    },

    # Health and Environment
    "air_quality": {
        "dataset_id": "c3uy-2p5r",
        "name": "Air Quality",
        "update_frequency": "monthly"
    },
    "restaurant_inspections": {
        "dataset_id": "43nn-pn8j",
        "name": "DOHMH New York City Restaurant Inspection Results",
        "update_frequency": "daily"
    }
}

# Data collection settings
DATA_SETTINGS = {
    "batch_size": 10000,
    "rate_limit": 1000,  # requests per hour
    "retry_attempts": 3,
    "timeout": 30  # seconds
}

# Geographic boundaries
NYC_BOROUGHS = [
    "MANHATTAN",
    "BROOKLYN",
    "QUEENS",
    "BRONX",
    "STATEN ISLAND"
]

# Date ranges for historical data
DATE_RANGES = {
    "current_year": "2024-01-01",
    "historical_start": "2020-01-01"
}