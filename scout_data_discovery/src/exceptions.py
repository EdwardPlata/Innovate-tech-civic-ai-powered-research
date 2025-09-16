"""
Custom exceptions for Scout Data Discovery package
"""


class ScoutDiscoveryError(Exception):
    """Base exception for Scout Data Discovery operations"""
    pass


class APIError(ScoutDiscoveryError):
    """Raised when API requests fail"""

    def __init__(self, message: str, status_code: int = None, response_text: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class DataQualityError(ScoutDiscoveryError):
    """Raised when data quality assessment fails"""
    pass


class ConfigurationError(ScoutDiscoveryError):
    """Raised when configuration is invalid"""
    pass


class DataDownloadError(ScoutDiscoveryError):
    """Raised when dataset download fails"""

    def __init__(self, message: str, dataset_id: str = None):
        super().__init__(message)
        self.dataset_id = dataset_id


class SearchError(ScoutDiscoveryError):
    """Raised when dataset search fails"""
    pass


class ValidationError(ScoutDiscoveryError):
    """Raised when data validation fails"""
    pass