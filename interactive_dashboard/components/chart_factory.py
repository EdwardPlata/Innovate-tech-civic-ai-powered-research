"""
Chart Factory Component
Wrapper for the chart factory functionality
"""

from .charts.factory import ChartFactory

# Re-export the ChartFactory class for backward compatibility
__all__ = ['ChartFactory']