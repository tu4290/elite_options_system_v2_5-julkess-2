"""
Compliance Decorators v2.5 - Automatic Component Tracking
==========================================================

This module provides decorators and wrappers for automatic compliance tracking
of dashboard components. It eliminates the need for manual tracking calls
throughout the codebase.

Key Features:
- @track_compliance decorator for component functions
- Automatic data source detection and tracking
- Performance-optimized with minimal overhead
- Integration with ComponentComplianceTracker

Author: EOTS v2.5 Development Team
Version: 2.5.0
"""

import logging
import functools
import inspect
from typing import Any, Callable, Dict, Optional, List
from datetime import datetime

from .component_compliance_tracker_v2_5 import (
    get_compliance_tracker, DataSourceType, track_component_creation, track_data_access
)
from data_models.eots_schemas_v2_5 import (
    FilteredDataBundleV2_5, FinalAnalysisBundleV2_5, ProcessedContractMetricsV2_5,
    ProcessedStrikeLevelMetricsV2_5, ProcessedUnderlyingAggregatesV2_5
)

logger = logging.getLogger(__name__)

def track_compliance(component_id: str, component_name: Optional[str] = None, 
                    auto_detect_data: bool = True, track_return_value: bool = True):
    """
    Decorator to automatically track component compliance.
    
    Args:
        component_id: Unique identifier for the component
        component_name: Human-readable name (defaults to function name)
        auto_detect_data: Whether to automatically detect data sources from arguments
        track_return_value: Whether to analyze the return value for data usage
    
    Usage:
        @track_compliance("unified_intelligence_hub", "Unified AI Intelligence Hub")
        def create_unified_ai_intelligence_hub(bundle_data, ai_settings, symbol):
            # Function implementation
            return html.Div(...)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function name if component_name not provided
            display_name = component_name or func.__name__.replace('_', ' ').title()
            
            # Register component
            metadata = {
                "function_name": func.__name__,
                "module": func.__module__,
                "creation_time": datetime.now().isoformat()
            }
            track_component_creation(component_id, display_name, metadata)
            
            # Auto-detect data sources from arguments if enabled
            if auto_detect_data:
                _detect_and_track_data_sources(component_id, args, kwargs)
            
            # Execute the original function
            try:
                result = func(*args, **kwargs)
                
                # Track return value analysis if enabled
                if track_return_value and result is not None:
                    _analyze_return_value(component_id, result)
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Error in tracked component {component_id}: {e}")
                # Track the error but don't break the component
                track_data_access(component_id, DataSourceType.NO_OPTIONS_DATA, 
                                metadata={"error": str(e), "error_type": type(e).__name__})
                raise
        
        return wrapper
    return decorator

def track_data_bundle_usage(component_id: str):
    """
    Decorator specifically for functions that use data bundles.
    
    Args:
        component_id: Component identifier
    
    Usage:
        @track_data_bundle_usage("performance_chart")
        def create_performance_chart(filtered_bundle, original_bundle):
            # Function implementation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Analyze arguments for data bundle usage
            _analyze_data_bundle_arguments(component_id, func, args, kwargs)
            
            # Execute function
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def track_metrics_usage(metric_names: List[str], component_id: Optional[str] = None):
    """
    Decorator for functions that calculate or use EOTS metrics.
    
    Args:
        metric_names: List of metric names being calculated/used
        component_id: Associated component ID (if any)
    
    Usage:
        @track_metrics_usage(["vapi_fa", "dwfd", "tw_laf"])
        def calculate_eots_metrics(contract_data):
            # Metrics calculation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determine if filtered data is being used
            used_filtered_data = _detect_filtered_data_usage(args, kwargs)
            
            # Track each metric
            tracker = get_compliance_tracker()
            for metric_name in metric_names:
                data_source_type = DataSourceType.METRICS_FILTERED if used_filtered_data else DataSourceType.METRICS_RAW
                tracker.track_metrics_calculation(
                    metric_name=metric_name,
                    used_filtered_data=used_filtered_data,
                    data_source_type=data_source_type,
                    component_id=component_id,
                    metadata={
                        "function_name": func.__name__,
                        "calculation_time": datetime.now().isoformat()
                    }
                )
            
            # Execute function
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def _detect_and_track_data_sources(component_id: str, args: tuple, kwargs: dict) -> None:
    """Detect and track data sources from function arguments."""
    try:
        # Check all arguments for data bundle types
        all_args = list(args) + list(kwargs.values())
        
        for arg in all_args:
            if isinstance(arg, FilteredDataBundleV2_5):
                # Component is using filtered data bundle
                track_data_access(component_id, DataSourceType.FILTERED_OPTIONS, arg,
                                metadata={"source": "filtered_bundle_argument"})
                
                if arg.filtered_options_contracts:
                    track_data_access(component_id, DataSourceType.FILTERED_OPTIONS, 
                                    arg.filtered_options_contracts,
                                    metadata={"contract_count": len(arg.filtered_options_contracts)})
                
                if arg.filtered_strike_data:
                    track_data_access(component_id, DataSourceType.FILTERED_STRIKES,
                                    arg.filtered_strike_data,
                                    metadata={"strike_count": len(arg.filtered_strike_data)})
            
            elif isinstance(arg, FinalAnalysisBundleV2_5):
                # Component is using original data bundle
                if (arg.processed_data_bundle and 
                    arg.processed_data_bundle.options_data_with_metrics):
                    track_data_access(component_id, DataSourceType.RAW_OPTIONS,
                                    arg.processed_data_bundle.options_data_with_metrics,
                                    metadata={"source": "original_bundle_argument",
                                            "contract_count": len(arg.processed_data_bundle.options_data_with_metrics)})
                
                if (arg.processed_data_bundle and 
                    arg.processed_data_bundle.strike_level_data_with_metrics):
                    track_data_access(component_id, DataSourceType.RAW_STRIKES,
                                    arg.processed_data_bundle.strike_level_data_with_metrics,
                                    metadata={"source": "original_bundle_argument",
                                            "strike_count": len(arg.processed_data_bundle.strike_level_data_with_metrics)})
            
            elif isinstance(arg, list):
                # Check if it's a list of contracts or strikes
                if arg and isinstance(arg[0], ProcessedContractMetricsV2_5):
                    # This is likely filtered data if it's a small subset
                    track_data_access(component_id, DataSourceType.FILTERED_OPTIONS, arg,
                                    metadata={"source": "contract_list_argument",
                                            "contract_count": len(arg)})
                
                elif arg and isinstance(arg[0], ProcessedStrikeLevelMetricsV2_5):
                    # This is likely filtered strike data
                    track_data_access(component_id, DataSourceType.FILTERED_STRIKES, arg,
                                    metadata={"source": "strike_list_argument",
                                            "strike_count": len(arg)})
            
            elif isinstance(arg, ProcessedUnderlyingAggregatesV2_5):
                # Underlying data usage (always applicable)
                track_data_access(component_id, DataSourceType.UNDERLYING_DATA, arg,
                                metadata={"source": "underlying_data_argument"})
            
            elif isinstance(arg, dict) and "ai_settings" in str(type(arg)):
                # Check if ai_settings contains filtered_bundle
                if isinstance(arg, dict) and "filtered_bundle" in arg:
                    if arg["filtered_bundle"]:
                        track_data_access(component_id, DataSourceType.FILTERED_OPTIONS,
                                        arg["filtered_bundle"],
                                        metadata={"source": "ai_settings_filtered_bundle"})
    
    except Exception as e:
        logger.debug(f"Error detecting data sources for {component_id}: {e}")

def _analyze_data_bundle_arguments(component_id: str, func: Callable, args: tuple, kwargs: dict) -> None:
    """Analyze function arguments specifically for data bundle usage."""
    try:
        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Check for specific parameter names that indicate data usage
        for param_name, param_value in bound_args.arguments.items():
            if "filtered" in param_name.lower() and isinstance(param_value, FilteredDataBundleV2_5):
                track_data_access(component_id, DataSourceType.FILTERED_OPTIONS, param_value,
                                metadata={"parameter_name": param_name})
            
            elif "bundle" in param_name.lower() and isinstance(param_value, FinalAnalysisBundleV2_5):
                track_data_access(component_id, DataSourceType.RAW_OPTIONS, param_value,
                                metadata={"parameter_name": param_name})
            
            elif "data" in param_name.lower() and isinstance(param_value, list):
                # Analyze list contents
                if param_value and isinstance(param_value[0], ProcessedContractMetricsV2_5):
                    # Determine if this is likely filtered data based on size and context
                    is_filtered = "filtered" in param_name.lower() or len(param_value) < 1000
                    data_source = DataSourceType.FILTERED_OPTIONS if is_filtered else DataSourceType.RAW_OPTIONS
                    track_data_access(component_id, data_source, param_value,
                                    metadata={"parameter_name": param_name, "inferred_filtered": is_filtered})
    
    except Exception as e:
        logger.debug(f"Error analyzing data bundle arguments for {component_id}: {e}")

def _detect_filtered_data_usage(args: tuple, kwargs: dict) -> bool:
    """Detect if function is using filtered data based on arguments."""
    try:
        all_args = list(args) + list(kwargs.values())
        
        # Look for FilteredDataBundleV2_5 instances
        for arg in all_args:
            if isinstance(arg, FilteredDataBundleV2_5):
                return True
            
            # Check for filtered data in nested structures
            if isinstance(arg, dict):
                for value in arg.values():
                    if isinstance(value, FilteredDataBundleV2_5):
                        return True
        
        # Check parameter names for "filtered" keyword
        if isinstance(kwargs, dict):
            for param_name in kwargs.keys():
                if "filtered" in param_name.lower():
                    return True
        
        return False
    
    except Exception:
        return False

def _analyze_return_value(component_id: str, return_value: Any) -> None:
    """Analyze return value for additional compliance insights."""
    try:
        # This could be extended to analyze Dash components for data references
        # For now, just log that the component was successfully created
        track_data_access(component_id, DataSourceType.NO_OPTIONS_DATA, return_value,
                        metadata={"return_type": type(return_value).__name__,
                                "component_created": True})
    
    except Exception as e:
        logger.debug(f"Error analyzing return value for {component_id}: {e}")

# Convenience decorators for common component types
def track_chart_component(component_id: str, chart_name: Optional[str] = None):
    """Decorator specifically for chart/visualization components."""
    return track_compliance(component_id, chart_name or f"{component_id}_chart", 
                          auto_detect_data=True, track_return_value=False)

def track_panel_component(component_id: str, panel_name: Optional[str] = None):
    """Decorator specifically for panel/layout components."""
    return track_compliance(component_id, panel_name or f"{component_id}_panel",
                          auto_detect_data=True, track_return_value=False)

def track_intelligence_component(component_id: str, intelligence_name: Optional[str] = None):
    """Decorator specifically for AI intelligence components."""
    return track_compliance(component_id, intelligence_name or f"{component_id}_intelligence",
                          auto_detect_data=True, track_return_value=False) 