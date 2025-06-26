"""
AI Hub Compliance Manager v2.5 - Pydantic-First Control Panel Validation
========================================================================

This module ensures that the AI Hub dashboard respects control panel parameters
and provides comprehensive validation and compliance reporting.

Key Features:
- Pydantic-first validation of control panel parameters
- Real-time compliance monitoring and reporting
- Data filtering based on user-specified DTE and price ranges
- Comprehensive compliance scoring and recommendations
- Integration with EOTS v2.5 schemas for type safety

Author: EOTS v2.5 Development Team
Version: 2.5.0
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import time

from data_models.eots_schemas_v2_5 import (
    FinalAnalysisBundleV2_5,
    ControlPanelParametersV2_5,
    FilteredDataBundleV2_5,
    AIHubComplianceReportV2_5,
    ProcessedContractMetricsV2_5,
    ProcessedStrikeLevelMetricsV2_5
)

# Import real compliance tracking
from .component_compliance_tracker_v2_5 import get_compliance_tracker

logger = logging.getLogger(__name__)

class AIHubComplianceManager:
    """Pydantic-first compliance manager for AI Hub dashboard."""
    
    def __init__(self):
        """Initialize the compliance manager."""
        self.current_params: Optional[ControlPanelParametersV2_5] = None
        self.last_compliance_check: Optional[datetime] = None
        self.compliance_history: List[AIHubComplianceReportV2_5] = []
    
    def validate_control_panel_params(self, symbol: str, dte_min: int, dte_max: int, 
                                    price_range_percent: int, fetch_interval: int = 30) -> ControlPanelParametersV2_5:
        """
        Validate and create Pydantic model for control panel parameters.
        
        Args:
            symbol: Trading symbol
            dte_min: Minimum DTE
            dte_max: Maximum DTE 
            price_range_percent: Price range percentage
            fetch_interval: Data fetch interval in seconds
            
        Returns:
            ControlPanelParametersV2_5: Validated parameters
            
        Raises:
            ValidationError: If parameters are invalid
        """
        try:
            # Create and validate Pydantic model
            params = ControlPanelParametersV2_5(
                symbol=symbol.upper(),
                dte_min=dte_min,
                dte_max=dte_max,
                price_range_percent=price_range_percent,
                fetch_interval_seconds=fetch_interval
            )
            
            self.current_params = params
            logger.info(f"✅ CONTROL PANEL VALIDATION: {params.get_filter_description()}")
            
            return params
            
        except Exception as e:
            logger.error(f"❌ CONTROL PANEL VALIDATION FAILED: {e}")
            raise
    
    def create_filtered_bundle(self, original_bundle: FinalAnalysisBundleV2_5, 
                             control_params: ControlPanelParametersV2_5) -> FilteredDataBundleV2_5:
        """
        Create a filtered data bundle that respects control panel parameters.
        
        Args:
            original_bundle: Original unfiltered data bundle
            control_params: Validated control panel parameters
            
        Returns:
            FilteredDataBundleV2_5: Filtered data bundle with compliance report
        """
        logger.info(f"🔍 FILTERING DATA: Applying filters {control_params.get_filter_description()}")
        
        # Create filtered bundle using Pydantic class method
        filtered_bundle = FilteredDataBundleV2_5.create_from_bundle(original_bundle, control_params)
        
        # 🚀 REAL COMPLIANCE TRACKING: Set up compliance tracker with real data
        tracker = get_compliance_tracker()
        tracker.set_control_parameters(control_params, filtered_bundle)
        
        # Update compliance report with REAL tracking data instead of fake empty lists
        real_compliance_summary = tracker.get_compliance_summary()
        
        # Update the compliance report with real data
        filtered_bundle.compliance_report.components_respecting_filters = real_compliance_summary["components_respecting_filters"]
        filtered_bundle.compliance_report.components_not_respecting_filters = real_compliance_summary["components_not_respecting_filters"]
        filtered_bundle.compliance_report.metrics_calculated_with_filters = real_compliance_summary["metrics_calculated_with_filters"]
        filtered_bundle.compliance_report.metrics_using_raw_data = real_compliance_summary["metrics_using_raw_data"]
        
        # Recalculate compliance score with real data
        filtered_bundle.compliance_report.calculate_compliance_score()
        
        # Log filtering results
        logger.info(f"📊 FILTER RESULTS: {filtered_bundle.get_filter_summary()}")
        logger.info(f"✅ REAL COMPLIANCE SCORE: {filtered_bundle.compliance_report.compliance_score:.2f} "
                   f"({filtered_bundle.compliance_report.compliance_status})")
        logger.info(f"🎯 REAL COMPONENT TRACKING: {len(real_compliance_summary['components_respecting_filters'])} compliant, "
                   f"{len(real_compliance_summary['components_not_respecting_filters'])} non-compliant")
        
        # Store compliance report in history
        self.compliance_history.append(filtered_bundle.compliance_report)
        self.last_compliance_check = datetime.now()
        
        return filtered_bundle
    
    def validate_component_compliance(self, component_name: str, data_used: Any, 
                                    control_params: ControlPanelParametersV2_5) -> bool:
        """
        Validate that a dashboard component is using filtered data.
        
        Args:
            component_name: Name of the dashboard component
            data_used: Data being used by the component
            control_params: Control panel parameters to check against
            
        Returns:
            bool: True if component is compliant, False otherwise
        """
        try:
            # Check if data_used is filtered data or raw data
            if isinstance(data_used, list):
                # For contract lists, check if DTE and price filters were applied
                if len(data_used) > 0 and hasattr(data_used[0], 'dte_calc'):
                    # Check DTE compliance
                    for item in data_used[:10]:  # Sample first 10 items
                        if hasattr(item, 'dte_calc') and item.dte_calc is not None:
                            if not (control_params.dte_min <= item.dte_calc <= control_params.dte_max):
                                logger.warning(f"⚠️ COMPLIANCE VIOLATION: {component_name} using unfiltered DTE data")
                                return False
                        
                        if hasattr(item, 'strike') and item.strike is not None:
                            # Would need underlying price to check price range compliance
                            pass
            
            logger.debug(f"✅ COMPONENT COMPLIANCE: {component_name} appears compliant")
            return True
            
        except Exception as e:
            logger.error(f"❌ COMPLIANCE CHECK ERROR: {component_name} - {e}")
            return False
    
    def generate_compliance_display_info(self, filtered_bundle: FilteredDataBundleV2_5) -> Dict[str, Any]:
        """
        Generate display information for compliance status.
        
        Args:
            filtered_bundle: Filtered data bundle with compliance report
            
        Returns:
            Dict[str, Any]: Display information for compliance status
        """
        report = filtered_bundle.compliance_report
        params = filtered_bundle.applied_filters
        
        # Determine status color
        if report.compliance_score >= 0.9:
            status_color = "#28a745"  # Green
            status_icon = "✅"
        elif report.compliance_score >= 0.7:
            status_color = "#ffc107"  # Yellow
            status_icon = "⚠️"
        else:
            status_color = "#dc3545"  # Red
            status_icon = "❌"
        
        return {
            "status_text": f"{status_icon} AI Hub Compliance: {report.compliance_status}",
            "status_color": status_color,
            "compliance_score": report.compliance_score,
            "filter_description": params.get_filter_description(),
            "filter_summary": filtered_bundle.get_filter_summary(),
            "contracts_filtered": f"{filtered_bundle.filtered_contracts_count:,} contracts (from {filtered_bundle.original_contracts_count:,})",
            "filter_efficiency": f"{(filtered_bundle.filtered_contracts_count / filtered_bundle.original_contracts_count * 100):.1f}%" if filtered_bundle.original_contracts_count > 0 else "0%",
            "processing_time": f"{filtered_bundle.filter_duration_ms:.1f}ms",
            "issues_count": len(report.compliance_issues),
            "recommendations_count": len(report.compliance_recommendations),
            "last_check": report.report_timestamp.strftime("%H:%M:%S"),
            "components_compliant": len(report.components_respecting_filters),
            "components_non_compliant": len(report.components_not_respecting_filters)
        }
    
    def create_compliance_status_card(self, filtered_bundle: FilteredDataBundleV2_5) -> Dict[str, Any]:
        """
        Create a compliance status card for display in the AI Hub.
        
        Args:
            filtered_bundle: Filtered data bundle with compliance report
            
        Returns:
            Dict[str, Any]: Card configuration for compliance display
        """
        display_info = self.generate_compliance_display_info(filtered_bundle)
        
        return {
            "title": "🎯 Control Panel Compliance",
            "status": display_info["status_text"],
            "status_color": display_info["status_color"],
            "metrics": [
                {"label": "Compliance Score", "value": f"{display_info['compliance_score']:.1%}", "color": display_info["status_color"]},
                {"label": "Active Filters", "value": display_info["filter_description"], "color": "#6c757d"},
                {"label": "Data Filtered", "value": display_info["contracts_filtered"], "color": "#17a2b8"},
                {"label": "Filter Efficiency", "value": display_info["filter_efficiency"], "color": "#28a745"},
                {"label": "Processing Time", "value": display_info["processing_time"], "color": "#6f42c1"}
            ],
            "details": {
                "last_check": display_info["last_check"],
                "issues": display_info["issues_count"],
                "recommendations": display_info["recommendations_count"],
                "components_status": f"{display_info['components_compliant']} compliant, {display_info['components_non_compliant']} non-compliant"
            }
        }
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of compliance status and history.
        
        Returns:
            Dict[str, Any]: Compliance summary
        """
        if not self.compliance_history:
            return {
                "status": "NO_DATA",
                "message": "No compliance checks performed yet",
                "last_check": None,
                "average_score": 0.0
            }
        
        recent_report = self.compliance_history[-1]
        avg_score = sum(r.compliance_score for r in self.compliance_history) / len(self.compliance_history)
        
        return {
            "status": recent_report.compliance_status,
            "current_score": recent_report.compliance_score,
            "average_score": avg_score,
            "last_check": recent_report.report_timestamp,
            "total_checks": len(self.compliance_history),
            "current_filters": self.current_params.get_filter_description() if self.current_params else "None",
            "trend": "IMPROVING" if len(self.compliance_history) > 1 and recent_report.compliance_score > self.compliance_history[-2].compliance_score else "STABLE"
        }

# Global compliance manager instance
_compliance_manager = AIHubComplianceManager()

def get_compliance_manager() -> AIHubComplianceManager:
    """Get the global compliance manager instance."""
    return _compliance_manager

def validate_ai_hub_control_panel(symbol: str, dte_min: int, dte_max: int, 
                                price_range_percent: int, fetch_interval: int = 30) -> ControlPanelParametersV2_5:
    """
    Convenience function to validate control panel parameters.
    
    Args:
        symbol: Trading symbol
        dte_min: Minimum DTE
        dte_max: Maximum DTE
        price_range_percent: Price range percentage
        fetch_interval: Data fetch interval
        
    Returns:
        ControlPanelParametersV2_5: Validated parameters
    """
    return get_compliance_manager().validate_control_panel_params(
        symbol, dte_min, dte_max, price_range_percent, fetch_interval
    )

def create_compliant_ai_hub_data(original_bundle: FinalAnalysisBundleV2_5, 
                               symbol: str, dte_min: int, dte_max: int, 
                               price_range_percent: int) -> FilteredDataBundleV2_5:
    """
    Create a compliant filtered data bundle for the AI Hub.
    
    Args:
        original_bundle: Original data bundle
        symbol: Trading symbol
        dte_min: Minimum DTE
        dte_max: Maximum DTE
        price_range_percent: Price range percentage
        
    Returns:
        FilteredDataBundleV2_5: Filtered and compliant data bundle
    """
    # Validate control panel parameters
    control_params = validate_ai_hub_control_panel(symbol, dte_min, dte_max, price_range_percent)
    
    # Create filtered bundle
    return get_compliance_manager().create_filtered_bundle(original_bundle, control_params) 