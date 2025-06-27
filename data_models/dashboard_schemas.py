"""
Pydantic models for the EOTS v2.5 Dashboard and UI components.

This module defines the data structures used for dashboard configurations,
UI components, and display settings across the EOTS platform.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict

class DashboardModeType(str, Enum):
    """Defines the different dashboard modes available in the EOTS UI."""
    STANDARD = "standard"; ADVANCED = "advanced"; AI = "ai"; VOLATILITY = "volatility"
    STRUCTURE = "structure"; TIMEDECAY = "timedecay"; CUSTOM = "custom"

# DashboardModeSettings is defined in configuration_schemas.py
# ChartType, ChartLayoutConfigV2_5, ControlPanelParametersV2_5 moved to configuration_schemas.py

class DashboardConfigV2_5(BaseModel):
    """
    Main configuration for the EOTS dashboard, defining modes, charts, and component behavior.
    Note: Static setup aspects (available modes, default chart layouts, default control panel settings)
    are primarily configured via EOTSConfigV2_5.visualization_settings.
    This model can be used by the dashboard application for its internal state management,
    potentially loading initial values from EOTSConfigV2_5.visualization_settings.
    """
    available_modes: Dict[DashboardModeType, Any] = Field(default_factory=dict, description="Runtime collection of available dashboard modes and their settings, potentially derived from EOTSConfigV2_5.visualization_settings.modes_detail_config. Value type can be dashboard_schemas.DashboardModeSettings or configuration_schemas.DashboardModeSettings depending on source.")
    default_mode: Optional[DashboardModeType] = Field(None, description="Default dashboard mode loaded on startup, potentially derived from EOTSConfigV2_5.visualization_settings.default_mode_label.")
    chart_configs: Dict[str, Any] = Field(default_factory=dict, description="Runtime configuration for individual charts by their unique identifier, potentially derived from EOTSConfigV2_5.visualization_settings.default_chart_layouts. Value type can be configuration_schemas.ChartLayoutConfigV2_5.")
    control_panel: Optional[Any] = Field(None, description="Runtime parameters for the control panel, potentially derived from EOTSConfigV2_5.visualization_settings.default_control_panel_settings. Value type can be configuration_schemas.ControlPanelParametersV2_5.")
    components_respecting_filters: List[str] = Field(default_factory=list, description="Components that respect global filters (e.g., symbol, time frame).")
    components_not_respecting_filters: List[str] = Field(default_factory=list, description="Components that ignore global filters.")
    model_config = ConfigDict(extra='forbid')

class ComponentComplianceV2_5(BaseModel):
    """Tracks compliance and performance metrics for dashboard components."""
    component_id: str = Field(..., description="Unique identifier for the dashboard component.")
    respects_filters: bool = Field(True, description="Whether this component respects global filters.")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics (e.g., load_time, render_time).")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of last update (UTC).")
    model_config = ConfigDict(extra='forbid')

class DashboardStateV2_5(BaseModel):
    """Tracks the current state and user interactions with the dashboard UI."""
    current_mode: Optional[DashboardModeType] = Field(None, description="Currently active dashboard mode.")
    active_components: List[str] = Field(default_factory=list, description="List of currently active component IDs.")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User-specific UI preferences and settings.")
    last_interaction: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of last user interaction (UTC).")
    model_config = ConfigDict(extra='forbid')
