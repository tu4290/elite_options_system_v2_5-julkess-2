"""
Pydantic models for the EOTS v2.5 Dashboard and UI components.

This module defines the data structures used for dashboard configurations,
UI components, and display settings across the EOTS platform.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, ConfigDict

class DashboardModeType(str, Enum):
    """Defines the different dashboard modes available in the EOTS UI."""
    STANDARD = "standard"; ADVANCED = "advanced"; AI = "ai"; VOLATILITY = "volatility"
    STRUCTURE = "structure"; TIMEDECAY = "timedecay"; CUSTOM = "custom"

class ChartType(str, Enum):
    """Supported chart types for the dashboard visualizations."""
    LINE = "line"; BAR = "bar"; SCATTER = "scatter"; HEATMAP = "heatmap"; GAUGE = "gauge"
    CANDLESTICK = "candlestick"; HISTOGRAM = "histogram"; PIE = "pie"; TABLE = "table"

class DashboardModeSettings(BaseModel):
    """Configuration for a specific dashboard mode's display settings."""
    module_name: str = Field(..., description="Python module containing the dashboard mode implementation.")
    charts: List[str] = Field(default_factory=list, description="List of chart/component identifiers to display in this mode.")
    label: str = Field("", description="User-friendly display name for this mode.")
    description: str = Field("", description="Description of what this dashboard mode shows.")
    icon: str = Field("", description="Icon identifier for the mode selector.")
    model_config = ConfigDict(extra='forbid')

class ChartLayoutConfigV2_5(BaseModel):
    """Unified chart layout configuration for consistent visualization across EOTS."""
    title_text: str = Field(..., description="Chart title text")
    chart_type: ChartType = Field(..., description="Type of chart to render")
    height: int = Field(300, ge=100, le=1000, description="Chart height in pixels")
    width: Union[int, str] = Field("100%", description="Chart width in pixels or percentage")
    x_axis_title: str = Field("", description="X-axis title")
    y_axis_title: str = Field("", description="Y-axis title")
    show_legend: bool = Field(True, description="Whether to show the legend")
    margin: Dict[str, int] = Field(
        default_factory=lambda: {"t": 30, "b": 30, "l": 50, "r": 30}, 
        description="Chart margins in pixels"
    )
    template: str = Field("plotly_white", description="Plotly template to use")
    model_config = ConfigDict(extra='forbid')

    def to_plotly_layout(self) -> Dict[str, Any]:
        """Convert to Plotly layout dictionary."""
        return {
            "title": {"text": self.title_text},
            "height": self.height,
            "width": self.width if isinstance(self.width, int) else None,
            "xaxis": {"title": self.x_axis_title},
            "yaxis": {"title": self.y_axis_title},
            "showlegend": self.show_legend,
            "margin": self.margin,
            "template": self.template
        }

class ControlPanelParametersV2_5(BaseModel):
    """Control panel parameters with strict validation for dashboard controls."""
    symbol: str = Field(..., description="Trading symbol to analyze (e.g., 'SPY', 'QQQ').")
    time_frame: str = Field("1D", description="Time frame for analysis (e.g., '1D', '1H', '15m').")
    start_date: Optional[datetime] = Field(None, description="Start date for historical data (UTC).")
    end_date: Optional[datetime] = Field(None, description="End date for historical data (UTC).")
    indicators: List[str] = Field(default_factory=list, description="Technical indicators to display (e.g., 'RSI', 'MACD').")
    show_volume: bool = Field(True, description="Toggle volume data display on/off.")
    show_grid: bool = Field(True, description="Toggle grid lines visibility.")
    theme: str = Field("dark", description="UI theme preference ('dark', 'light', or 'system').")
    refresh_interval: int = Field(60, ge=5, description="Auto-refresh interval in seconds (min 5s).")
    layout_columns: int = Field(2, ge=1, le=4, description="Number of columns in dashboard layout (1-4).")
    model_config = ConfigDict(extra='forbid')

class DashboardConfigV2_5(BaseModel):
    """Main configuration for the EOTS dashboard, defining modes, charts, and component behavior."""
    available_modes: Dict[DashboardModeType, DashboardModeSettings] = Field(default_factory=dict, description="Available dashboard modes and their configurations.")
    default_mode: DashboardModeType = Field(DashboardModeType.STANDARD, description="Default dashboard mode loaded on startup.")
    chart_configs: Dict[str, ChartLayoutConfigV2_5] = Field(default_factory=dict, description="Configuration for individual charts by their unique identifier.")
    control_panel: ControlPanelParametersV2_5 = Field(default_factory=ControlPanelParametersV2_5, description="Default parameters for the control panel.")
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
    current_mode: DashboardModeType = Field(DashboardModeType.STANDARD, description="Currently active dashboard mode.")
    active_components: List[str] = Field(default_factory=list, description="List of currently active component IDs.")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User-specific UI preferences and settings.")
    last_interaction: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of last user interaction (UTC).")
    model_config = ConfigDict(extra='forbid')
