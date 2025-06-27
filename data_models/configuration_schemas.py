"""
Pydantic models defining the structure for various configuration sections
of the EOTS v2.5 system, including system settings, dashboard modes,
coefficients, and parameters for different analytical components.
These models are primarily used by ConfigManagerV2_5 to validate and provide
access to the system's configuration.
"""
from pydantic import BaseModel, Field, FilePath
from typing import List, Dict, Any, Optional

# --- TimeOfDayDefinitions moved to context_schemas.py as it's used by TickerContextAnalyzer ---
# from .context_schemas import TimeOfDayDefinitions (if it were here)

# --- Dashboard Specific Configuration Models ---
class DashboardModeSettings(BaseModel):
    """Defines settings for a single dashboard mode."""
    label: str = Field(..., description="Display label for the mode in UI selectors.")
    module_name: str = Field(..., description="Python module name (e.g., 'main_dashboard_display_v2_5') to import for this mode's layout and callbacks.")
    charts: List[str] = Field(default_factory=list, description="List of chart/component identifier names expected to be displayed in this mode.")
    
    class Config:
        extra = 'forbid'


class MainDashboardDisplaySettings(BaseModel):
    """Settings specific to components on the main dashboard display."""
    regime_indicator: Dict[str, Any] = Field(default_factory=lambda: {
        "title": "Market Regime",
        "regime_colors": {"default": "secondary", "bullish": "success", "bearish": "danger", "neutral": "info", "unclear": "warning"}
    }, description="Configuration for the Market Regime indicator display.")
    
    flow_gauge: Dict[str, Any] = Field(default_factory=lambda: {
        "height": 200, "indicator_font_size": 16, "number_font_size": 24, "axis_range": [-3, 3],
        "threshold_line_color": "white", "margin": {"t": 60, "b": 40, "l": 20, "r": 20},
        "steps": [
            {"range": [-3, -2], "color": "#d62728"}, {"range": [-2, -0.5], "color": "#ff9896"},
            {"range": [-0.5, 0.5], "color": "#aec7e8"}, {"range": [0.5, 2], "color": "#98df8a"},
            {"range": [2, 3], "color": "#2ca02c"}
        ]
    }, description="Configuration for flow gauge visualizations.")
    
    gib_gauge: Dict[str, Any] = Field(default_factory=lambda: {
        "height": 180, "indicator_font_size": 14, "number_font_size": 20, "axis_range": [-1, 1],
        "dollar_axis_range": [-1000000, 1000000], "threshold_line_color": "white",
        "margin": {"t": 50, "b": 30, "l": 15, "r": 15},
        "steps": [
            {"range": [-1, -0.5], "color": "#d62728"}, {"range": [-0.5, -0.1], "color": "#ff9896"},
            {"range": [-0.1, 0.1], "color": "#aec7e8"}, {"range": [0.1, 0.5], "color": "#98df8a"},
            {"range": [0.5, 1], "color": "#2ca02c"}
        ],
        "dollar_steps": [
            {"range": [-1000000, -500000], "color": "#d62728"}, {"range": [-500000, -100000], "color": "#ff9896"},
            {"range": [-100000, 100000], "color": "#aec7e8"}, {"range": [100000, 500000], "color": "#98df8a"},
            {"range": [500000, 1000000], "color": "#2ca02c"}
        ]
    }, description="Configuration for GIB gauge visualizations.")
    
    mini_heatmap: Dict[str, Any] = Field(default_factory=lambda: {
        "height": 150, "colorscale": "RdYlGn", "margin": {"t": 50, "b": 30, "l": 40, "r": 40}
    }, description="Default settings for mini-heatmap components.")
    
    recommendations_table: Dict[str, Any] = Field(default_factory=lambda: {
        "title": "ATIF Recommendations", "max_rationale_length": 50, "page_size": 5,
        "style_cell": {"textAlign": "left", "padding": "5px", "minWidth": "80px", "width": "auto", "maxWidth": "200px"},
        "style_header": {"backgroundColor": "rgb(30, 30, 30)", "fontWeight": "bold", "color": "white"},
        "style_data": {"backgroundColor": "rgb(50, 50, 50)", "color": "white"}
    }, description="Configuration for the ATIF recommendations table.")
    
    ticker_context: Dict[str, Any] = Field(default_factory=lambda: {"title": "Ticker Context"}, description="Settings for ticker context display area.")
    
    class Config:
        extra = 'forbid'


class DashboardModeCollection(BaseModel):
    """Defines the collection of all available dashboard modes."""
    main: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Main Dashboard", module_name="main_dashboard_display_v2_5",
        charts=["regime_display", "flow_gauges", "gib_gauges", "recommendations_table"]
    ))
    flow: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Flow Analysis", module_name="flow_mode_display_v2_5",
        charts=["net_value_heatmap_viz", "net_cust_delta_flow_viz", "net_cust_gamma_flow_viz", "net_cust_vega_flow_viz"]
    ))
    # ... other default modes from original schema (structure, timedecay, advanced)
    structure: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Structure & Positioning", module_name="structure_mode_display_v2_5", 
        charts=["mspi_components", "sai_ssi_displays"]
    ))
    timedecay: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Time Decay & Pinning", module_name="time_decay_mode_display_v2_5",
        charts=["tdpi_displays", "vci_strike_charts"]
    ))
    advanced: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Advanced Flow Metrics", module_name="advanced_flow_mode_display_v2_5",
        charts=["vapi_gauges", "dwfd_gauges", "tw_laf_gauges"]
    ))
    
    class Config:
        extra = 'forbid'


class VisualizationSettings(BaseModel):
    """Overall visualization and dashboard settings."""
    dashboard_refresh_interval_seconds: int = Field(60, ge=10, description="Interval in seconds between automatic dashboard data refreshes.")
    max_table_rows_signals_insights: int = Field(10, ge=1, description="Maximum number of rows to display in signals and insights tables on the dashboard.")
    dashboard: Dict[str, Any] = Field(default_factory=lambda: { # Basic dashboard server settings
        "host": "127.0.0.1", "port": 8050, "debug": False, "auto_refresh_seconds": 30,
        "timestamp_format": "%Y-%m-%d %H:%M:%S %Z"
    }, description="Core Dash server and display settings.")
    modes_detail_config: DashboardModeCollection = Field(default_factory=DashboardModeCollection, description="Detailed configuration for each dashboard mode.")
    main_dashboard_settings: MainDashboardDisplaySettings = Field(default_factory=MainDashboardDisplaySettings, description="Specific settings for components on the main dashboard.")

    class Config:
        extra = 'forbid'


# --- Metric Calculation Coefficients ---
class DagAlphaCoeffs(BaseModel):
    """Coefficients for A-DAG calculation based on flow alignment."""
    aligned: float = Field(default=1.35, description="Coefficient when market flow aligns with OI structure.")
    opposed: float = Field(default=0.65, description="Coefficient when market flow opposes OI structure.")
    neutral: float = Field(default=1.0, description="Coefficient when market flow is neutral to OI structure.")
    class Config: extra = 'forbid'

class TdpiBetaCoeffs(BaseModel):
    """Coefficients for D-TDPI calculation based on Charm flow alignment."""
    aligned: float = Field(default=1.35, description="Coefficient for aligned Charm flow.")
    opposed: float = Field(default=0.65, description="Coefficient for opposed Charm flow.")
    neutral: float = Field(default=1.0, description="Coefficient for neutral Charm flow.")
    class Config: extra = 'forbid'

class VriGammaCoeffs(BaseModel): # Note: "Gamma" here refers to vri_gamma, not option gamma
    """Coefficients for VRI 2.0 related to Vanna flow proxy alignment."""
    aligned: float = Field(default=1.35, description="Coefficient for aligned Vanna flow proxy.")
    opposed: float = Field(default=0.65, description="Coefficient for opposed Vanna flow proxy.")
    neutral: float = Field(default=1.0, description="Coefficient for neutral Vanna flow proxy.")
    class Config: extra = 'forbid'

class CoefficientsSettings(BaseModel):
    """Container for various metric calculation coefficients."""
    dag_alpha: DagAlphaCoeffs = Field(default_factory=DagAlphaCoeffs, description="A-DAG alpha coefficients.")
    tdpi_beta: TdpiBetaCoeffs = Field(default_factory=TdpiBetaCoeffs, description="D-TDPI beta coefficients.")
    vri_gamma: VriGammaCoeffs = Field(default_factory=VriGammaCoeffs, description="VRI 2.0 gamma (vri_gamma) coefficients for Vanna flow.")
    class Config: extra = 'forbid'


# --- Data Processor Settings ---
class DataProcessorSettings(BaseModel):
    """Settings for the InitialDataProcessorV2_5, including factors and coefficients."""
    factors: Dict[str, Any] = Field(default_factory=dict, description="Various numerical factors used in metric calculations (e.g., tdpi_gaussian_width).")
    coefficients: CoefficientsSettings = Field(default_factory=CoefficientsSettings, description="Collection of Greek interaction coefficients.")
    iv_context_parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for IV contextualization (e.g., vol_trend_avg_days).")
    class Config: extra = 'forbid'


# --- Market Regime Engine Settings ---
class MarketRegimeEngineSettings(BaseModel):
    """Configuration for the MarketRegimeEngineV2_5."""
    default_regime: str = Field(default="REGIME_UNCLEAR_OR_TRANSITIONING", description="Default market regime if no other rules match.")
    regime_evaluation_order: List[str] = Field(default_factory=list, description="Order in which to evaluate market regime rules (most specific first).")
    regime_rules: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Dictionary of rules defining conditions for each market regime.")
    # TimeOfDayDefinitions is now in context_schemas.py and imported there if needed by MRE logic.
    class Config: extra = 'forbid'


# --- System, Fetcher, Data Management Settings ---
class SystemSettings(BaseModel):
    """General system-level settings for EOTS v2.5."""
    project_root_override: Optional[str] = Field(None, description="Absolute path to override auto-detected project root. Use null for auto-detection.")
    logging_level: str = Field("INFO", description="Minimum logging level (e.g., DEBUG, INFO, WARNING, ERROR).")
    log_to_file: bool = Field(True, description="If true, logs will be written to the file specified in log_file_path.")
    log_file_path: str = Field("logs/eots_v2_5.log", description="Relative path from project root for the log file.") # pattern="\\.log$" removed for simplicity
    max_log_file_size_bytes: int = Field(10485760, ge=1024, description="Maximum size of a single log file in bytes before rotation.")
    backup_log_count: int = Field(5, ge=0, description="Number of old log files to keep after rotation.")
    live_mode: bool = Field(True, description="If true, system attempts to use live data sources; affects error handling.")
    fail_fast_on_errors: bool = Field(True, description="If true, system may halt on critical data quality or API errors.")
    metrics_for_dynamic_threshold_distribution_tracking: List[str] = Field(
        default_factory=lambda: ["GIB_OI_based_Und", "VAPI_FA_Z_Score_Und", "DWFD_Z_Score_Und", "TW_LAF_Z_Score_Und"],
        description="List of underlying aggregate metric names to track historically for dynamic threshold calculations."
    )
    signal_activation: Dict[str, bool] = Field(default_factory=lambda: {"EnableAllSignals": True}, description="Master toggles for enabling or disabling specific signal generation routines or categories.")
    class Config: extra = 'forbid'

class ConvexValueAuthSettings(BaseModel):
    """Authentication settings for ConvexValue API."""
    use_env_variables: bool = Field(True, description="If true, attempts to load credentials from environment variables first.")
    auth_method: str = Field("email_password", description="Authentication method for ConvexValue API (e.g., 'email_password', 'api_key').")
    # Specific credential fields would be here if not using env variables, e.g., email: Optional[str], password: Optional[SecretStr]
    class Config: extra = 'forbid'

class DataFetcherSettings(BaseModel):
    """Settings for data fetching components."""
    convexvalue_auth: ConvexValueAuthSettings = Field(default_factory=ConvexValueAuthSettings, description="Authentication settings for ConvexValue.")
    tradier_api_key: str = Field(..., description="API Key for Tradier (sensitive, ideally from env var).")
    tradier_account_id: str = Field(..., description="Account ID for Tradier (sensitive, ideally from env var).")
    max_retries: int = Field(3, ge=0, description="Maximum number of retry attempts for a failing API call.")
    retry_delay_seconds: float = Field(5.0, ge=0, description="Base delay in seconds between API call retries.")
    # Added fields based on ValidationError
    api_keys: Optional[Dict[str, str]] = Field(default_factory=dict, description="Optional dictionary for API keys if not using direct fields.")
    retry_attempts: Optional[int] = Field(3, description="Number of retry attempts for API calls.") # Defaulted from error
    retry_delay: Optional[float] = Field(5.0, description="Delay in seconds between retries.") # Defaulted from error
    timeout: Optional[float] = Field(30.0, description="Timeout in seconds for API requests.") # Defaulted from error
    class Config: extra = 'forbid'

class DataManagementSettings(BaseModel):
    """Settings related to data caching and storage."""
    data_cache_dir: str = Field("data_cache_v2_5", description="Root directory for caching temporary data.")
    historical_data_store_dir: str = Field("data_cache_v2_5/historical_data_store", description="Directory for persistent historical market and metric data.")
    performance_data_store_dir: str = Field("data_cache_v2_5/performance_data_store", description="Directory for storing trade recommendation performance data.")
    # Added fields based on ValidationError
    cache_directory: Optional[str] = Field("data_cache_v2_5", description="Cache directory path.") # Defaulted from error
    data_store_directory: Optional[str] = Field("data_cache_v2_5/data_store", description="Data store directory path.") # Defaulted from error
    cache_expiry_hours: Optional[float] = Field(24.0, description="Cache expiry in hours.") # Defaulted from error
    class Config: extra = 'forbid'


# --- Settings for Specific Analytical Components ---
class EnhancedFlowMetricSettings(BaseModel):
    """Parameters for Tier 3 Enhanced Rolling Flow Metrics (VAPI-FA, DWFD, TW-LAF)."""
    vapi_fa_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters specific to VAPI-FA calculation (e.g., primary_flow_interval, iv_source_key).")
    dwfd_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters specific to DWFD calculation (e.g., flow_interval, fvd_weight_factor).")
    tw_laf_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters specific to TW-LAF calculation (e.g., time_weights_for_intervals, spread_calculation_params).")
    # Common params (can be overridden within specific metric_params if needed)
    z_score_window: int = Field(20, ge=5, le=200, description="Default window size for Z-score normalization of enhanced flow metrics.")
    # Added fields based on ValidationError
    acceleration_calculation_intervals: Optional[List[str]] = Field(default_factory=list, description="Intervals for flow acceleration calculation.")
    time_intervals: Optional[List[int]] = Field(default_factory=list, description="Time intervals for flow metrics.")
    liquidity_weight: Optional[float] = Field(None, description="Weight for liquidity adjustment.")
    divergence_threshold: Optional[float] = Field(None, description="Threshold for flow divergence.")
    lookback_periods: Optional[List[int]] = Field(default_factory=list, description="Lookback periods for flow calculations.")
    class Config: extra = 'forbid'

class StrategySettings(BaseModel): # General strategy settings, can be expanded
    """High-level strategy settings, often used for ATIF and TPO guidance."""
    # Example: thresholds for various signals or conviction modifiers
    # This model uses 'allow' as it's a common place for various ad-hoc strategy params.
    # Better practice would be to define specific sub-models for different strategy aspects.
    class Config: extra = 'allow'

class LearningParams(BaseModel):
    """Parameters governing the ATIF's performance-based learning loop."""
    performance_tracker_query_lookback: int = Field(90, ge=1, description="Number of days of historical performance data ATIF considers for learning.")
    learning_rate_for_signal_weights: float = Field(0.05, ge=0, le=1, description="Aggressiveness (0-1) of ATIF's adjustment to signal weights based on new performance.")
    learning_rate_for_target_adjustments: float = Field(0.02, ge=0, le=1, description="Aggressiveness (0-1) of ATIF's adjustment to target parameters based on performance.")
    min_trades_for_statistical_significance: int = Field(20, ge=1, description="Minimum number of trades for a specific setup/symbol/regime before performance significantly influences weights.")
    class Config: extra = 'forbid'

class AdaptiveTradeIdeaFrameworkSettings(BaseModel):
    """Comprehensive settings for the Adaptive Trade Idea Framework (ATIF)."""
    min_conviction_to_initiate_trade: float = Field(2.5, ge=0, le=5, description="Minimum ATIF conviction score (0-5 scale) to generate a new trade recommendation.")
    signal_integration_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for how ATIF integrates and weights raw signals (e.g., base_signal_weights, performance_weighting_sensitivity).")
    regime_context_weight_multipliers: Dict[str, float] = Field(default_factory=dict, description="Multipliers applied to signal weights based on current market regime.")
    conviction_mapping_params: Dict[str, Any] = Field(default_factory=dict, description="Rules and thresholds for mapping ATIF's internal assessment to a final conviction score.")
    strategy_specificity_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Rule set mapping [Assessment + Conviction + Regime + Context + IV] to specific option strategies, DTEs, and deltas.")
    intelligent_recommendation_management_rules: Dict[str, Any] = Field(default_factory=dict, description="Rules for adaptive exits, parameter adjustments, and partial position management.")
    learning_params: LearningParams = Field(default_factory=LearningParams, description="Parameters for ATIF's learning loop.")
    class Config: extra = 'forbid'

class TickerContextAnalyzerSettings(BaseModel):
    """Settings for the TickerContextAnalyzerV2_5."""
    # General settings
    lookback_days: int = Field(252, description="Default lookback days for historical analysis (e.g., for IV rank).") # Approx 1 year
    correlation_window: int = Field(60, description="Window for calculating correlations if used.")
    volatility_windows: List[int] = Field(default_factory=lambda: [1, 5, 20], description="Windows for short, medium, long term volatility analysis.")
    # Example specific settings for a ticker or default profile
    SPY: Dict[str, Any] = Field(default_factory=dict, description="Specific context analysis parameters for SPY.")
    DEFAULT_TICKER_PROFILE: Dict[str, Any] = Field(default_factory=dict, description="Default parameters for tickers not explicitly defined.")
    # Parameters for fetching external data if used (e.g., earnings calendar)
    # use_yahoo_finance: bool = False
    # yahoo_finance_rate_limit_seconds: float = 2.0
    # Added fields based on ValidationError
    volume_threshold: Optional[int] = Field(None, description="Volume threshold for ticker context analysis.")
    use_yahoo_finance: Optional[bool] = Field(False, description="Flag to use Yahoo Finance for data.")
    yahoo_finance_rate_limit_seconds: Optional[float] = Field(2.0, description="Rate limit in seconds for Yahoo Finance API calls.")
    class Config: extra = 'forbid'

class KeyLevelIdentifierSettings(BaseModel):
    """Settings for the KeyLevelIdentifierV2_5."""
    lookback_periods: int = Field(20, description="Lookback period for identifying significant prior S/R from A-MSPI history.")
    min_touches: int = Field(2, description="Minimum times a historical level must have been touched to be considered significant.")
    level_tolerance: float = Field(0.005, ge=0, le=0.1, description="Percentage tolerance for clustering nearby levels (e.g., 0.5% = 0.005).")
    # Thresholds for identifying levels from various sources
    nvp_support_quantile: float = Field(0.95, ge=0, le=1, description="Quantile for identifying strong NVP support levels.")
    nvp_resistance_quantile: float = Field(0.95, ge=0, le=1, description="Quantile for identifying strong NVP resistance levels.")
    # Other source-specific thresholds (e.g., for SGDHP, UGCH scores) would be defined here.
    # Added fields based on ValidationError
    volume_threshold: Optional[float] = Field(None, description="Volume threshold for key level identification.")
    oi_threshold: Optional[int] = Field(None, description="Open interest threshold for key level identification.")
    gamma_threshold: Optional[float] = Field(None, description="Gamma threshold for key level identification.")
    class Config: extra = 'forbid'

class HeatmapGenerationSettings(BaseModel):
    """Parameters for generating data for Enhanced Heatmaps."""
    ugch_params: Dict[str, Any] = Field(default_factory=lambda: {"greek_weights": {"norm_DXOI":1.0, "norm_GXOI":1.0}}, description="Parameters for UGCH data generation, e.g., weights for each Greek.")
    sgdhp_params: Dict[str, Any] = Field(default_factory=lambda: {"proximity_sensitivity_param": 0.02}, description="Parameters for SGDHP data generation, e.g., price proximity sensitivity.")
    ivsdh_params: Dict[str, Any] = Field(default_factory=lambda: {"time_decay_sensitivity_factor": 0.1}, description="Parameters for IVSDH data generation.")
    # Added field based on ValidationError
    flow_normalization_window: Optional[int] = Field(None, description="Window for flow normalization in heatmap generation.")
    class Config: extra = 'forbid'

class PerformanceTrackerSettingsV2_5(BaseModel):
    """Settings for the PerformanceTrackerV2_5 module."""
    performance_data_directory: str = Field("data_cache_v2_5/performance_data_store", description="Directory for storing performance tracking data files.")
    historical_window_days: int = Field(365, ge=1, description="Number of days of performance data to retain and consider for analysis.")
    weight_smoothing_factor: float = Field(0.1, ge=0, le=1, description="Smoothing factor for performance-based weight adjustments by ATIF (0=no new learning, 1=full new learning).")
    min_sample_size_for_stats: int = Field(10, ge=1, description="Minimum number of trade samples required for calculating reliable performance statistics for a setup/signal.")
    # confidence_threshold: float = Field(0.75, ge=0, le=1, description="Confidence threshold for performance-based adjustments - not directly used by ATIF learning_rate, but for user interpretation.")
    # update_interval_seconds: int = Field(3600, ge=1, description="Interval for batch updates or re-analysis of performance data (if applicable).")
    tracking_enabled: bool = Field(True, description="Master toggle for enabling/disabling performance tracking.")
    metrics_to_track_display: List[str] = Field(default_factory=lambda: ["returns", "sharpe_ratio", "max_drawdown", "win_rate"], description="List of performance metrics to display on dashboard.")
    # reporting_frequency: str = Field("daily", description="Frequency of generating performance reports (if applicable).")
    # benchmark_symbol: str = Field("SPY", description="Benchmark symbol for relative performance comparison.")
    # Added fields based on ValidationError
    min_sample_size: Optional[int] = Field(10, description="Minimum sample size for performance statistics.")
    confidence_threshold: Optional[float] = Field(0.75, description="Confidence threshold for performance adjustments.")
    update_interval_seconds: Optional[int] = Field(3600, description="Interval for performance data updates.")
    class Config: extra = 'forbid'

class DTDPIParamsConfig(BaseModel):
    """Specific parameters for D-TDPI calculation."""
    tdpi_beta_flow_alignment_coeff: float = Field(default=0.4, ge=0, le=2.0, description="Coefficient for flow alignment in D-TDPI. Default 0.4")
    # Add other D-TDPI specific parameters here if any in the future
    class Config: extra = 'forbid'

class AdaptiveMetricParameters(BaseModel): # Top-level container for all adaptive metric specific settings in config
    """Container for settings related to all Tier 2 Adaptive Metrics."""
    a_dag_settings: Dict[str, Any] = Field(default_factory=dict, description="Settings for Adaptive Delta Adjusted Gamma Exposure (A-DAG).")
    e_sdag_settings: Dict[str, Any] = Field(default_factory=dict, description="Settings for Enhanced Skew and Delta Adjusted Gamma Exposure (E-SDAG) methodologies.")
    d_tdpi_settings: DTDPIParamsConfig = Field(default_factory=DTDPIParamsConfig, description="Settings for Dynamic Time Decay Pressure Indicator (D-TDPI).")
    vri_2_0_settings: Dict[str, Any] = Field(default_factory=dict, description="Settings for Volatility Regime Indicator Version 2.0 (VRI 2.0).")
    enhanced_heatmap_settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Settings for enhanced heatmap generation.")
    class Config: extra = 'forbid'

class TickerContextAnalyzerSettings(BaseModel):
    """Settings for the TickerContextAnalyzerV2_5."""
    # General settings
    lookback_days: int = Field(252, description="Default lookback days for historical analysis (e.g., for IV rank).") # Approx 1 year
    correlation_window: int = Field(60, description="Window for calculating correlations if used.")
    volatility_windows: List[int] = Field(default_factory=lambda: [1, 5, 20], description="Windows for short, medium, long term volatility analysis.")
    # Example specific settings for a ticker or default profile
    SPY: Dict[str, Any] = Field(default_factory=dict, description="Specific context analysis parameters for SPY.")
    DEFAULT_TICKER_PROFILE: Dict[str, Any] = Field(default_factory=dict, description="Default parameters for tickers not explicitly defined.")
    # Added fields based on ValidationError
    volume_threshold: Optional[int] = Field(1000000, description="Volume threshold for ticker context analysis.")
    use_yahoo_finance: Optional[bool] = Field(False, description="Flag to use Yahoo Finance for data.")
    yahoo_finance_rate_limit_seconds: Optional[float] = Field(2.0, description="Rate limit in seconds for Yahoo Finance API calls.")
    class Config: extra = 'forbid'

class KeyLevelIdentifierSettings(BaseModel):
    """Settings for the KeyLevelIdentifierV2_5."""
    lookback_periods: int = Field(20, description="Lookback period for identifying significant prior S/R from A-MSPI history.")
    min_touches: int = Field(2, description="Minimum times a historical level must have been touched to be considered significant.")
    level_tolerance: float = Field(0.005, ge=0, le=0.1, description="Percentage tolerance for clustering nearby levels (e.g., 0.5% = 0.005).")
    nvp_support_quantile: float = Field(0.95, ge=0, le=1, description="Quantile for identifying strong NVP support levels.")
    nvp_resistance_quantile: float = Field(0.95, ge=0, le=1, description="Quantile for identifying strong NVP resistance levels.")
    # Added fields based on ValidationError
    volume_threshold: Optional[float] = Field(1.5, description="Volume threshold for key level identification.")
    oi_threshold: Optional[int] = Field(1000, description="Open interest threshold for key level identification.")
    gamma_threshold: Optional[float] = Field(0.1, description="Gamma threshold for key level identification.")
    class Config: extra = 'forbid'

class HeatmapGenerationSettings(BaseModel):
    """Parameters for generating data for Enhanced Heatmaps."""
    ugch_params: Dict[str, Any] = Field(default_factory=lambda: {"greek_weights": {"norm_DXOI":1.0, "norm_GXOI":1.0}}, description="Parameters for UGCH data generation, e.g., weights for each Greek.")
    sgdhp_params: Dict[str, Any] = Field(default_factory=lambda: {"proximity_sensitivity_param": 0.02}, description="Parameters for SGDHP data generation, e.g., price proximity sensitivity.")
    ivsdh_params: Dict[str, Any] = Field(default_factory=lambda: {"time_decay_sensitivity_factor": 0.1}, description="Parameters for IVSDH data generation.")
    # Added field based on ValidationError
    flow_normalization_window: Optional[int] = Field(100, description="Window for flow normalization in heatmap generation.")
    class Config: extra = 'forbid'

# --- Symbol Specific Overrides Structure ---
class SymbolDefaultOverridesStrategySettingsTargets(BaseModel):
    """Defines default target parameters for strategy settings."""
    target_atr_stop_loss_multiplier: float = Field(1.5, ge=0.1)
    class Config: extra = 'forbid'

class SymbolDefaultOverridesStrategySettings(BaseModel):
    """Defines default strategy settings, can be overridden per symbol."""
    targets: SymbolDefaultOverridesStrategySettingsTargets = Field(default_factory=SymbolDefaultOverridesStrategySettingsTargets)
    class Config: extra = 'forbid'

class SymbolDefaultOverrides(BaseModel):
    """Container for default settings that can be overridden by specific symbols."""
    strategy_settings: Optional[SymbolDefaultOverridesStrategySettings] = Field(default_factory=SymbolDefaultOverridesStrategySettings)
    # Can add other overridable sections here, e.g., market_regime_engine_settings
    class Config: extra = 'allow' # Allow adding other top-level setting blocks for DEFAULT

class SymbolSpecificOverrides(BaseModel):
    """
    Main container for symbol-specific configuration overrides.
    Keys are ticker symbols (e.g., "SPY", "AAPL") or "DEFAULT".
    Values are dictionaries structured like parts of the main config.
    """
    DEFAULT: Optional[SymbolDefaultOverrides] = Field(default_factory=SymbolDefaultOverrides, description="Default override profile applied if no ticker-specific override exists.")
    SPY: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Specific overrides for SPY.")
    # Add other commonly traded symbols as needed, e.g., AAPL: Optional[Dict[str, Any]]
    class Config: extra = 'allow' # Allows adding new ticker symbols as keys


# --- Database and Collector Settings (if used) ---
class DatabaseSettings(BaseModel):
    """Database connection settings (if a central DB is used)."""
    host: str = Field(..., description="Database host address.")
    port: int = Field(5432, description="Database port number.")
    database: str = Field(..., description="Database name.")
    user: str = Field(..., description="Database username.")
    password: str = Field(..., description="Database password (sensitive).") # Consider pydantic.SecretStr
    min_connections: int = Field(1, ge=0, description="Minimum number of connections in pool.")
    max_connections: int = Field(10, ge=1, description="Maximum number of connections in pool.")
    class Config: extra = 'forbid'

class IntradayCollectorSettings(BaseModel):
    """Settings for an intraday metrics collector service (if separate)."""
    watched_tickers: List[str] = Field(default_factory=lambda: ["SPY", "QQQ"], description="List of tickers for intraday metric collection.")
    metrics_to_collect: List[str] = Field(default_factory=lambda: ["vapi_fa", "dwfd", "tw_laf"], description="Specific metrics to collect intraday.")
    cache_dir: str = Field("cache/intraday_metrics_collector", description="Directory for intraday collector cache.")
    collection_interval_seconds: int = Field(60, ge=5, description="Interval in seconds between metric collections.")
    market_open_time: str = Field("09:30:00", description="Market open time (HH:MM:SS) for collector activity.")
    market_close_time: str = Field("16:00:00", description="Market close time (HH:MM:SS) for collector activity.")
    reset_cache_at_eod: bool = Field(True, description="Whether to wipe intraday cache at end of day.")
    # Added fields based on ValidationError
    metrics: Optional[List[str]] = Field(default_factory=list, description="List of metrics for the intraday collector.")
    reset_at_eod: Optional[bool] = Field(True, description="Whether to reset cache at EOD for intraday collector.") # Field already existed, ensuring Optional and default
    class Config: extra = 'forbid'


# --- Top-Level Configuration Model ---
class EOTSConfigV2_5(BaseModel):
    """
    The root model for the EOTS v2.5 system configuration (config_v2_5.json).
    It defines all valid parameters, their types, default values, and descriptions,
    ensuring configuration integrity and providing a structured way to access settings.
    """
    system_settings: SystemSettings = Field(default_factory=SystemSettings)
    data_fetcher_settings: DataFetcherSettings
    data_management_settings: DataManagementSettings = Field(default_factory=DataManagementSettings)
    database_settings: Optional[DatabaseSettings] = Field(None, description="Optional database connection settings.")
    
    data_processor_settings: DataProcessorSettings = Field(default_factory=DataProcessorSettings)
    adaptive_metric_parameters: AdaptiveMetricParameters = Field(default_factory=AdaptiveMetricParameters)
    enhanced_flow_metric_settings: EnhancedFlowMetricSettings = Field(default_factory=EnhancedFlowMetricSettings)
    key_level_identifier_settings: KeyLevelIdentifierSettings = Field(default_factory=KeyLevelIdentifierSettings)
    heatmap_generation_settings: HeatmapGenerationSettings = Field(default_factory=HeatmapGenerationSettings)
    market_regime_engine_settings: MarketRegimeEngineSettings = Field(default_factory=MarketRegimeEngineSettings)
    
    # Strategy and ATIF settings are often complex and interlinked
    strategy_settings: StrategySettings = Field(default_factory=StrategySettings, description="General strategy parameters, often used by ATIF/TPO.")
    adaptive_trade_idea_framework_settings: AdaptiveTradeIdeaFrameworkSettings
    
    ticker_context_analyzer_settings: TickerContextAnalyzerSettings = Field(default_factory=TickerContextAnalyzerSettings)
    performance_tracker_settings_v2_5: PerformanceTrackerSettingsV2_5 = Field(default_factory=PerformanceTrackerSettingsV2_5)
    
    visualization_settings: VisualizationSettings = Field(default_factory=VisualizationSettings)
    symbol_specific_overrides: SymbolSpecificOverrides = Field(default_factory=SymbolSpecificOverrides)
    
    # TimeOfDayDefinitions is now in context_schemas.py and typically loaded into MarketRegimeEngine or TickerContextAnalyzer
    # For direct config access if needed by other general utils:
    time_of_day_definitions: Optional[Dict[str,str]] = Field(None, description="Optional: Can load TimeOfDayDefinitions directly here if not handled by MRE/TCA init. Otherwise, they use context_schemas.TimeOfDayDefinitions.")

    intraday_collector_settings: Optional[IntradayCollectorSettings] = Field(None, description="Optional settings for a separate intraday metrics collector service.")

    class Config:
        json_schema_extra = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "EOTS_V2_5_Config_Schema_Root",
            "description": "Root schema for EOTS v2.5 configuration (config_v2_5.json). Defines all valid parameters, types, defaults, and descriptions for system operation."
        }
        extra = 'forbid' # Enforce strict configuration structure at the root level
