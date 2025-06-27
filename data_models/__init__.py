"""
EOTS v2.5 Data Models Package

This package consolidates all Pydantic models used for data representation
and configuration structuring within the Elite Options Trading System v2.5.
Models are organized into logical modules for better maintainability and clarity.

Modules:
- `base_types`: Common simple type aliases (e.g., PandasDataFrame).
- `raw_data`: Schemas for data as fetched from external sources.
- `processed_data`: Schemas for data after EOTS metric calculation and processing.
- `context_schemas`: Schemas defining contextual information (ticker context, time definitions).
- `signal_level_schemas`: Schemas for generated signals and identified key price levels.
- `atif_schemas`: Schemas for ATIF's internal assessments and directives.
- `recommendation_schemas`: Schemas for fully parameterized trade recommendations.
- `bundle_schemas`: Schemas for top-level data bundles for final output.
- `configuration_schemas`: Schemas defining the structure of the system's JSON configuration.
"""

# Import models from their respective modules to make them available at the package level
from .base_types import PandasDataFrame

from .raw_data import (
    RawOptionsContractV2_5,
    RawUnderlyingDataV2_5,
    RawUnderlyingDataCombinedV2_5,
    UnprocessedDataBundleV2_5
)

from .processed_data import (
    ProcessedContractMetricsV2_5,
    ProcessedStrikeLevelMetricsV2_5,
    ProcessedUnderlyingAggregatesV2_5,
    ProcessedDataBundleV2_5
)

from .context_schemas import (
    TickerContextDictV2_5,
    TimeOfDayDefinitions
)

from .signal_level_schemas import (
    SignalPayloadV2_5,
    KeyLevelV2_5,
    KeyLevelsDataV2_5
)

from .atif_schemas import (
    ATIFSituationalAssessmentProfileV2_5,
    ATIFStrategyDirectivePayloadV2_5,
    ATIFManagementDirectiveV2_5
)

from .recommendation_schemas import (
    TradeParametersV2_5,
    ActiveRecommendationPayloadV2_5
)

from .bundle_schemas import (
    FinalAnalysisBundleV2_5
)

from .configuration_schemas import (
    DashboardModeSettings,
    MainDashboardDisplaySettings,
    DashboardModeCollection,
    VisualizationSettings,
    DagAlphaCoeffs,
    TdpiBetaCoeffs,
    VriGammaCoeffs,
    CoefficientsSettings,
    DataProcessorSettings,
    MarketRegimeEngineSettings,
    SystemSettings,
    ConvexValueAuthSettings,
    DataFetcherSettings,
    DataManagementSettings,
    EnhancedFlowMetricSettings,
    StrategySettings,
    LearningParams,
    AdaptiveTradeIdeaFrameworkSettings,
    TickerContextAnalyzerSettings,
    KeyLevelIdentifierSettings,
    HeatmapGenerationSettings,
    PerformanceTrackerSettingsV2_5,
    AdaptiveMetricParameters,
    SymbolDefaultOverridesStrategySettingsTargets,
    SymbolDefaultOverridesStrategySettings,
    SymbolDefaultOverrides,
    SymbolSpecificOverrides,
    DatabaseSettings,
    IntradayCollectorSettings,
    EOTSConfigV2_5
)

# Define __all__ for explicit public API of the package
__all__ = [
    # Base Types
    "PandasDataFrame",

    # Raw Data
    "RawOptionsContractV2_5",
    "RawUnderlyingDataV2_5",
    "RawUnderlyingDataCombinedV2_5",
    "UnprocessedDataBundleV2_5",

    # Processed Data
    "ProcessedContractMetricsV2_5",
    "ProcessedStrikeLevelMetricsV2_5",
    "ProcessedUnderlyingAggregatesV2_5",
    "ProcessedDataBundleV2_5",

    # Context & Config Schemas (from context_schemas.py)
    "TickerContextDictV2_5",
    "TimeOfDayDefinitions",

    # Signal & Level Schemas
    "SignalPayloadV2_5",
    "KeyLevelV2_5",
    "KeyLevelsDataV2_5",

    # ATIF Schemas
    "ATIFSituationalAssessmentProfileV2_5",
    "ATIFStrategyDirectivePayloadV2_5",
    "ATIFManagementDirectiveV2_5",

    # Recommendation Schemas
    "TradeParametersV2_5",
    "ActiveRecommendationPayloadV2_5",

    # Bundle Schemas
    "FinalAnalysisBundleV2_5",

    # Configuration Schemas (from configuration_schemas.py)
    "DashboardModeSettings",
    "MainDashboardDisplaySettings",
    "DashboardModeCollection",
    "VisualizationSettings",
    "ChartType",
    "ChartLayoutConfigV2_5",
    "ControlPanelParametersV2_5",
    "DagAlphaCoeffs",
    "TdpiBetaCoeffs",
    "VriGammaCoeffs",
    "CoefficientsSettings",
    "DataProcessorSettings",
    "MarketRegimeEngineSettings",
    "SystemSettings",
    "ConvexValueAuthSettings",
    "DataFetcherSettings",
    "DataManagementSettings",
    "EnhancedFlowMetricSettings",
    "StrategySettings",
    "LearningParams",
    "AdaptiveTradeIdeaFrameworkSettings",
    "TickerContextAnalyzerSettings",
    "KeyLevelIdentifierSettings",
    "HeatmapGenerationSettings",
    "PerformanceTrackerSettingsV2_5",
    "AdaptiveMetricParameters",
    "SymbolDefaultOverridesStrategySettingsTargets",
    "SymbolDefaultOverridesStrategySettings",
    "SymbolDefaultOverrides",
    "SymbolSpecificOverrides",
    "DatabaseSettings",
    "IntradayCollectorSettings",
    "EOTSConfigV2_5"
]
