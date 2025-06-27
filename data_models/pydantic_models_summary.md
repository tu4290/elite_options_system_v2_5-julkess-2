# Pydantic Models Audit and Analysis Summary

This document provides a comprehensive summary of the Pydantic models within the EOTS v2.5 system, their roles, definitions, dependencies, and usage across the codebase. This summary reflects changes made to consolidate dashboard configurations and address redundancies.

## Table of Contents

1.  [Base Types (`data_models/base_types.py`)](#base-types)
2.  [Raw Data Schemas (`data_models/raw_data.py`)](#raw-data-schemas)
3.  [Context Schemas (`data_models/context_schemas.py`)](#context-schemas)
4.  [Processed Data Schemas (`data_models/processed_data.py`)](#processed-data-schemas)
5.  [Signal and Key Level Schemas (`data_models/signal_level_schemas.py`)](#signal-and-key-level-schemas)
6.  [ATIF Schemas (`data_models/atif_schemas.py`)](#atif-schemas)
7.  [Recommendation Schemas (`data_models/recommendation_schemas.py`)](#recommendation-schemas)
8.  [Bundle Schemas (`data_models/bundle_schemas.py`)](#bundle-schemas)
9.  [HuiHui AI System Schemas (`data_models/hui_hui_schemas.py`)](#huihui-ai-system-schemas)
10. [Mixture of Experts (MOE) Schemas (`data_models/moe_schemas_v2_5.py`)](#mixture-of-experts-moe-schemas)
11. [Performance Schemas (`data_models/performance_schemas.py`)](#performance-schemas)
12. [Dashboard Schemas (`data_models/dashboard_schemas.py`)](#dashboard-schemas)
13. [Configuration Schemas (`data_models/configuration_schemas.py`)](#configuration-schemas)
14. [Critical AI & System Models (from `data_models/deprecated_files/eots_schemas_v2_5.py`)](#critical-ai-system-models-from-deprecated)
    *   [AI Prediction Models](#ai-prediction-models)
    *   [AI Adaptation Models](#ai-adaptation-models)
    *   [AI Hub Compliance Models](#ai-hub-compliance-models)
    *   [Advanced Options & Learning Cycle Models](#advanced-options--learning-cycle-models)
    *   [Consolidated Settings Models (Deprecated)](#consolidated-settings-models-deprecated)

---

## 1. Base Types (`data_models/base_types.py`)

*   **`DataFrameSchema`**
    *   **Role**: Custom generic Pydantic type for validating Pandas DataFrames against a specified row-schema model. Ensures data integrity for DataFrame structures.
    *   **Defined in**: `data_models/base_types.py`
    *   **Dependencies**: Used internally by Pydantic when type-hinted (e.g., `DataFrameSchema[SomeRowModel]`). Its validation logic is critical wherever DataFrames need strict schema enforcement.
    *   **Usage**: Provides a mechanism to integrate Pandas DataFrames into the Pydantic validation ecosystem, crucial for components processing tabular data that must adhere to a specific structure.

*   **`PandasDataFrame`**
    *   **Role**: Type alias for `Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]`, offering flexibility in type hinting Pandas DataFrames.
    *   **Defined in**: `data_models/base_types.py`
    *   **Dependencies**:
        *   Re-exported by `data_models/__init__.py`.
        *   Used in `data_models/processed_data.py` (specifically in `ProcessedUnderlyingAggregatesV2_5` for `ivsdh_surface_data`).
    *   **Usage**: Applied in type hints where Pandas DataFrames are used, allowing for either actual DataFrame objects or their dictionary/list representations, especially before full validation with `DataFrameSchema` or when a less strict type hint is sufficient.

---

## 2. Raw Data Schemas (`data_models/raw_data.py`)

These models represent data as fetched from external sources with `Config.extra = 'allow'` for API flexibility.

*   **`RawOptionsContractV2_5`**
    *   **Role**: DTO for a single raw options contract's data directly from the source API (e.g., ConvexValue).
    *   **Defined in**: `data_models/raw_data.py`
    *   **Dependencies**: Nested within `UnprocessedDataBundleV2_5`.
    *   **Usage**: Consumed by `data_management/initial_processor_v2_5.py` as part of the initial data processing pipeline.

*   **`RawUnderlyingDataV2_5`**
    *   **Role**: DTO for raw underlying asset data from a primary data source.
    *   **Defined in**: `data_models/raw_data.py`
    *   **Dependencies**: Serves as a base for `RawUnderlyingDataCombinedV2_5`.
    *   **Usage**: Part of the raw data ingestion, populating fields directly from an API like ConvexValue `get_und`.

*   **`RawUnderlyingDataCombinedV2_5`**
    *   **Role**: DTO that consolidates raw underlying data from primary (e.g., ConvexValue) and supplementary sources (e.g., Tradier for OHLCV).
    *   **Defined in**: `data_models/raw_data.py` (inherits from `RawUnderlyingDataV2_5`)
    *   **Dependencies**: Nested within `UnprocessedDataBundleV2_5`.
    *   **Usage**: Consumed by `data_management/initial_processor_v2_5.py` for further processing.

*   **`UnprocessedDataBundleV2_5`**
    *   **Role**: Container DTO for all raw data (options contracts and combined underlying data) fetched at the start of an analysis cycle.
    *   **Defined in**: `data_models/raw_data.py`
    *   **Dependencies**:
        *   Produced by data fetching logic, likely orchestrated by `core_analytics_engine/its_orchestrator_v2_5.py`.
        *   Primary input for `data_management/initial_processor_v2_5.py`.
    *   **Usage**: Represents the initial state of data before any EOTS-specific calculations or transformations.

---

## 3. Context Schemas (`data_models/context_schemas.py`)

*   **`MarketRegimeState`** (Enum)
    *   **Role**: Enum defining standardized market regime classifications (e.g., BULLISH_TREND, VOLATILITY_EXPANSION).
    *   **Defined in**: `data_models/context_schemas.py`
    *   **Dependencies**:
        *   Used extensively by `core_analytics_engine/market_intelligence_engine_v2_5.py` for determining and working with regimes.
        *   Referenced in `core_analytics_engine/its_orchestrator_v2_5.py`.
    *   **Usage**: Ensures consistent representation and interpretation of market regimes throughout the system.

*   **`TickerContextDictV2_5`**
    *   **Role**: DTO holding various dynamic contextual flags and states for a specific ticker (e.g., 0DTE status, FOMC meeting day, intraday session, liquidity profile).
    *   **Defined in**: `data_models/context_schemas.py`
    *   **Dependencies**:
        *   Populated by `TickerContextAnalyzerV2_5` (inferred).
        *   Integrated into `ProcessedUnderlyingAggregatesV2_5` within `data_models/processed_data.py`.
        *   Consumed by various analytics modules (`core_analytics_engine/*`) and `huihui_integration/experts/*` to adapt logic based on context.
    *   **Usage**: Allows the system to tailor analysis and decision-making based on specific, timely characteristics of the asset being analyzed.

*   **`TimeOfDayDefinitions`**
    *   **Role**: Configuration model defining critical market time points (market open, close, pre-market start, EOD calculation times).
    *   **Defined in**: `data_models/context_schemas.py`
    *   **Dependencies**:
        *   Loaded as part of `EOTSConfigV2_5` (defined in `data_models/configuration_schemas.py`).
        *   Likely used by components like `MarketRegimeEngine`, `TickerContextAnalyzer`, and dashboard displays for time-sensitive logic.
    *   **Usage**: Standardizes key operational times for consistent behavior across time-dependent modules.

---

## 4. Processed Data Schemas (`data_models/processed_data.py`)

These models represent data after initial EOTS processing, cleaning, and metric calculation.

*   **`ProcessedContractMetricsV2_5`**
    *   **Role**: DTO for individual option contracts, enriched with calculated metrics (especially 0DTE suite metrics).
    *   **Defined in**: `data_models/processed_data.py` (inherits from `RawOptionsContractV2_5`)
    *   **Dependencies**: Nested within `ProcessedDataBundleV2_5`.
    *   **Usage**: Produced by `data_management/initial_processor_v2_5.py`. Represents the first level of EOTS-specific data enrichment for options contracts.

*   **`ProcessedStrikeLevelMetricsV2_5`**
    *   **Role**: DTO consolidating various metrics at individual strike prices (OI exposures, Greek flows, transactional pressures, adaptive metrics).
    *   **Defined in**: `data_models/processed_data.py`
    *   **Dependencies**: Nested within `ProcessedDataBundleV2_5`.
    *   **Usage**: Produced by `data_management/initial_processor_v2_5.py`. Forms the basis for identifying key levels and understanding market structure. Consumed by `core_analytics_engine/signal_generator_v2_5.py`, `core_analytics_engine/market_regime_engine_v2_5.py`, and dashboard components.

*   **`ProcessedUnderlyingAggregatesV2_5`**
    *   **Role**: DTO for fully processed and enriched underlying asset data, including aggregate metrics, classified market regime, ticker context, and dynamic thresholds.
    *   **Defined in**: `data_models/processed_data.py` (inherits from `RawUnderlyingDataCombinedV2_5`)
    *   **Dependencies**: Nested within `ProcessedDataBundleV2_5`. Contains `TickerContextDictV2_5` and potentially `PandasDataFrame` (`ivsdh_surface_data`).
    *   **Usage**: Produced by `data_management/initial_processor_v2_5.py`. Key input for higher-level analytical components like `SignalGeneratorV2_5`, `MarketRegimeEngine`, ATIF, and HuiHui experts.

*   **`ProcessedDataBundleV2_5`**
    *   **Role**: Central DTO for all processed data from an EOTS analysis cycle.
    *   **Defined in**: `data_models/processed_data.py`
    *   **Dependencies**:
        *   Produced by `data_management/initial_processor_v2_5.py`.
        *   Primary input for `core_analytics_engine/adaptive_trade_idea_framework_v2_5.py`, `core_analytics_engine/signal_generator_v2_5.py`, `huihui_integration/*`.
        *   Nested within `FinalAnalysisBundleV2_5`.
        *   Used by `core_analytics_engine/its_orchestrator_v2_5.py`.
    *   **Usage**: Represents the main data state after EOTS metric calculations, ready for advanced analytics and decision-making.

---

## 5. Signal and Key Level Schemas (`data_models/signal_level_schemas.py`)

*   **`SignalPayloadV2_5`**
    *   **Role**: DTO representing a single, discrete trading signal.
    *   **Defined in**: `data_models/signal_level_schemas.py`
    *   **Dependencies**: Produced by `core_analytics_engine/signal_generator_v2_5.py` and `core_analytics_engine/market_intelligence_engine_v2_5.py`. Primary input for ATIF. Included in `FinalAnalysisBundleV2_5`.
    *   **Usage**: Encapsulates signal event information for ATIF evaluation.

*   **`KeyLevelV2_5`**
    *   **Role**: DTO for a single identified key price level.
    *   **Defined in**: `data_models/signal_level_schemas.py`
    *   **Dependencies**: Nested within `KeyLevelsDataV2_5`.
    *   **Usage**: Produced by `KeyLevelIdentifierV2_5` (orchestrated by `ITSOrchestratorV2_5`).

*   **`KeyLevelsDataV2_5`**
    *   **Role**: DTO container aggregating all identified key levels.
    *   **Defined in**: `data_models/signal_level_schemas.py`
    *   **Dependencies**: Produced by `core_analytics_engine/its_orchestrator_v2_5.py`. Consumed by ATIF and `TradeParameterOptimizerV2_5`. Included in `FinalAnalysisBundleV2_5`. Displayed on dashboard.
    *   **Usage**: Provides a structured overview of critical price zones.

---

## 6. ATIF Schemas (`data_models/atif_schemas.py`)

Adaptive Trade Idea Framework (ATIF) DTOs.

*   **`ATIFSituationalAssessmentProfileV2_5`**
    *   **Role**: DTO for ATIF's internal market situation assessment.
    *   **Defined in**: `data_models/atif_schemas.py`
    *   **Dependencies**: Used by `core_analytics_engine/adaptive_trade_idea_framework_v2_5.py` and `core_analytics_engine/atif_engine_v2_5.py`. Nested in `ATIFStrategyDirectivePayloadV2_5`.
    *   **Usage**: Captures ATIF's aggregated scores before final conviction.

*   **`ATIFStrategyDirectivePayloadV2_5`**
    *   **Role**: DTO for an ATIF strategic directive before optimization.
    *   **Defined in**: `data_models/atif_schemas.py`
    *   **Dependencies**: Produced by ATIF engines. Input for `TradeParameterOptimizerV2_5`. Included in `FinalAnalysisBundleV2_5`.
    *   **Usage**: Specifies strategy type, DTE/delta ranges, and conviction for trade parameterization.

*   **`ATIFManagementDirectiveV2_5`**
    *   **Role**: DTO for ATIF management actions on active trades.
    *   **Defined in**: `data_models/atif_schemas.py`
    *   **Dependencies**: Produced by ATIF. Consumed by `ITSOrchestratorV2_5`.
    *   **Usage**: Instructs modifications to active trades.

---

## 7. Recommendation Schemas (`data_models/recommendation_schemas.py`)

*   **`TradeParametersV2_5`**
    *   **Role**: DTO for parameters of a single options trade leg.
    *   **Defined in**: `data_models/recommendation_schemas.py`
    *   **Dependencies**: Nested in `ActiveRecommendationPayloadV2_5`.
    *   **Usage**: Produced by `TradeParameterOptimizerV2_5`. Details a specific option contract.

*   **`ActiveRecommendationPayloadV2_5`**
    *   **Role**: DTO for a fully parameterized, active/managed trade recommendation.
    *   **Defined in**: `data_models/recommendation_schemas.py`
    *   **Dependencies**: Produced by `TradeParameterOptimizerV2_5` (refined by `ATIFEngineV2_5`). Consumed by `PerformanceTrackerV2_5`. Displayed on dashboard. Included in `FinalAnalysisBundleV2_5`.
    *   **Usage**: Tracks actionable trade ideas through their lifecycle.

---

## 8. Bundle Schemas (`data_models/bundle_schemas.py`)

*   **`FinalAnalysisBundleV2_5`**
    *   **Role**: Comprehensive DTO for all analytical outputs of a cycle.
    *   **Defined in**: `data_models/bundle_schemas.py`
    *   **Dependencies**: Nests `ProcessedDataBundleV2_5`, `SignalPayloadV2_5`, `KeyLevelsDataV2_5`, etc. Produced by `ITSOrchestratorV2_5`. Key input for dashboard modes, `MCPUnifiedManagerV2_5`, and `huihui_integration`.
    *   **Usage**: Primary data product for UI and external system consumption.

---

## 9. HuiHui AI System Schemas (`data_models/hui_hui_schemas.py`)

DTOs and configuration for the HuiHui AI Expert System.

*   **`HuiHuiExpertType`** (Enum): Defines HuiHui expert types.
*   **`HuiHuiModelConfigV2_5`**: Config for HuiHui AI models.
*   **`HuiHuiExpertConfigV2_5`**: Config for individual HuiHui experts.
*   **`HuiHuiAnalysisRequestV2_5`**: DTO for analysis tasks to HuiHui.
*   **`HuiHuiAnalysisResponseV2_5`**: DTO for HuiHui analysis results.
*   **`HuiHuiUsageRecordV2_5`**: Tracks HuiHui expert usage.
*   **`HuiHuiPerformanceMetricsV2_5`**: Aggregated performance for HuiHui experts.
*   **`HuiHuiEnsembleConfigV2_5`**: Config for HuiHui expert ensembles.
*   **`HuiHuiUserFeedbackV2_5`**: Stores user feedback.
*   **Usage**: Define the data contract for HuiHui AI system interactions.

---

## 10. Mixture of Experts (MOE) Schemas (`data_models/moe_schemas_v2_5.py`)

DTOs for orchestrating multiple specialized AI models.

*   **Enums (`ExpertStatus`, `RoutingStrategy`, etc.)**: Standardize MOE states.
*   **`MOEExpertRegistryV2_5`**: DTO for registering MOE experts.
*   **`MOEGatingNetworkV2_5`**: DTO for gating network routing decisions.
*   **`MOEExpertResponseV2_5`**: DTO for an individual expert's response.
*   **`MOEUnifiedResponseV2_5`**: DTO for aggregated MOE system response.
*   **Usage**: Define data structures for an MOE system. Likely used by a central MOE engine.

---

## 11. Performance Schemas (`data_models/performance_schemas.py`)

*   **Enums (`PerformanceInterval`, `PerformanceMetricType`)**: Standardize performance data categories.
*   **`PerformanceMetricV2_5`**: Generic DTO for a single performance metric.
*   **`SystemPerformanceV2_5`**: DTO for system-level metrics.
*   **`BacktestPerformanceV2_5`**: DTO for strategy backtest results.
*   **`ExecutionMetricsV2_5`**: DTO for trade execution metrics.
*   **`PerformanceReportV2_5`**: Aggregated DTO for comprehensive performance reports.
*   **Usage**: Used by performance tracking/reporting modules.

---

## 12. Dashboard Schemas (`data_models/dashboard_schemas.py`)

*   **`DashboardModeType`** (Enum): Defines EOTS UI dashboard modes.
*   **`DashboardModeSettings`** (in `dashboard_schemas.py`):
    *   **Role**: Configuration for a specific dashboard mode's display settings (module, charts, label, icon). Note: A similar `DashboardModeSettings` exists in `configuration_schemas.py` used by `DashboardModeCollection`. The version here is more detailed for direct dashboard use.
    *   **Defined in**: `data_models/dashboard_schemas.py`
    *   **Dependencies**: Used by `DashboardConfigV2_5`.
*   **`DashboardConfigV2_5`**:
    *   **Role**: Main configuration structure for the EOTS dashboard application's internal state management. Static setup aspects (available modes, default chart layouts, default control panel settings) are primarily configured via `EOTSConfigV2_5.visualization_settings`. This model can be used by the dashboard application to hold its runtime representation of these settings.
    *   **Defined in**: `data_models/dashboard_schemas.py`
    *   **Dependencies**: Likely used at the entry point of the dashboard application. Its fields `available_modes`, `default_mode`, `chart_configs`, and `control_panel` would derive initial values from `EOTSConfigV2_5.visualization_settings`.
*   **`ComponentComplianceV2_5`**:
    *   **Role**: Tracks dashboard component compliance with filters.
    *   **Defined in**: `data_models/dashboard_schemas.py`
    *   **Dependencies**: Used by `dashboard_application/modes/ai_dashboard/component_compliance_tracker_v2_5.py`.
*   **`DashboardStateV2_5`**:
    *   **Role**: DTO for tracking the live state of the dashboard UI.
    *   **Defined in**: `data_models/dashboard_schemas.py`
    *   **Dependencies**: Managed by dashboard callback/state logic.
*   **Usage**: These models structure the runtime configuration, user inputs, and state management of the EOTS dashboard application. Static initial setup parameters are largely drawn from `EOTSConfigV2_5.visualization_settings`.

---

## 13. Configuration Schemas (`data_models/configuration_schemas.py`)

*   **`ChartType`** (Enum)
    *   **Role**: Enum defining supported chart types for dashboard visualizations.
    *   **Defined in**: `data_models/configuration_schemas.py` (Moved from `dashboard_schemas.py`)
    *   **Dependencies**: Used by `ChartLayoutConfigV2_5`.
*   **`ChartLayoutConfigV2_5`**
    *   **Role**: Unified chart layout configuration for consistent visualization.
    *   **Defined in**: `data_models/configuration_schemas.py` (Moved from `dashboard_schemas.py`)
    *   **Dependencies**: Used in `VisualizationSettings.default_chart_layouts`. Consumed by dashboard components.
*   **`ControlPanelParametersV2_5`**
    *   **Role**: DTO for defining the structure and defaults of dashboard control panel parameters.
    *   **Defined in**: `data_models/configuration_schemas.py` (Moved from `dashboard_schemas.py`)
    *   **Dependencies**: Used in `VisualizationSettings.default_control_panel_settings`. Instantiated by dashboard input validators.
*   **`EOTSConfigV2_5`**
    *   **Role**: The root Pydantic model for the entire system's configuration.
    *   **Defined in**: `data_models/configuration_schemas.py`
    *   **Dependencies**: Consumed by `utils/config_manager_v2_5.py`. Accessed by virtually all modules. Nests most other models in this file.
    *   **Usage**: Central point for all system settings.
*   **`VisualizationSettings`**
    *   **Role**: Overall visualization and dashboard static setup settings.
    *   **Defined in**: `data_models/configuration_schemas.py`
    *   **Dependencies**: Nested in `EOTSConfigV2_5`. Now includes `default_mode_label`, `default_chart_layouts` (using `ChartLayoutConfigV2_5`), and `default_control_panel_settings` (using `ControlPanelParametersV2_5`).
    *   **Usage**: Primary source for initial dashboard setup parameters read from the main system configuration.
*   **Other Sub-Configuration Models** (e.g., `SystemSettings`, `DataFetcherSettings`, `DashboardModeSettings`, `MainDashboardDisplaySettings`, `DashboardModeCollection`, etc.)
    *   **Role**: Define structure for specific sections of the system configuration.
    *   **Defined in**: `data_models/configuration_schemas.py`
    *   **Dependencies**: Nested within `EOTSConfigV2_5`.
    *   **Usage**: Provide typed access to specific groups of settings.
    *   **Note on Internal Redundancy**: Definitions for `TickerContextAnalyzerSettings`, `KeyLevelIdentifierSettings`, `HeatmapGenerationSettings` appeared twice. The system uses the later, more complete definitions for these due to Python's class resolution order. Manual cleanup of the earlier definitions is advisable for code hygiene.

---

## 14. Critical AI & System Models (from `data_models/deprecated_files/eots_schemas_v2_5.py`)

(This section remains as a reference to important models in the deprecated file, as per the original analysis. These are not part of the focused refactoring of non-deprecated files but are important for overall system understanding.)

The following models are defined in `data_models/deprecated_files/eots_schemas_v2_5.py`. Their definitions should ideally be moved to appropriate primary schema files if they are actively used and unique.

### AI Prediction Models
*   `AIPredictionRequestV2_5`, `AIPredictionV2_5`, `AIPredictionPerformanceV2_5`
### AI Adaptation Models
*   `AIAdaptationRequestV2_5`, `AIAdaptationV2_5`, `AIAdaptationPerformanceV2_5`
### AI Hub Compliance Models
*   `ControlPanelParametersV2_5` (Strict Version), `AIHubComplianceReportV2_5`, `FilteredDataBundleV2_5`
### Advanced Options & Learning Cycle Models
*   `AdvancedOptionsMetricsV2_5`, `MarketPattern`, `StrategicRecommendationV2_5`, `LearningInsightV2_5`, `MarketPredictionV2_5`, `PredictionPerformanceV2_5`, `PredictionConfigV2_5`, `UnifiedPredictionResult`, `OptimizationConfigV2_5`, `OptimizationMetricsV2_5`, `ParameterOptimizationResultV2_5`, `SystemStateV2_5`
### Consolidated Settings Models (Deprecated)
*   `DataManagementConfigV2_5`, `AnalyticsEngineConfigV2_5`, `AIConfigV2_5`, `TradingConfigV2_5`, `DashboardConfigV2_5`

---

This summary provides an overview of the Pydantic models identified within the EOTS v2.5 system's `data_models` directory, reflecting recent refactoring efforts.
