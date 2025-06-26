# Pydantic Models Audit and Analysis Summary

This document provides a comprehensive summary of the Pydantic models within the EOTS v2.5 system, their roles, definitions, dependencies, and usage across the codebase.

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
        *   Populated by `TickerContextAnalyzerV2_5` (not explicitly seen but inferred).
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
    *   **Usage**: Produced by `data_management/initial_processor_v2_5.py` (which likely uses `core_analytics_engine/metrics_calculator_v2_5.py`). Represents the first level of EOTS-specific data enrichment for options contracts.

*   **`ProcessedStrikeLevelMetricsV2_5`**
    *   **Role**: DTO consolidating various metrics at individual strike prices (OI exposures, Greek flows, transactional pressures, adaptive metrics).
    *   **Defined in**: `data_models/processed_data.py`
    *   **Dependencies**: Nested within `ProcessedDataBundleV2_5`.
    *   **Usage**: Produced by `data_management/initial_processor_v2_5.py`. Forms the basis for identifying key levels and understanding market structure. Consumed by `core_analytics_engine/signal_generator_v2_5.py`, `core_analytics_engine/market_regime_engine_v2_5.py`, and dashboard components.

*   **`ProcessedUnderlyingAggregatesV2_5`**
    *   **Role**: DTO for fully processed and enriched underlying asset data, including aggregate metrics, classified market regime, ticker context, and dynamic thresholds.
    *   **Defined in**: `data_models/processed_data.py` (inherits from `RawUnderlyingDataCombinedV2_5`)
    *   **Dependencies**: Nested within `ProcessedDataBundleV2_5`. Contains `TickerContextDictV2_5` and potentially `PandasDataFrame` (`ivsdh_surface_data`).
    *   **Usage**: Produced by `data_management/initial_processor_v2_5.py` (leveraging `core_analytics_engine/metrics_calculator_v2_5.py`). Key input for higher-level analytical components like `SignalGeneratorV2_5`, `MarketRegimeEngine`, ATIF, and HuiHui experts.

*   **`ProcessedDataBundleV2_5`**
    *   **Role**: Central DTO for all processed data from an EOTS analysis cycle (enriched options, strike-level, and underlying data).
    *   **Defined in**: `data_models/processed_data.py`
    *   **Dependencies**:
        *   Produced by `data_management/initial_processor_v2_5.py` (which uses `core_analytics_engine/metrics_calculator_v2_5.py`).
        *   Primary input for `core_analytics_engine/adaptive_trade_idea_framework_v2_5.py`, `core_analytics_engine/signal_generator_v2_5.py`, `huihui_integration/*`.
        *   Nested within `FinalAnalysisBundleV2_5`.
        *   Used by `core_analytics_engine/its_orchestrator_v2_5.py` for coordinating further analysis.
    *   **Usage**: Represents the main data state after EOTS metric calculations, ready for advanced analytics and decision-making.

---

## 5. Signal and Key Level Schemas (`data_models/signal_level_schemas.py`)

*   **`SignalPayloadV2_5`**
    *   **Role**: DTO representing a single, discrete trading signal generated by the system (e.g., directional, volatility, structural).
    *   **Defined in**: `data_models/signal_level_schemas.py`
    *   **Dependencies**:
        *   Produced by `core_analytics_engine/signal_generator_v2_5.py` and `core_analytics_engine/market_intelligence_engine_v2_5.py`.
        *   Primary input for `core_analytics_engine/adaptive_trade_idea_framework_v2_5.py` (ATIF).
        *   Included in `FinalAnalysisBundleV2_5`.
    *   **Usage**: Encapsulates information about a specific signal event, including its nature, strength, and context, for ATIF evaluation.

*   **`KeyLevelV2_5`**
    *   **Role**: DTO representing a single identified key price level (e.g., support, resistance, pin zone, volatility trigger).
    *   **Defined in**: `data_models/signal_level_schemas.py`
    *   **Dependencies**: Nested within `KeyLevelsDataV2_5`.
    *   **Usage**: Produced by `KeyLevelIdentifierV2_5` (likely orchestrated by `core_analytics_engine/its_orchestrator_v2_5.py`). Details a specific critical price zone.

*   **`KeyLevelsDataV2_5`**
    *   **Role**: DTO container aggregating all identified key levels for an analysis cycle, categorized by type.
    *   **Defined in**: `data_models/signal_level_schemas.py`
    *   **Dependencies**:
        *   Produced by `core_analytics_engine/its_orchestrator_v2_5.py` (which likely uses `KeyLevelIdentifierV2_5`).
        *   Consumed by `core_analytics_engine/adaptive_trade_idea_framework_v2_5.py` and `core_analytics_engine/trade_parameter_optimizer_v2_5.py`.
        *   Included in `FinalAnalysisBundleV2_5`.
        *   Displayed in `dashboard_application/modes/structure_mode_display_v2_5.py`.
    *   **Usage**: Provides a structured overview of critical price zones to various analytical components and the UI.

---

## 6. ATIF Schemas (`data_models/atif_schemas.py`)

Adaptive Trade Idea Framework (ATIF) DTOs.

*   **`ATIFSituationalAssessmentProfileV2_5`**
    *   **Role**: DTO representing ATIF's internal, holistic assessment of the current market situation.
    *   **Defined in**: `data_models/atif_schemas.py`
    *   **Dependencies**:
        *   Produced and used internally by `core_analytics_engine/adaptive_trade_idea_framework_v2_5.py` and `core_analytics_engine/atif_engine_v2_5.py`.
        *   Nested within `ATIFStrategyDirectivePayloadV2_5`.
    *   **Usage**: Captures ATIF's aggregated weighted scores for various outlooks (bullish, bearish, volatility) before final conviction.

*   **`ATIFStrategyDirectivePayloadV2_5`**
    *   **Role**: DTO representing a strategic directive formulated by ATIF when a trading opportunity is identified with sufficient conviction.
    *   **Defined in**: `data_models/atif_schemas.py`
    *   **Dependencies**:
        *   Produced by `core_analytics_engine/adaptive_trade_idea_framework_v2_5.py` and `core_analytics_engine/atif_engine_v2_5.py`.
        *   Primary input for `core_analytics_engine/trade_parameter_optimizer_v2_5.py`.
        *   Included in `FinalAnalysisBundleV2_5` as `atif_recommendations_v2_5` (pre-optimization).
        *   Referenced by `dashboard_application/*` and `huihui_integration/*`.
    *   **Usage**: Specifies the type of options strategy, target DTE/delta ranges, and ATIF's conviction, serving as input for detailed trade parameterization.

*   **`ATIFManagementDirectiveV2_5`**
    *   **Role**: DTO representing a specific management action directive issued by ATIF for an existing, active trade recommendation.
    *   **Defined in**: `data_models/atif_schemas.py`
    *   **Dependencies**:
        *   Produced by `core_analytics_engine/adaptive_trade_idea_framework_v2_5.py`.
        *   Consumed by `core_analytics_engine/its_orchestrator_v2_5.py` to modify active trades.
    *   **Usage**: Instructs how to modify an active trade (e.g., exit, adjust stop-loss/targets) based on evolving market conditions.

---

## 7. Recommendation Schemas (`data_models/recommendation_schemas.py`)

*   **`TradeParametersV2_5`**
    *   **Role**: DTO encapsulating the precise, executable parameters for a single leg of an options trade.
    *   **Defined in**: `data_models/recommendation_schemas.py`
    *   **Dependencies**: Nested within `ActiveRecommendationPayloadV2_5`.
    *   **Usage**: Produced by `core_analytics_engine/trade_parameter_optimizer_v2_5.py`. Details a specific option contract part of a broader strategy.

*   **`ActiveRecommendationPayloadV2_5`**
    *   **Role**: DTO representing a fully formulated and parameterized trade recommendation that is active or recently managed.
    *   **Defined in**: `data_models/recommendation_schemas.py`
    *   **Dependencies**:
        *   Primarily produced by `core_analytics_engine/trade_parameter_optimizer_v2_5.py` and potentially updated by `core_analytics_engine/atif_engine_v2_5.py` (via management directives).
        *   Consumed by `data_management/performance_tracker_v2_5.py` for outcome tracking.
        *   Displayed by `dashboard_application/modes/main_dashboard_display_v2_5.py`.
        *   Included as a list in `FinalAnalysisBundleV2_5`.
    *   **Usage**: The primary data structure for displaying and tracking actionable trade ideas, including their lifecycle from issuance to closure and P&L.

---

## 8. Bundle Schemas (`data_models/bundle_schemas.py`)

*   **`FinalAnalysisBundleV2_5`**
    *   **Role**: Comprehensive, top-level DTO encapsulating all analytical outputs for a single symbol from one full EOTS analysis cycle.
    *   **Defined in**: `data_models/bundle_schemas.py`
    *   **Dependencies**:
        *   Nests `ProcessedDataBundleV2_5`, `SignalPayloadV2_5` (list), `KeyLevelsDataV2_5`, `ATIFStrategyDirectivePayloadV2_5` (list), and `ActiveRecommendationPayloadV2_5` (list).
        *   Produced by `core_analytics_engine/its_orchestrator_v2_5.py`.
        *   Key input for almost all `dashboard_application/modes/*` for UI rendering.
        *   Consumed by `core_analytics_engine/mcp_unified_manager_v2_5.py` and `huihui_integration/*` for AI processing.
    *   **Usage**: The primary data product consumed by the dashboard for visualization and by any external systems needing a complete analytical picture.

---

## 9. HuiHui AI System Schemas (`data_models/hui_hui_schemas.py`)

DTOs and configuration for the HuiHui AI Expert System.

*   **`HuiHuiExpertType`** (Enum): Defines types of HuiHui experts.
*   **`HuiHuiModelConfigV2_5`**: Configuration for HuiHui AI models (temperature, tokens, etc.).
*   **`HuiHuiExpertConfigV2_5`**: Configuration for individual HuiHui experts.
*   **`HuiHuiAnalysisRequestV2_5`**:
    *   **Role**: DTO for submitting analysis tasks to HuiHui.
    *   **Dependencies**: Can contain `ProcessedDataBundleV2_5`. Produced by `dashboard_application/modes/ai_dashboard/pydantic_intelligence_engine_v2_5.py` and `core_analytics_engine/huihui_ai_integration_v2_5.py`. Consumed by `huihui_integration/core/*` and `huihui_integration/orchestrator_bridge/*`.
*   **`HuiHuiAnalysisResponseV2_5`**:
    *   **Role**: DTO for results from a HuiHui analysis.
    *   **Dependencies**: Produced by `huihui_integration/core/*` and `dashboard_application/modes/ai_dashboard/pydantic_intelligence_engine_v2_5.py`. Can contain `AIPredictionV2_5` (defined in deprecated, as `List[Dict[str,Any]]`).
*   **`HuiHuiUsageRecordV2_5`**: Tracks usage of HuiHui experts.
*   **`HuiHuiPerformanceMetricsV2_5`**: Aggregated performance for HuiHui experts.
*   **`HuiHuiEnsembleConfigV2_5`**: Configuration for ensembles of HuiHui experts.
*   **`HuiHuiUserFeedbackV2_5`**: Stores user feedback on HuiHui analyses.
*   **Usage**: These models define the data contract for interaction with the HuiHui AI system, from configuration and requests to responses and performance monitoring.

---

## 10. Mixture of Experts (MOE) Schemas (`data_models/moe_schemas_v2_5.py`)

DTOs for a system that orchestrates multiple specialized AI models.

*   **Enums (`ExpertStatus`, `RoutingStrategy`, etc.)**: Standardize MOE states and strategies.
*   **`MOEExpertRegistryV2_5`**: DTO for registering experts in the MOE system.
*   **`MOEGatingNetworkV2_5`**: DTO for the gating network's decision on expert routing.
*   **`MOEExpertResponseV2_5`**: DTO for an individual expert's response.
*   **`MOEUnifiedResponseV2_5`**: DTO for the aggregated response from the MOE system.
*   **Usage**: These models define the data structures for an MOE system, managing expert registration, request routing to appropriate experts, and consolidating their responses. Likely used by a central MOE engine within `core_analytics_engine`.

---

## 11. Performance Schemas (`data_models/performance_schemas.py`)

*   **Enums (`PerformanceInterval`, `PerformanceMetricType`)**: Standardize performance data categories.
*   **`PerformanceMetricV2_5`**: Generic DTO for a single performance metric.
*   **`SystemPerformanceV2_5`**: DTO for system-level hardware/OS metrics.
*   **`BacktestPerformanceV2_5`**: DTO for strategy backtest results.
*   **`ExecutionMetricsV2_5`**: DTO for trade execution metrics.
*   **`PerformanceReportV2_5`**:
    *   **Role**: Aggregated DTO for a comprehensive performance report.
    *   **Defined in**: `data_models/performance_schemas.py`
    *   **Dependencies**: Nests other performance models.
    *   **Usage**: Likely produced by a performance tracking/reporting module and used for display or storage.

---

## 12. Dashboard Schemas (`data_models/dashboard_schemas.py`)

*   **Enums (`DashboardModeType`, `ChartType`)**: Categorize dashboard modes and chart types.
*   **`DashboardModeSettings`**: Configuration for a single dashboard mode's display. (Note: also a version in `configuration_schemas.py` via `DashboardModeCollection`).
*   **`ChartLayoutConfigV2_5`**: Unified chart layout settings, with a method to convert to Plotly layout.
*   **`ControlPanelParametersV2_5`**:
    *   **Role**: DTO for user inputs from the dashboard's control panel (symbol, DTE, etc.). Critical for data fetching and filtering.
    *   **Defined in**: `data_models/dashboard_schemas.py`
    *   **Dependencies**: Instantiated by `dashboard_application/modes/PYDANTIC_FIRST_CONTROL_PANEL_VALIDATOR_V2_5.py` and `dashboard_application/modes/ai_dashboard/ai_hub_compliance_manager_v2_5.py`.
*   **`DashboardConfigV2_5`**: Main configuration model specifically for the dashboard application's internal structure if not using `VisualizationSettings` from main config.
*   **`ComponentComplianceV2_5`**: Tracks dashboard component compliance with filters. Used by `dashboard_application/modes/ai_dashboard/component_compliance_tracker_v2_5.py`.
*   **`DashboardStateV2_5`**: Tracks the live state of the dashboard UI.
*   **Usage**: These models structure the configuration, user inputs, and state management of the EOTS dashboard application.

---

## 13. Configuration Schemas (`data_models/configuration_schemas.py`)

*   **`EOTSConfigV2_5`**
    *   **Role**: The root Pydantic model for the entire system's configuration. It validates and provides typed access to all settings.
    *   **Defined in**: `data_models/configuration_schemas.py`
    *   **Dependencies**:
        *   Consumed by `utils/config_manager_v2_5.py` to load and validate `config_v2_5.json`.
        *   Accessed by virtually all modules in the application to retrieve their specific configurations.
        *   Nests almost all other configuration models defined in this file.
    *   **Usage**: Central point for all system settings, ensuring type safety and structure for configurations.

*   **Various Sub-Configuration Models** (e.g., `SystemSettings`, `DataFetcherSettings`, `DataProcessorSettings`, `MarketRegimeEngineSettings`, `AdaptiveTradeIdeaFrameworkSettings`, `VisualizationSettings`, `SymbolSpecificOverrides`, `CoefficientsSettings`, `LearningParams`, `IntradayCollectorSettings`, etc.)
    *   **Role**: Each model defines the structure for a specific section of the system's configuration.
    *   **Defined in**: `data_models/configuration_schemas.py`
    *   **Dependencies**: Nested within `EOTSConfigV2_5`.
    *   **Usage**: Provide typed access to specific groups of settings used by different components of the EOTS system.
    *   **Note on Redundancy**: `TickerContextAnalyzerSettings`, `KeyLevelIdentifierSettings`, `HeatmapGenerationSettings` are defined twice within this file. The first occurrences or those with more complete defaults should be considered canonical.

---

## 14. Critical AI & System Models (from `data_models/deprecated_files/eots_schemas_v2_5.py`)

The following models are defined in `data_models/deprecated_files/eots_schemas_v2_5.py`. They are listed here because they appear to be critical for AI features or system operations and may not have counterparts in the primary schema files, or the deprecated versions are more detailed/used. Their definitions should ideally be moved to the appropriate primary schema files.

### AI Prediction Models

*   **`AIPredictionRequestV2_5`**: DTO to request an AI prediction.
*   **`AIPredictionV2_5`**: DTO representing a stored AI prediction with outcomes and performance.
*   **`AIPredictionPerformanceV2_5`**: DTO for tracking aggregate prediction performance.
*   **Usage**: Define the lifecycle of an AI prediction from request to performance analysis. Likely used by an AI prediction engine and performance tracking modules.

### AI Adaptation Models

*   **`AIAdaptationRequestV2_5`**: DTO to request an AI model adaptation.
*   **`AIAdaptationV2_5`**: DTO representing a stored AI adaptation and its status. (Config `extra='allow'`)
*   **`AIAdaptationPerformanceV2_5`**: DTO for tracking aggregate adaptation performance.
*   **Usage**: Define the lifecycle for AI model adaptations based on performance or new insights. Used by an AI adaptation engine.

### AI Hub Compliance Models

*   **`ControlPanelParametersV2_5` (Strict Version)**: A frozen, no-default version for AI Hub compliance.
    *   **Usage**: Ensures exact parameters are used in AI Hub data filtering. Used by `AIHubComplianceReportV2_5` and `FilteredDataBundleV2_5` (deprecated versions).
*   **`AIHubComplianceReportV2_5`**: DTO for reporting compliance of data filtering with control panel settings.
    *   **Usage**: Generated by `AIHubComplianceManagerV2_5` (inferred), interacts with `ComponentComplianceTrackerV2_5`.
*   **`FilteredDataBundleV2_5`**: DTO representing a `FinalAnalysisBundleV2_5` after applying control panel filters.
    *   **Usage**: Created by `AIHubComplianceManagerV2_5` (inferred), ensures dashboard components can consume data that respects user filters. Contains transformation logic in `create_from_bundle`.

### Advanced Options & Learning Cycle Models

*   **`AdvancedOptionsMetricsV2_5`**: DTO for advanced options metrics (LWPAI, VABAI, etc.).
    *   **Usage**: Referenced by `TickerContextAnalyzerSettings` in the main configuration, suggesting it's produced there or by a related metrics calculator.
*   **`MarketPattern`**: DTO for detected market patterns with detailed validation and supporting data fields.
*   **`StrategicRecommendationV2_5`**: A general DTO for strategic recommendations (potentially different from `ActiveRecommendationPayloadV2_5`).
*   **`LearningInsightV2_5`**: DTO for insights generated by the learning system, including adaptation suggestions and verification metrics.
*   **`MarketPredictionV2_5`, `PredictionPerformanceV2_5`, `PredictionConfigV2_5`**: Simpler set of prediction models compared to `AIPrediction...` set.
*   **`UnifiedPredictionResult`**: Bundles predictions and performance metrics.
*   **`OptimizationConfigV2_5`, `OptimizationMetricsV2_5`, `ParameterOptimizationResultV2_5`**: For system parameter optimization cycles.
*   **`SystemStateV2_5`**: High-level DTO nesting various AI health and learning status models (`AISystemHealthV2_5`, `LearningStatusV2_5`, `LearningMetricsV2_5`).
    *   **Usage**: Provides a snapshot of the overall AI system's state and health.

### Consolidated Settings Models (Deprecated)

*   **`DataManagementConfigV2_5`, `AnalyticsEngineConfigV2_5`, `AIConfigV2_5`, `TradingConfigV2_5`, `DashboardConfigV2_5` (large consolidated model)**:
    *   **Role**: These appear to be an alternative, more grouped way of structuring the main `EOTSConfigV2_5`.
    *   **Usage**: Defined in the deprecated file, they might represent a planned refactoring or a way to manage subsections of the configuration. `AIConfigV2_5` specifically groups AI prediction and performance tracker settings.

---

This summary provides an overview of the Pydantic models identified within the EOTS v2.5 system's `data_models` directory.
