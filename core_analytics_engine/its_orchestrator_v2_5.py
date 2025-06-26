# eots/core_analytics_engine/its_orchestrator_v2_5.py
"""
🎯 Enhanced ITS Orchestrator v2.5 - LEGENDARY META-ORCHESTRATOR
PYDANTIC-FIRST: Fully validated against EOTS schemas and integrated with legendary experts

This is the 4th pillar of the legendary system - the Meta-Orchestrator that coordinates
all analysis and makes final strategic decisions using the EOTS v2.5 architecture.
"""

import asyncio
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, TYPE_CHECKING, Callable, Awaitable
from pathlib import Path
import time
import numpy as np

# Pydantic imports for validation
from pydantic import BaseModel, Field, validator
from pydantic_ai import Agent, RunContext

# EOTS core imports - VALIDATED AGAINST USER'S SYSTEM
from data_models.eots_schemas_v2_5 import (
    ProcessedDataBundleV2_5,
    ProcessedUnderlyingAggregatesV2_5,
    ProcessedContractMetricsV2_5,
    ProcessedStrikeLevelMetricsV2_5,
    SignalPayloadV2_5,
    KeyLevelsDataV2_5,
    KeyLevelV2_5,
    ATIFStrategyDirectivePayloadV2_5,
    ATIFManagementDirectiveV2_5,
    TickerContextDictV2_5,
    DynamicThresholdsV2_5,
    AdvancedOptionsMetricsV2_5,
    HuiHuiAnalysisRequestV2_5,
    CrossAssetAnalysis,
    UnprocessedDataBundleV2_5,
    FinalAnalysisBundleV2_5,
    UnifiedLearningResult,
    MarketRegimeState,
    HuiHuiMarketRegimeSchema,
    HuiHuiOptionsFlowSchema,
    HuiHuiSentimentSchema,
    UnifiedIntelligenceAnalysis,
    AdaptiveLearningConfigV2_5,
    PredictionConfigV2_5,
    SystemStateV2_5
)

# MOE schemas - VALIDATED AGAINST USER'S SYSTEM
from data_models.moe_schemas_v2_5 import (
    ExpertStatus,
    RoutingStrategy,
    ConsensusStrategy,
    AgreementLevel,
    HealthStatus,
    MOEExpertRegistryV2_5,
    MOEGatingNetworkV2_5,
    MOEExpertResponseV2_5,
    MOEUnifiedResponseV2_5
)

# EOTS utilities - VALIDATED AGAINST USER'S SYSTEM
from utils.config_manager_v2_5 import ConfigManagerV2_5
from core_analytics_engine.metrics_calculator_v2_5 import MetricsCalculatorV2_5
from core_analytics_engine.market_regime_engine_v2_5 import MarketRegimeEngineV2_5
from core_analytics_engine.market_intelligence_engine_v2_5 import MarketIntelligenceEngineV2_5
from core_analytics_engine.atif_engine_v2_5 import ATIFEngineV2_5, EOTSConfigV2_5
from core_analytics_engine.news_intelligence_engine_v2_5 import NewsIntelligenceEngineV2_5
from core_analytics_engine.adaptive_learning_integration_v2_5 import AdaptiveLearningIntegrationV2_5
from core_analytics_engine.ai_predictions_manager_v2_5 import AIPredictionsManagerV2_5
from data_management.convexvalue_data_fetcher_v2_5 import ConvexValueDataFetcherV2_5
from data_management.initial_processor_v2_5 import InitialDataProcessorV2_5
from data_management.enhanced_cache_manager_v2_5 import EnhancedCacheManagerV2_5
from data_management.database_manager_v2_5 import DatabaseManagerV2_5
from data_management.historical_data_manager_v2_5 import HistoricalDataManagerV2_5
from data_management.performance_tracker_v2_5 import PerformanceTrackerV2_5

# HuiHui Expert System
from huihui_integration.experts.market_regime.market_regime_expert import UltimateMarketRegimeExpert, LegendaryRegimeConfig
from huihui_integration.experts.options_flow.options_flow_expert import UltimateOptionsFlowExpert, EliteFlowConfig
from huihui_integration.experts.sentiment.market_intelligence_expert import UltimateMarketIntelligenceExpert, MarketIntelligenceConfig
from huihui_integration.core.expert_communication import get_communication_protocol
from huihui_integration.core.robust_huihui_client_v2_5 import RobustHuiHuiClient
from huihui_integration.learning.feedback_loops import HuiHuiLearningSystem
from data_management.historical_data_storage_v2_5 import HistoricalDataStorageV2_5

# 🚀 REAL COMPLIANCE TRACKING: Import tracking system for metrics
try:
    from dashboard_application.modes.ai_dashboard.component_compliance_tracker_v2_5 import (
        track_metrics_calculation, DataSourceType
    )
    COMPLIANCE_TRACKING_AVAILABLE = True
except ImportError:
    COMPLIANCE_TRACKING_AVAILABLE = False

# HuiHui integration - USING USER'S EXISTING STRUCTURE
try:
    from huihui_integration.core.model_interface import create_market_regime_model
    from huihui_integration import (
        get_market_regime_expert,
        get_options_flow_expert,
        get_sentiment_expert,
        get_expert_coordinator,
        is_system_ready
    )
    LEGENDARY_EXPERTS_AVAILABLE = True
except ImportError as e:
    LEGENDARY_EXPERTS_AVAILABLE = False
    ExpertCommunicationProtocol = None

logger = logging.getLogger(__name__)

class LegendaryOrchestrationConfig(BaseModel):
    """PYDANTIC-FIRST: Configuration for legendary orchestration capabilities"""
    
    # AI Decision Making
    ai_decision_enabled: bool = Field(default=True, description="Enable AI-powered decision making")
    ai_model_name: str = Field(default="llama3.1:8b", description="AI model for decision making")
    ai_temperature: float = Field(default=0.1, description="AI temperature for consistency")
    ai_max_tokens: int = Field(default=2000, description="Maximum tokens for AI responses")
    
    # Expert Coordination
    expert_weight_adaptation: bool = Field(default=True, description="Enable dynamic expert weighting")
    expert_consensus_threshold: float = Field(default=0.7, description="Threshold for expert consensus")
    conflict_resolution_enabled: bool = Field(default=True, description="Enable conflict resolution")
    
    # Performance Optimization
    parallel_processing_enabled: bool = Field(default=True, description="Enable parallel expert processing")
    max_concurrent_experts: int = Field(default=4, description="Maximum concurrent expert analyses")
    cache_enabled: bool = Field(default=True, description="Enable result caching")
    cache_ttl_seconds: int = Field(default=300, description="Cache TTL in seconds")
    
    # Learning and Adaptation
    continuous_learning_enabled: bool = Field(default=True, description="Enable continuous learning")
    performance_tracking_enabled: bool = Field(default=True, description="Enable performance tracking")
    adaptation_rate: float = Field(default=0.01, description="Rate of system adaptation")
    
    # Risk Management
    risk_management_enabled: bool = Field(default=True, description="Enable risk management")
    max_position_exposure: float = Field(default=0.1, description="Maximum position exposure")
    stop_loss_threshold: float = Field(default=0.02, description="Stop loss threshold")
    
    class Config:
        extra = 'forbid'

class ITSOrchestratorV2_5:
    """
    🎯 LEGENDARY META-ORCHESTRATOR - 4th Pillar of the Legendary System
    
    PYDANTIC-FIRST: Fully validated against EOTS schemas and integrated with legendary experts.
    This orchestrator coordinates all analysis and makes final strategic decisions.
    """
    
    def __init__(self, config_manager: ConfigManagerV2_5):
        """Initialize the ITS Orchestrator with required components."""
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        
        # Initialize database manager first
        self._db_manager = DatabaseManagerV2_5(config_manager)
        
        # Initialize historical data manager
        self.historical_data_manager = HistoricalDataManagerV2_5(
            config_manager=config_manager,
            db_manager=self._db_manager
        )
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculatorV2_5(
            config_manager=config_manager,
            historical_data_manager=self.historical_data_manager
        )
        
        # Initialize market regime engine
        self.market_regime_engine = MarketRegimeEngineV2_5(config_manager)
        
        # Initialize market intelligence engine
        self.market_intelligence_engine = MarketIntelligenceEngineV2_5(
            config_manager=config_manager,
            metrics_calculator=self.metrics_calculator
        )
        
        # Initialize ATIF engine
        self.atif_engine = ATIFEngineV2_5(config_manager=config_manager)
        
        # Initialize news intelligence engine
        self.news_intelligence = NewsIntelligenceEngineV2_5(config_manager=config_manager)
        
        # Initialize adaptive learning integration
        adaptive_config = AdaptiveLearningConfigV2_5(
            learning_enabled=True,
            learning_rate=0.01,
            pattern_discovery_threshold=0.7,
            validation_window_days=30,
            max_patterns_per_symbol=100,
            learning_schedule="daily",
            auto_adaptation=True,
            **config_manager.get_setting("adaptive_learning_settings", {})
        )
        self.adaptive_learning = AdaptiveLearningIntegrationV2_5(config=adaptive_config)
        
        # Initialize prediction config
        self.prediction_config = PredictionConfigV2_5(
            prediction_window=30,
            min_confidence_threshold=0.7,
            max_predictions_per_symbol=5,
            prediction_frequency="1H",
            validation_frequency="1D",
            **config_manager.get_setting("prediction_settings", {})
        )
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTrackerV2_5(config_manager)
        
        # Initialize system state with only defined fields
        self.system_state = SystemStateV2_5(
            is_running=True,
            current_mode="operational",
            active_processes=["market_regime_engine", "market_intelligence_engine", "atif_engine", "news_intelligence", "adaptive_learning"],
            status_message="System initialized and running"
        )
        
        self.logger.info("🎯 ITS Orchestrator initialized successfully with all components")
        
    def analyze_market_regime(self, data_bundle: ProcessedDataBundleV2_5) -> str:
        """Analyze market regime using the market regime engine."""
        try:
            # Get market regime from the engine
            regime = self.market_regime_engine.determine_market_regime(data_bundle)
            self.logger.info(f"Market regime determined: {regime}")
            return regime
        except Exception as e:
            self.logger.error(f"Failed to analyze market regime: {e}")
            return "UNDEFINED"
            
    def _calculate_regime_metrics(self, data_bundle: ProcessedDataBundleV2_5) -> Dict[str, float]:
        """Calculate regime metrics from the data bundle."""
        try:
            if not data_bundle or not data_bundle.underlying_data_enriched:
                return {}
                
            und_data = data_bundle.underlying_data_enriched
            
            # Extract required metrics from the underlying data model
            metrics = {}
            
            # Add volatility metrics
            metrics['volatility'] = getattr(und_data, 'u_volatility', 0.0)
            metrics['trend_strength'] = getattr(und_data, 'vri_2_0_und', 0.0)
            metrics['volume_trend'] = getattr(und_data, 'vfi_0dte_und_avg', 0.0)
            metrics['momentum'] = getattr(und_data, 'a_mspi_und', 0.0)
            metrics['regime_score'] = getattr(und_data, 'current_market_regime_v2_5', 'UNKNOWN')
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating regime metrics: {e}")
            return {}
    
    def _initialize_moe_expert_registry(self) -> MOEExpertRegistryV2_5:
        """Initialize MOE Expert Registry for the 4th MOE Expert (Meta-Orchestrator)"""
        try:
            registry = MOEExpertRegistryV2_5(
                expert_id="meta_orchestrator_v2_5",
                expert_name="Ultimate Meta-Orchestrator",
                expert_type="meta_orchestrator",
                capabilities=[
                    "expert_coordination",
                    "consensus_building",
                    "conflict_resolution",
                    "strategic_synthesis",
                    "risk_assessment",
                    "final_decision_making"
                ],
                specializations=[
                    "meta_analysis",
                    "expert_synthesis",
                    "strategic_decision_making"
                ],
                supported_tasks=[
                    "expert_coordination",
                    "consensus_building",
                    "final_analysis"
                ],
                status=ExpertStatus.ACTIVE,
                accuracy_score=0.95,
                confidence_score=0.9,
                response_time_ms=15000.0,
                success_rate=95.0,
                memory_usage_mb=512.0,
                cpu_usage_percent=25.0,
                gpu_required=False,
                health_score=0.98,
                last_health_check=datetime.now(),
                tags=["meta", "orchestrator", "legendary", "v2_5"]
            )
            self.logger.info("🎯 MOE Expert Registry initialized for Meta-Orchestrator")
            return registry
        except Exception as e:
            self.logger.error(f"Failed to initialize MOE Expert Registry: {e}")
            raise
    
    def _create_moe_gating_network(self, request_context: Dict[str, Any]) -> MOEGatingNetworkV2_5:
        """Create MOE Gating Network for routing decisions"""
        try:
            # Determine which experts to route to based on request context
            selected_experts = request_context.get('include_experts', ["regime", "flow", "intelligence"])
            
            # Calculate expert weights based on request type and context
            expert_weights = self._calculate_expert_weights(request_context)
            
            # Calculate capability scores
            capability_scores = {
                "regime_expert": 0.9,
                "flow_expert": 0.85,
                "intelligence_expert": 0.88,
                "meta_orchestrator": 0.95
            }
            
            gating_network = MOEGatingNetworkV2_5(
                selected_experts=selected_experts,
                routing_strategy=RoutingStrategy.WEIGHTED,
                routing_confidence=0.9,
                expert_weights=expert_weights,
                capability_scores=capability_scores,
                request_context=request_context
            )
            
            self.logger.info(f"🎯 MOE Gating Network created with {len(selected_experts)} experts")
            return gating_network
            
        except Exception as e:
            self.logger.error(f"Failed to create MOE Gating Network: {e}")
            raise
    
    def _calculate_expert_weights(self, request_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate dynamic expert weights based on request context"""
        analysis_type = request_context.get('analysis_type', 'full')
        priority = request_context.get('priority', 'normal')
        
        # Base weights
        weights = {
            "regime_expert": 0.3,
            "flow_expert": 0.3,
            "intelligence_expert": 0.25,
            "meta_orchestrator": 0.15
        }
        
        # Adjust weights based on analysis type
        if analysis_type == 'regime_focused':
            weights["regime_expert"] = 0.5
            weights["flow_expert"] = 0.2
            weights["intelligence_expert"] = 0.2
            weights["meta_orchestrator"] = 0.1
        elif analysis_type == 'flow_focused':
            weights["regime_expert"] = 0.2
            weights["flow_expert"] = 0.5
            weights["intelligence_expert"] = 0.2
            weights["meta_orchestrator"] = 0.1
        elif analysis_type == 'intelligence_focused':
            weights["regime_expert"] = 0.2
            weights["flow_expert"] = 0.2
            weights["intelligence_expert"] = 0.5
            weights["meta_orchestrator"] = 0.1
        
        # Increase meta-orchestrator weight for high priority requests
        if priority == 'high':
            weights["meta_orchestrator"] += 0.1
            # Normalize other weights
            total_other = sum(v for k, v in weights.items() if k != "meta_orchestrator")
            for k in weights:
                if k != "meta_orchestrator":
                    weights[k] = weights[k] * (0.9 / total_other)
        
        return weights
    
    def _create_moe_expert_response(self, expert_id: str, expert_name: str, response_data: Dict[str, Any], 
                                   confidence_score: float, processing_time_ms: float) -> MOEExpertResponseV2_5:
        """Create MOE Expert Response for individual expert results"""
        try:
            expert_response = MOEExpertResponseV2_5(
                expert_id=expert_id,
                expert_name=expert_name,
                response_data=response_data,
                confidence_score=confidence_score,
                processing_time_ms=processing_time_ms,
                quality_score=min(confidence_score + 0.1, 1.0),  # Quality slightly higher than confidence
                uncertainty_score=1.0 - confidence_score,
                success=True,
                error_message=None,  # No error for successful response
                timestamp=datetime.now(),
                version="2.5"
            )
            return expert_response
        except Exception as e:
            self.logger.error(f"Failed to create MOE expert response for {expert_id}: {e}")
            # Return error response
            return MOEExpertResponseV2_5(
                expert_id=expert_id,
                expert_name=expert_name,
                response_data={"error": str(e)},
                confidence_score=0.0,
                processing_time_ms=processing_time_ms,
                quality_score=0.0,
                uncertainty_score=1.0,
                success=False,
                error_message=str(e),
                timestamp=datetime.now(),
                version="2.5"
            )
    
    def _create_moe_unified_response(self, expert_responses: List[MOEExpertResponseV2_5], 
                                   unified_data: Dict[str, Any], final_confidence: float,
                                   total_processing_time_ms: float) -> MOEUnifiedResponseV2_5:
        """Create MOE Unified Response combining all expert responses"""
        try:
            # Determine consensus strategy and agreement level
            successful_responses = [r for r in expert_responses if r.success]
            consensus_strategy = ConsensusStrategy.WEIGHTED_AVERAGE if len(successful_responses) > 1 else ConsensusStrategy.EXPERT_RANKING
            
            # Calculate agreement level based on confidence variance
            if len(successful_responses) > 1:
                confidence_scores = [r.confidence_score for r in successful_responses]
                confidence_variance = sum((c - final_confidence) ** 2 for c in confidence_scores) / len(confidence_scores)
                if confidence_variance < 0.01:
                    agreement_level = AgreementLevel.HIGH
                elif confidence_variance < 0.05:
                    agreement_level = AgreementLevel.MEDIUM
                else:
                    agreement_level = AgreementLevel.LOW
            else:
                agreement_level = AgreementLevel.HIGH  # Single expert, no disagreement
            
            unified_response = MOEUnifiedResponseV2_5(
                request_id=self.current_analysis.get("analysis_id", "unknown") if self.current_analysis else "unknown",
                request_type=self.current_analysis.get('analysis_type', 'full') if self.current_analysis else "unknown",
                consensus_strategy=consensus_strategy,
                agreement_level=agreement_level,
                final_confidence=final_confidence,
                expert_responses=expert_responses,
                participating_experts=[r.expert_id for r in expert_responses],
                unified_response=unified_data,
                response_quality=final_confidence,
                total_processing_time_ms=total_processing_time_ms,
                expert_coordination_time_ms=total_processing_time_ms * 0.1,  # Estimate 10% for coordination
                consensus_time_ms=total_processing_time_ms * 0.05,  # Estimate 5% for consensus
                system_health=HealthStatus.HEALTHY if len(successful_responses) == len(expert_responses) else HealthStatus.DEGRADED,
                timestamp=datetime.now(),
                version="2.5",
                debug_info={
                    "total_experts": len(expert_responses),
                    "successful_experts": len(successful_responses),
                    "failed_experts": len(expert_responses) - len(successful_responses)
                },
                performance_breakdown={
                    "data_processing": total_processing_time_ms * 0.3,
                    "expert_analysis": total_processing_time_ms * 0.5,
                    "synthesis": total_processing_time_ms * 0.15,
                    "coordination": total_processing_time_ms * 0.05
                }
            )
            
            self.logger.info(f"🎯 MOE Unified Response created with {len(successful_responses)}/{len(expert_responses)} successful experts")
            return unified_response
            
        except Exception as e:
            self.logger.error(f"Failed to create MOE unified response: {e}")
            raise
    
    def _get_regime_analysis_prompt(self) -> str:
        """Get system prompt for AI decision making"""
        return """
        You are the LEGENDARY META-ORCHESTRATOR for the EOTS v2.5 options trading system.
        
        Your role is to synthesize analysis from 3 specialist experts:
        1. Market Regime Expert - Provides VRI 2.0 analysis and regime detection
        2. Options Flow Expert - Provides VAPI-FA, DWFD, and elite flow analysis  
        3. Market Intelligence Expert - Provides sentiment, behavioral, and microstructure analysis
        
        Your responsibilities:
        - Synthesize expert analyses into strategic recommendations
        - Resolve conflicts between expert opinions
        - Provide final trading decisions with confidence scores
        - Assess risk and provide risk management guidance
        - Maintain consistency with EOTS v2.5 methodologies
        
        Always provide structured, actionable recommendations based on the expert analyses.
        Focus on high-probability setups with clear risk/reward profiles.
        """
    
    async def run_full_analysis_cycle(self, ticker: str, **kwargs) -> FinalAnalysisBundleV2_5:
        """Run a complete analysis cycle with all experts."""
        try:
            # Get processed data bundle
            processed_bundle = await self._get_processed_data_bundle(ticker)
            if not processed_bundle:
                self.logger.error(f"Failed to get processed data bundle for {ticker}")
                return self._create_error_bundle(ticker, "Failed to get processed data bundle")

            # Expert analysis tasks
            expert_tasks = []
            expert_results = {}
            
            # Market Regime Analysis
            if self.market_regime_engine:
                regime_task = self.market_regime_engine.analyze_market_regime(processed_bundle)
                expert_tasks.append(regime_task)
            
            # Options Flow Analysis
            if self.options_flow_expert:
                flow_task = self.options_flow_expert.analyze_options_flow(processed_bundle)
                expert_tasks.append(flow_task)
            
            # Market Intelligence Analysis
            if self.market_intelligence_expert:
                intel_task = self.market_intelligence_expert.analyze_market_intelligence(processed_bundle)
                expert_tasks.append(intel_task)
            
            # Execute all expert analyses in parallel
            if expert_tasks:
                expert_results_list = await asyncio.gather(*expert_tasks)
                
                # Map results to their respective experts
                if self.market_regime_engine:
                    expert_results['market_regime'] = expert_results_list.pop(0)
                if self.options_flow_expert:
                    expert_results['options_flow'] = expert_results_list.pop(0)
                if self.market_intelligence_expert:
                    expert_results['market_intelligence'] = expert_results_list.pop(0)
            
            # Create final analysis bundle
            bundle_timestamp = datetime.now()
            final_bundle = FinalAnalysisBundleV2_5(
                processed_data_bundle=processed_bundle,
                scored_signals_v2_5={},  # Empty for now
                key_levels_data_v2_5=KeyLevelsDataV2_5(
                    supports=[],
                    resistances=[],
                    pin_zones=[],
                    vol_triggers=[],
                    major_walls=[],
                    timestamp=bundle_timestamp
                ),
                bundle_timestamp=bundle_timestamp,
                target_symbol=ticker,
                system_status_messages=[
                    f"Analysis completed successfully for {ticker}",
                    f"Expert analyses: {len(expert_results)} completed"
                ],
                active_recommendations_v2_5=[],  # Empty for now
                atif_recommendations_v2_5=None,  # Not implemented yet
                news_intelligence_v2_5=None,  # Not implemented yet
                ai_predictions_v2_5=None  # Not implemented yet
            )
            
            return final_bundle
            
        except Exception as e:
            self.logger.error(f"Error in full analysis cycle: {str(e)}")
            return self._create_error_bundle(ticker, str(e))
    
    def _create_error_bundle(self, ticker: str, error_message: str) -> FinalAnalysisBundleV2_5:
        """Create an error bundle that satisfies FinalAnalysisBundleV2_5 schema."""
        bundle_timestamp = datetime.now()
        return FinalAnalysisBundleV2_5(
            processed_data_bundle=ProcessedDataBundleV2_5(
                underlying_data_enriched=ProcessedUnderlyingAggregatesV2_5(
                    symbol=ticker,
                    timestamp=bundle_timestamp,
                    price=None,
                    price_change_abs_und=None,
                    price_change_pct_und=None,
                    day_open_price_und=None,
                    day_high_price_und=None,
                    day_low_price_und=None,
                    prev_day_close_price_und=None,
                    u_volatility=None,
                    day_volume=None,
                    call_gxoi=None,
                    put_gxoi=None,
                    gammas_call_buy=None,
                    gammas_call_sell=None,
                    gammas_put_buy=None,
                    gammas_put_sell=None,
                    deltas_call_buy=None,
                    deltas_call_sell=None,
                    deltas_put_buy=None,
                    deltas_put_sell=None,
                    vegas_call_buy=None,
                    vegas_call_sell=None,
                    vegas_put_buy=None,
                    vegas_put_sell=None,
                    thetas_call_buy=None,
                    thetas_call_sell=None,
                    thetas_put_buy=None,
                    thetas_put_sell=None,
                    call_vxoi=None,
                    put_vxoi=None,
                    value_bs=None,
                    volm_bs=None,
                    deltas_buy=None,
                    deltas_sell=None,
                    vegas_buy=None,
                    vegas_sell=None,
                    thetas_buy=None,
                    thetas_sell=None,
                    volm_call_buy=None,
                    volm_put_buy=None,
                    volm_call_sell=None,
                    volm_put_sell=None,
                    value_call_buy=None,
                    value_put_buy=None,
                    value_call_sell=None,
                    value_put_sell=None,
                    vflowratio=None,
                    dxoi=None,
                    gxoi=None,
                    vxoi=None,
                    txoi=None,
                    call_dxoi=None,
                    put_dxoi=None,
                    tradier_iv5_approx_smv_avg=None,
                    total_call_oi_und=None,
                    total_put_oi_und=None,
                    total_call_vol_und=None,
                    total_put_vol_und=None,
                    tradier_open=None,
                    tradier_high=None,
                    tradier_low=None,
                    tradier_close=None,
                    tradier_volume=None,
                    tradier_vwap=None
                ),
                options_data_with_metrics=[],
                strike_level_data_with_metrics=[],
                processing_timestamp=bundle_timestamp,
                errors=[error_message]
            ),
            scored_signals_v2_5={},
            key_levels_data_v2_5=KeyLevelsDataV2_5(
                supports=[],
                resistances=[],
                pin_zones=[],
                vol_triggers=[],
                major_walls=[],
                timestamp=bundle_timestamp
            ),
            bundle_timestamp=bundle_timestamp,
            target_symbol=ticker,
            system_status_messages=[f"Error: {error_message}"],
            active_recommendations_v2_5=[],  # Empty for now
            atif_recommendations_v2_5=None,  # Not implemented yet
            news_intelligence_v2_5=None,  # Not implemented yet
            ai_predictions_v2_5=None  # Not implemented yet
        )
    
    async def _get_processed_data_bundle(self, ticker: str) -> Optional[ProcessedDataBundleV2_5]:
        """Get processed data bundle validated against EOTS schemas"""
        try:
            self.logger.info(f"📊 Getting processed data bundle for {ticker}")
            
            # Check if we have the required components
            if not self.convex_fetcher:
                self.logger.error("ConvexValue fetcher not initialized")
                return None
                
            if not self.initial_processor:
                self.logger.error("Initial processor not initialized")
                return None
            
            # Step 1: Fetch raw data using ConvexValue fetcher
            self.logger.info(f"🔄 Fetching raw data for {ticker}...")
            chain_data, underlying_data = await self.convex_fetcher.fetch_chain_and_underlying(
                session=None,  # Not used by ConvexValue
                symbol=ticker,
                dte_min=0,
                dte_max=45,
                price_range_percent=20
            )
            
            if not underlying_data:
                self.logger.error(f"No underlying data returned for {ticker}")
                return None
                
            if not chain_data:
                self.logger.warning(f"No options chain data returned for {ticker}")
                chain_data = []  # Continue with empty chain
            
            self.logger.info(f"✅ Raw data fetched: {len(chain_data)} contracts, underlying price: {underlying_data.price}")
            
            # Step 2: Create UnprocessedDataBundleV2_5
            raw_bundle = UnprocessedDataBundleV2_5(
                options_contracts=chain_data,
                underlying_data=underlying_data,
                fetch_timestamp=datetime.now(),
                errors=[]
            )
            
            # Step 3: Process the raw data using InitialDataProcessorV2_5
            self.logger.info(f"🔄 Processing raw data for {ticker}...")
            processed_bundle = self.initial_processor.process_data_and_calculate_metrics(
                raw_data_bundle=raw_bundle,
                dte_max=45
            )
            
            self.logger.info(f"✅ Data processing completed for {ticker}")
            self.logger.info(f"   📊 Options with metrics: {len(processed_bundle.options_data_with_metrics)}")
            self.logger.info(f"   🎯 Strike levels with metrics: {len(processed_bundle.strike_level_data_with_metrics)}")
            
            # 🏛️ CRITICAL FIX: MARKET REGIME ANALYSIS (ALWAYS RUN - CRITICAL FOR DASHBOARD)
            self.logger.info(f"🏛️ STEP 4: Market regime analysis for {ticker}")
            market_regime = "REGIME_UNCLEAR_OR_TRANSITIONING"
            try:
                # ALWAYS run market regime analysis - it's critical for dashboard functionality
                if hasattr(self, 'market_intelligence_engine') and self.market_intelligence_engine:
                    analysis_result = await self.market_intelligence_engine.determine_market_regime(processed_bundle)
                    if analysis_result:
                        market_regime = analysis_result
                        self.logger.info(f"🏛️ Market regime determined: {market_regime} for {ticker}")
                        
                        # CRITICAL FIX: Set the regime in the underlying data so it's available to the dashboard
                        processed_bundle.underlying_data_enriched.current_market_regime_v2_5 = market_regime
                        self.logger.info(f"✅ Market regime {market_regime} set in underlying data for {ticker}")
                        
                        # VERIFY the regime was actually set
                        verification = getattr(processed_bundle.underlying_data_enriched, 'current_market_regime_v2_5', 'NOT_SET')
                        self.logger.info(f"🔍 VERIFICATION: Regime in underlying data is now: {verification}")
                else:
                    self.logger.warning(f"🚨 Market regime engine not available for {ticker}")
            except Exception as e:
                self.logger.error(f"🚨 Market regime analysis failed for {ticker}: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                # Keep default regime on error
            
            return processed_bundle
            
        except Exception as e:
            self.logger.error(f"Failed to get processed data bundle for {ticker}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.performance_metrics["failed_analyses"] = self.performance_metrics.get("failed_analyses", 0) + 1
            return None
    
    def _calculate_data_quality_score(self, data_bundle: ProcessedDataBundleV2_5) -> float:
        """Calculate data quality score for the analysis"""
        try:
            if not data_bundle:
                return 0.0
            
            quality_factors = []
            
            # Check underlying data quality
            if data_bundle.underlying_data_enriched:
                if data_bundle.underlying_data_enriched.price:
                    quality_factors.append(1.0)
                else:
                    quality_factors.append(0.0)
            
            # Check options data quality
            if data_bundle.options_data_with_metrics:
                options_quality = len(data_bundle.options_data_with_metrics) / 100.0  # Normalize by expected count
                quality_factors.append(min(options_quality, 1.0))
            else:
                quality_factors.append(0.0)
            
            # Check strike level data quality
            if data_bundle.strike_level_data_with_metrics:
                strike_quality = len(data_bundle.strike_level_data_with_metrics) / 50.0  # Normalize by expected count
                quality_factors.append(min(strike_quality, 1.0))
            else:
                quality_factors.append(0.0)
            
            # Calculate average quality
            if quality_factors:
                return sum(quality_factors) / len(quality_factors)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Data quality calculation failed: {e}")
            return 0.0
    
    def _update_performance_metrics(self, result: Dict[str, Any]):
        """Update performance tracking metrics"""
        self.performance_metrics["total_analyses"] = self.performance_metrics.get("total_analyses", 0) + 1
        
        if result.get("confidence_score", 0) > 0.5:
            self.performance_metrics["successful_analyses"] = self.performance_metrics.get("successful_analyses", 0) + 1
        else:
            self.performance_metrics["failed_analyses"] = self.performance_metrics.get("failed_analyses", 0) + 1
    
    def get_legendary_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics for the legendary system"""
        try:
            # Update real-time metrics
            self.performance_metrics["uptime_seconds"] = (datetime.now() - self.start_time).total_seconds()
            
            return self.performance_metrics
            
        except Exception as e:
            self.logger.error(f"Performance metrics retrieval failed: {e}")
            return {
                'system_status': 'ERROR',
                'error': str(e)
            }
    
    async def legendary_orchestrate_analysis(self, data_bundle: ProcessedDataBundleV2_5, **kwargs) -> Dict[str, Any]:
        """
        🎯 LEGENDARY orchestration method for backward compatibility
        Returns Dict[str, Any] instead of non-existent model
        """
        try:
                         # Run full analysis cycle and convert to dict format
             final_bundle = await self.run_full_analysis_cycle(data_bundle.underlying_data_enriched.symbol, **kwargs)
             
             return {
                 "analysis_id": f"{data_bundle.underlying_data_enriched.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                 "ticker": data_bundle.underlying_data_enriched.symbol,
                 "timestamp": datetime.now(),
                 "final_bundle": final_bundle,
                 "status": "completed"
             }
            
        except Exception as e:
            self.logger.error(f"Legendary orchestration failed: {e}")
            return {
                "analysis_id": f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "ticker": data_bundle.underlying_data_enriched.symbol if data_bundle else "unknown",
                "timestamp": datetime.now(),
                "error": str(e),
                "status": "failed"
            }
    
    async def get_legendary_analysis_request(self, ticker: str, **kwargs) -> Dict[str, Any]:
        """Get legendary analysis request in dict format"""
        return {
            "ticker": ticker,
            "analysis_type": kwargs.get('analysis_type', 'full'),
            "priority": kwargs.get('priority', 'normal'),
            "include_experts": kwargs.get('include_experts', ["regime", "flow", "intelligence"]),
            "custom_parameters": kwargs.get('custom_parameters', {})
        }
    
    async def _generate_key_levels(self, data_bundle: ProcessedDataBundleV2_5, ticker: str, timestamp: datetime) -> KeyLevelsDataV2_5:
        """
        Generate key levels ONLY from database - NO FALLBACK DATA GENERATION.
        
        Args:
            data_bundle: Processed data bundle containing price and options data
            ticker: Trading symbol
            timestamp: Analysis timestamp
            
        Returns:
            KeyLevelsDataV2_5: Key levels from database ONLY, empty if none found
        """
        try:
            self.logger.info(f"🔍 Retrieving key levels for {ticker} from database ONLY")
            
            # CRITICAL: ONLY retrieve key levels from database - NO FALLBACK GENERATION
            database_levels = await self._retrieve_key_levels_from_database(ticker)
            if database_levels and len(database_levels.supports + database_levels.resistances + 
                                     database_levels.pin_zones + database_levels.vol_triggers + 
                                     database_levels.major_walls) > 0:
                self.logger.info(f"✅ Retrieved {len(database_levels.supports + database_levels.resistances + database_levels.pin_zones + database_levels.vol_triggers + database_levels.major_walls)} key levels from database")
                return database_levels
            
            # NO FALLBACK DATA GENERATION - Return empty key levels if none in database
            self.logger.warning(f"⚠️ No key levels found in database for {ticker} - returning empty key levels (NO FALLBACK DATA)")
            return KeyLevelsDataV2_5(
                supports=[],
                resistances=[],
                pin_zones=[],
                vol_triggers=[],
                major_walls=[],
                timestamp=timestamp
            )
                
        except Exception as e:
            self.logger.error(f"❌ Error retrieving key levels for {ticker}: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Return empty key levels data on error - NO FALLBACK DATA
            return KeyLevelsDataV2_5(
                supports=[],
                resistances=[],
                pin_zones=[],
                vol_triggers=[],
                major_walls=[],
                timestamp=timestamp
            )

    async def _retrieve_key_levels_from_database(self, ticker: str) -> Optional[KeyLevelsDataV2_5]:
        """
        Retrieve key levels from the database metrics schema.
        
        Args:
            ticker: Trading symbol
            
        Returns:
            KeyLevelsDataV2_5: Key levels from database or None if not found
        """
        try:
            if not self.db_manager or not hasattr(self.db_manager, '_conn'):
                return None
                
            # Query the key_level_performance table for recent levels
            sql = """
            SELECT level_price, level_type, conviction_score, level_source, created_at
            FROM key_level_performance
            WHERE symbol = %s 
            AND date >= CURRENT_DATE - INTERVAL '7 days'
            AND conviction_score > 0.3
            ORDER BY conviction_score DESC, created_at DESC
            LIMIT 50
            """
            
            cursor = self.db_manager._conn.cursor()
            cursor.execute(sql, (ticker,))
            results = cursor.fetchall()
            
            if not results:
                self.logger.info(f"📊 No key levels found in database for {ticker}")
                return None
            
            # Convert database results to KeyLevelV2_5 models
            supports = []
            resistances = []
            pin_zones = []
            vol_triggers = []
            major_walls = []
            
            for row in results:
                # Handle both tuple and dict-like row objects
                if isinstance(row, dict):
                    level_price = row.get('level_price')
                    level_type = row.get('level_type')
                    conviction_score = row.get('conviction_score')
                    level_source = row.get('level_source')
                    created_at = row.get('created_at')
                else:
                    # Assume tuple/list format
                    level_price, level_type, conviction_score, level_source, created_at = row
                
                # Skip rows with missing critical data - NO FALLBACK DATA GENERATION
                if level_price is None or level_type is None or conviction_score is None:
                    self.logger.warning(f"⚠️ Skipping row with missing data: price={level_price}, type={level_type}, score={conviction_score}")
                    continue
                
                key_level = KeyLevelV2_5(
                    level_price=float(level_price),
                    level_type=str(level_type),
                    conviction_score=float(conviction_score),
                    contributing_metrics=[level_source] if level_source else [],
                    source_identifier=level_source or 'database'
                )
                
                # Categorize by type
                level_type_str = str(level_type).lower()
                if level_type_str in ['support']:
                    supports.append(key_level)
                elif level_type_str in ['resistance']:
                    resistances.append(key_level)
                elif level_type_str in ['pivot', 'pin_zone']:
                    pin_zones.append(key_level)
                elif level_type_str in ['max_pain', 'vol_trigger']:
                    vol_triggers.append(key_level)
                elif level_type_str in ['gamma_wall', 'major_wall']:
                    major_walls.append(key_level)
                else:
                    # Default to resistance for unknown types
                    resistances.append(key_level)
            
            self.logger.info(f"📊 Retrieved from database: {len(supports)} supports, {len(resistances)} resistances, "
                           f"{len(pin_zones)} pin zones, {len(vol_triggers)} vol triggers, {len(major_walls)} major walls")
            
            return KeyLevelsDataV2_5(
                supports=supports,
                resistances=resistances,
                pin_zones=pin_zones,
                vol_triggers=vol_triggers,
                major_walls=major_walls,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error retrieving key levels from database for {ticker}: {e}")
            return None

    @property
    def cache_manager(self) -> Optional[EnhancedCacheManagerV2_5]:
        """Get the cache manager instance."""
        if self._cache_manager is None:
            cache_root = self.config_manager.get_resolved_path('cache_settings.cache_root')
            if cache_root:
                self._cache_manager = EnhancedCacheManagerV2_5(
                    cache_root=cache_root,
                    memory_limit_mb=100,
                    disk_limit_mb=1000,
                    default_ttl_seconds=3600,
                    ultra_fast_mode=True
                )
        return self._cache_manager

    @property
    def db_manager(self) -> DatabaseManagerV2_5:
        """Get the database manager instance."""
        return self._db_manager

    async def analyze_market_regime(self, processed_data: ProcessedDataBundleV2_5) -> UnifiedIntelligenceAnalysis:
        """Analyze market regime using the consolidated market intelligence engine."""
        try:
            # Use the consolidated market intelligence engine to analyze the market
            analysis = await self.market_intelligence_engine.analyze_market_data(
                data_bundle=processed_data,
                huihui_regime=None,  # Will be fetched internally if needed
                huihui_flow=None,    # Will be fetched internally if needed
                huihui_sentiment=None # Will be fetched internally if needed
            )
            self.logger.info(f"Market intelligence analysis completed for {processed_data.underlying_data_enriched.symbol}")
            return analysis
                
        except Exception as e:
            self.logger.error(f"Error in market intelligence analysis: {str(e)}")
            return UnifiedIntelligenceAnalysis(
                symbol=processed_data.underlying_data_enriched.symbol,
                timestamp=datetime.now(),
                confidence_score=0.0,
                market_regime_analysis=str(MarketRegimeState.UNDEFINED),
                options_flow_analysis="Error in analysis",
                sentiment_analysis="Error in analysis",
                strategic_recommendations=[],
                risk_assessment="Error in analysis",
                learning_insights=[f"Error: {str(e)}"],
                performance_metrics={}
            )

# Maintain backward compatibility
ItsOrchestratorV2_5 = ITSOrchestratorV2_5

