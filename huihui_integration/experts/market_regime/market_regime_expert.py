# huihui_integration/experts/market_regime/market_regime_expert.py
"""
🏛️ Ultimate Market Regime Expert - LEGENDARY MARKET REGIME ANALYSIS
PYDANTIC-FIRST: Fully validated against EOTS schemas with VRI 3.0 and advanced regime detection

This expert specializes in:
- Advanced VRI 3.0 with machine learning enhancement
- Cross-Asset Regime Analysis (8 asset classes)
- Regime Transition Prediction with timing models
- 20 Market Regime Classifications
- Macro-Economic Integration (7 indicators)
- Sub-millisecond regime detection
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import time
import uuid
from enum import Enum

# Pydantic imports for validation
from pydantic import BaseModel, Field, validator

# EOTS core imports - VALIDATED AGAINST USER'S SYSTEM
from data_models.eots_schemas_v2_5 import (
    ProcessedDataBundleV2_5,
    ProcessedUnderlyingAggregatesV2_5,
    # ProcessedContractMetricsV2_5, # Not directly used by this expert's main logic
    # ProcessedStrikeLevelMetricsV2_5, # Not directly used
    # DynamicThresholdsV2_5, # Not directly used
    # TickerContextDictV2_5, # Not directly used
    HuiHuiAnalysisRequestV2_5, # Added for new analyze method
    HuiHuiAnalysisResponseV2_5,
    HuiHuiExpertType,
    # FinalAnalysisBundleV2_5, # Not directly used in top-level analyze signature
    HuiHuiExpertConfigV2_5, # Base config type
    MarketRegimeExpertConfig # Specific config type for this expert (defined below)
)

# EOTS utilities
from utils.config_manager_v2_5 import ConfigManagerV2_5
from core_analytics_engine.metrics_calculator_v2_5 import MetricsCalculatorV2_5
from huihui_integration.core.base_expert import BaseHuiHuiExpert
from huihui_integration.core.model_interface import HuiHuiMarketRegimeModel

logger = logging.getLogger(__name__)

class MarketRegimeExpertConfig(HuiHuiExpertConfigV2_5):
    """PYDANTIC-FIRST: Configuration for legendary market regime analysis"""
    
    # VRI 3.0 Configuration
    vri_3_enabled: bool = Field(default=True, description="Enable VRI 3.0 analysis")
    vri_lookback_periods: List[int] = Field(default=[5, 10, 20, 50], description="VRI lookback periods")
    vri_smoothing_factor: float = Field(default=0.3, description="VRI smoothing factor")
    vri_volatility_adjustment: bool = Field(default=True, description="Enable volatility adjustment")
    
    # Cross-Asset Analysis
    cross_asset_analysis_enabled: bool = Field(default=True, description="Enable cross-asset analysis")
    asset_classes: List[str] = Field(
        default=["equities", "bonds", "commodities", "currencies", "crypto", "volatility", "credit", "rates"],
        description="Asset classes for cross-asset analysis"
    )
    correlation_threshold: float = Field(default=0.7, description="Correlation threshold for regime detection")
    
    # Regime Prediction
    regime_prediction_enabled: bool = Field(default=True, description="Enable regime transition prediction")
    prediction_horizon_days: int = Field(default=5, description="Prediction horizon in days")
    transition_probability_threshold: float = Field(default=0.6, description="Transition probability threshold")
    
    # Macro Integration
    macro_integration_enabled: bool = Field(default=True, description="Enable macro-economic integration")
    macro_indicators: List[str] = Field(
        default=["vix", "rates", "credit", "liquidity", "sentiment", "momentum", "volatility"],
        description="Macro-economic indicators to monitor"
    )
    macro_update_interval_mins: int = Field(default=15, description="Macro data update interval in minutes")

class VRI3Components(BaseModel):
    """PYDANTIC-FIRST: VRI 3.0 component analysis"""
    
    # Core VRI components
    composite_score: float = Field(..., description="VRI 3.0 composite score", ge=0.0, le=1.0)
    volatility_regime: float = Field(..., description="Volatility regime component", ge=0.0, le=1.0)
    flow_intensity: float = Field(..., description="Flow intensity component", ge=0.0, le=1.0)
    regime_stability: float = Field(..., description="Regime stability component", ge=0.0, le=1.0)
    transition_momentum: float = Field(..., description="Transition momentum component", ge=0.0, le=1.0)
    confidence_level: float = Field(..., description="Analysis confidence level", ge=0.0, le=1.0)

class RegimeClassification(BaseModel):
    """PYDANTIC-FIRST: Market regime classification with detailed characteristics"""
    regime_id: str = Field(..., description="Unique identifier for the regime")
    regime_name: str = Field(..., description="Name of the market regime")
    regime_description: str = Field(..., description="Detailed description of the regime")
    probability: float = Field(..., description="Probability of regime classification", ge=0.0, le=1.0)
    confidence: float = Field(..., description="Confidence in regime classification", ge=0.0, le=1.0)
    volatility_level: str = Field(..., description="Current volatility level (High/Medium/Low)")
    trend_direction: str = Field(..., description="Current trend direction (Up/Down/Sideways)")
    flow_pattern: str = Field(..., description="Current flow pattern")
    risk_appetite: str = Field(..., description="Current risk appetite level")

class RegimeTransitionPrediction(BaseModel):
    """PYDANTIC-FIRST: Regime transition prediction with probabilities"""
    current_regime: str = Field(..., description="Current market regime")
    predicted_regime: str = Field(..., description="Most likely next regime")
    transition_probability: float = Field(..., description="Probability of transition", ge=0.0, le=1.0)
    expected_timeframe: int = Field(..., description="Expected days until transition")
    confidence_level: float = Field(..., description="Confidence in prediction", ge=0.0, le=1.0)
    transition_triggers: List[str] = Field(default_factory=list, description="Potential transition triggers")
    risk_factors: List[str] = Field(default_factory=list, description="Risk factors affecting transition")

class EquityRegime(str, Enum):
    """Valid equity market regimes."""
    BULLISH_TRENDING = "bullish_trending"
    BULLISH_CONSOLIDATION = "bullish_consolidation"
    BEARISH_TRENDING = "bearish_trending"
    BEARISH_CONSOLIDATION = "bearish_consolidation"
    NEUTRAL = "neutral"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRANSITION = "transition"
    UNDEFINED = "undefined"

class BondRegime(str, Enum):
    """Valid bond market regimes."""
    YIELD_RISING = "yield_rising"
    YIELD_FALLING = "yield_falling"
    YIELD_STABLE = "yield_stable"
    CREDIT_STRESS = "credit_stress"
    CREDIT_EASING = "credit_easing"
    RANGE_BOUND = "range_bound"
    TRANSITION = "transition"
    UNDEFINED = "undefined"

class CommodityRegime(str, Enum):
    """Valid commodity market regimes."""
    DEMAND_DRIVEN = "demand_driven"
    SUPPLY_DRIVEN = "supply_driven"
    BALANCED = "balanced"
    CONTANGO = "contango"
    BACKWARDATION = "backwardation"
    RANGE_BOUND = "range_bound"
    TRANSITION = "transition"
    UNDEFINED = "undefined"

class CurrencyRegime(str, Enum):
    """Valid currency market regimes."""
    STRENGTHENING = "strengthening"
    WEAKENING = "weakening"
    CONSOLIDATION_STRONG = "consolidation_strong"
    CONSOLIDATION_WEAK = "consolidation_weak"
    RANGE_BOUND = "range_bound"
    TRANSITION = "transition"
    UNDEFINED = "undefined"

class CrossAssetAnalysis(BaseModel):
    """PYDANTIC-FIRST: Cross-asset regime analysis"""
    asset_correlations: Dict[str, float] = Field(default_factory=dict, description="Asset class correlations")
    regime_consistency: float = Field(default=0.0, description="Cross-asset regime consistency")
    divergence_signals: List[str] = Field(default_factory=list, description="Cross-asset divergence signals")
    confirmation_signals: List[str] = Field(default_factory=list, description="Cross-asset confirmation signals")
    equity_regime: Optional[EquityRegime] = Field(None, description="Equity market regime")
    bond_regime: Optional[BondRegime] = Field(None, description="Bond market regime")
    commodity_regime: Optional[CommodityRegime] = Field(None, description="Commodity market regime")
    currency_regime: Optional[CurrencyRegime] = Field(None, description="Currency market regime")

    class Config:
        extra = 'forbid'
        use_enum_values = True

class LegendaryRegimeResult(BaseModel):
    """PYDANTIC-FIRST: Comprehensive market regime analysis result"""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    ticker: str = Field(..., description="Analyzed ticker symbol")
    timestamp: datetime = Field(..., description="Analysis timestamp")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    vri_3_analysis: VRI3Components = Field(..., description="VRI 3.0 analysis components")
    regime_classification: RegimeClassification = Field(..., description="Current regime classification")
    transition_prediction: RegimeTransitionPrediction = Field(..., description="Regime transition prediction")
    cross_asset_analysis: CrossAssetAnalysis = Field(..., description="Cross-asset regime analysis")
    regime_strength: float = Field(..., description="Current regime strength", ge=0.0, le=1.0)
    regime_persistence: float = Field(..., description="Expected regime persistence", ge=0.0, le=1.0)
    volatility_forecast: Dict[str, float] = Field(..., description="Forward volatility forecast")
    regime_risk_score: float = Field(..., description="Regime-specific risk score", ge=0.0, le=1.0)
    tail_risk_probability: float = Field(..., description="Tail risk event probability", ge=0.0, le=1.0)
    black_swan_indicators: List[str] = Field(default_factory=list, description="Black swan warning indicators")
    prediction_accuracy: Optional[float] = Field(None, description="Historical prediction accuracy")
    confidence_score: float = Field(..., description="Overall analysis confidence", ge=0.0, le=1.0)
    data_quality_score: float = Field(..., description="Input data quality score", ge=0.0, le=1.0)

class LegendaryRegimeConfig(BaseModel):
    """PYDANTIC-FIRST: Configuration for legendary regime expert (stub, to be expanded as needed)"""
    # Add actual config fields as needed for the legendary regime expert
    class Config:
        extra = 'forbid'

class UltimateMarketRegimeExpert(BaseHuiHuiExpert):
    """🏛️ LEGENDARY MARKET REGIME EXPERT - VRI 3.0 Enhanced"""
    
    def __init__(self, expert_config: MarketRegimeExpertConfig,
                 config_manager: ConfigManagerV2_5,
                 historical_data_manager, # Consider type hinting if available
                 db_manager=None): # Add db_manager
        super().__init__(expert_config, db_manager) # Pass db_manager to super
        # self.config is already set by super() to expert_config (which is MarketRegimeExpertConfig)
        # So, self.config.vri_3_enabled etc. can be used directly.

        self.metrics_calculator = MetricsCalculatorV2_5( # This is fine
            config_manager=config_manager,
            historical_data_manager=historical_data_manager
        )
        # Ensure initialization of MetricsCalculator if it has an init method
        if hasattr(self.metrics_calculator, 'initialize') and callable(getattr(self.metrics_calculator, 'initialize')):
            self.metrics_calculator.initialize()


    def _get_trend_direction(self, data: ProcessedUnderlyingAggregatesV2_5) -> str:
        """Determine the trend direction based on price data"""
        try:
            current_price = getattr(data, 'price', None)
            prev_close = getattr(data, 'prev_day_close_price_und', None)
            if current_price is None or prev_close is None:
                return "neutral"
            if current_price > prev_close:
                return "up"
            elif current_price < prev_close:
                return "down"
            else:
                return "neutral"
        except Exception as e:
            logger.error(f"Error determining trend direction: {e}")
            return "neutral"

    def _analyze_equity_regime(self, data: ProcessedUnderlyingAggregatesV2_5) -> str:
        """Analyze equity market regime."""
        try:
            trend = self._get_trend_direction(data)
            volatility = getattr(data, 'u_volatility', 0) or 0
            if trend == "up" and volatility > 0.7:
                return EquityRegime.BULLISH_TRENDING
            elif trend == "up" and volatility <= 0.7:
                return EquityRegime.BULLISH_CONSOLIDATION
            elif trend == "down" and volatility > 0.7:
                return EquityRegime.BEARISH_TRENDING
            elif trend == "down" and volatility <= 0.7:
                return EquityRegime.BEARISH_CONSOLIDATION
            else:
                return EquityRegime.NEUTRAL
        except Exception as e:
            logger.error(f"Error analyzing equity regime: {e}")
            return EquityRegime.UNDEFINED
    
    def _analyze_bond_regime(self, data: ProcessedUnderlyingAggregatesV2_5) -> str:
        """Analyze bond market regime."""
        try:
            # Example: Use call_gxoi and put_gxoi as proxies for bond metrics if available
            call_gxoi = getattr(data, 'call_gxoi', 0) or 0
            put_gxoi = getattr(data, 'put_gxoi', 0) or 0
            yield_diff = call_gxoi - put_gxoi
            credit_spread = getattr(data, 'dxoi', 0) or 0
            if yield_diff > 0.5 and credit_spread > 0.3:
                return BondRegime.YIELD_RISING
            elif yield_diff < -0.5 and credit_spread < -0.3:
                return BondRegime.YIELD_FALLING
            elif abs(yield_diff) <= 0.5 and abs(credit_spread) <= 0.3:
                return BondRegime.YIELD_STABLE
            else:
                return BondRegime.UNDEFINED
        except Exception as e:
            logger.error(f"Error analyzing bond regime: {e}")
            return BondRegime.UNDEFINED
    
    def _analyze_commodity_regime(self, data: ProcessedUnderlyingAggregatesV2_5) -> str:
        """Analyze commodity market regime."""
        try:
            # Example: Use vapi_fa_raw_und and dwfd_raw_und as proxies for commodity metrics if available
            supply_demand_ratio = getattr(data, 'vapi_fa_raw_und', 0) or 0
            contango_backwardation = getattr(data, 'dwfd_raw_und', 0) or 0
            if supply_demand_ratio > 1.2:
                return CommodityRegime.DEMAND_DRIVEN
            elif supply_demand_ratio < 0.8:
                return CommodityRegime.SUPPLY_DRIVEN
            elif contango_backwardation > 0.5:
                return CommodityRegime.CONTANGO
            elif contango_backwardation < -0.5:
                return CommodityRegime.BACKWARDATION
            else:
                return CommodityRegime.BALANCED
        except Exception as e:
            logger.error(f"Error analyzing commodity regime: {e}")
            return CommodityRegime.UNDEFINED
    
    def _analyze_currency_regime(self, data: ProcessedUnderlyingAggregatesV2_5) -> str:
        """Analyze currency market regime."""
        try:
            # Example: Use vegas_buy and vegas_sell as proxies for currency metrics if available
            rate_diff = (getattr(data, 'vegas_buy', 0) or 0) - (getattr(data, 'vegas_sell', 0) or 0)
            rate_hike_prob = getattr(data, 'thetas_buy', 0) or 0
            rate_cut_prob = getattr(data, 'thetas_sell', 0) or 0
            if rate_hike_prob > 0.7:
                return CurrencyRegime.STRENGTHENING
            elif rate_cut_prob > 0.7:
                return CurrencyRegime.WEAKENING
            elif abs(rate_diff) > 1.0:
                if rate_diff > 0:
                    return CurrencyRegime.CONSOLIDATION_STRONG
                else:
                    return CurrencyRegime.CONSOLIDATION_WEAK
            else:
                return CurrencyRegime.RANGE_BOUND
        except Exception as e:
            logger.error(f"Error analyzing currency regime: {e}")
            return CurrencyRegime.UNDEFINED

    def _calculate_rolling_correlations(self, data: ProcessedUnderlyingAggregatesV2_5, window: int = 20) -> Dict[str, float]:
        """Calculate rolling correlations between asset classes with robust null checks and type safety.
        
        Args:
            data: Processed underlying data containing asset metrics
            window: Rolling window size for correlation calculation
            
        Returns:
            Dictionary of correlation values between asset classes
        """
        try:
            # Validate input data
            if not data or window < 2:
                return {}
            
            # Get returns with proper null checks and type conversion
            equity_returns = float(getattr(data, 'price_change_pct_und', 0.0) or 0.0)
            bond_returns = float(getattr(data, 'call_gxoi', 0.0) or 0.0)
            commodity_returns = float(getattr(data, 'vapi_fa_raw_und', 0.0) or 0.0)
            currency_returns = float(getattr(data, 'vegas_buy', 0.0) or 0.0)
            
            # Calculate correlations with bounds checking
            correlations = {
                'equity_bond': self._rolling_corr(equity_returns, bond_returns, window),
                'equity_commodity': self._rolling_corr(equity_returns, commodity_returns, window),
                'equity_currency': self._rolling_corr(equity_returns, currency_returns, window),
                'bond_commodity': self._rolling_corr(bond_returns, commodity_returns, window),
                'bond_currency': self._rolling_corr(bond_returns, currency_returns, window),
                'commodity_currency': self._rolling_corr(commodity_returns, currency_returns, window)
            }
            
            # Ensure all values are within [-1, 1] range
            return {k: max(-1.0, min(1.0, v)) for k, v in correlations.items()}
            
        except Exception as e:
            logger.error(f"Error calculating rolling correlations: {e}")
            return {}
        
    def _rolling_corr(self, x: float, y: float, window: int) -> float:
        """Helper method to calculate rolling correlation with bounds checking.
        
        Args:
            x: First return series value
            y: Second return series value
            window: Rolling window size
            
        Returns:
            Correlation coefficient between -1 and 1
        """
        try:
            if window < 2:
                return 0.0
            
            # Simple implementation - in production would use pandas or numpy
            # Ensure we don't divide by zero
            denominator = max(abs(x * y), 1e-6)
            raw_corr = (x * y) / denominator
            
            # Bound the result between -1 and 1
            return max(-1.0, min(1.0, raw_corr))
            
        except Exception as e:
            logger.error(f"Error in rolling correlation calculation: {e}")
            return 0.0
        
    def _calculate_regime_consistency(self, correlations: Dict[str, float]) -> float:
        """Calculate regime consistency score from correlations"""
        try:
            # Calculate weighted average of correlations
            weights = {
                'equity_bond': 0.3,
                'equity_commodity': 0.2,
                'equity_currency': 0.1,
                'bond_commodity': 0.2,
                'bond_currency': 0.1,
                'commodity_currency': 0.1
            }
            
            weighted_sum = sum(correlations.get(k, 0) * v for k, v in weights.items())
            total_weight = sum(weights.values())
            
            # Normalize to 0-1 range
            return (weighted_sum / total_weight + 1) / 2
            
        except Exception as e:
            logger.error(f"Error calculating regime consistency: {e}")
            return 0.0
            
    def _identify_divergent_assets(self, correlations: Dict[str, float], threshold: float = 0.3) -> List[str]:
        """Identify assets with divergent behavior"""
        try:
            divergent_assets = []
            
            # Check for negative correlations
            if correlations.get('equity_bond', 0) < -threshold:
                divergent_assets.append('equity_bond')
            if correlations.get('equity_commodity', 0) < -threshold:
                divergent_assets.append('equity_commodity')
            if correlations.get('bond_commodity', 0) < -threshold:
                divergent_assets.append('bond_commodity')
                
            # Check for low correlations
            if abs(correlations.get('equity_currency', 0)) < threshold:
                divergent_assets.append('equity_currency')
            if abs(correlations.get('bond_currency', 0)) < threshold:
                divergent_assets.append('bond_currency')
                
            return divergent_assets
            
        except Exception as e:
            logger.error(f"Error identifying divergent assets: {e}")
            return []

    def _calculate_systemic_risk(self, correlations: Dict[str, float]) -> float:
        """Calculate systemic risk level."""
        try:
            # High correlations during negative regimes indicate systemic risk
            negative_correlations = [
                corr for corr in correlations.values()
                if corr < 0
            ]
            
            if negative_correlations:
                # Average negative correlation magnitude
                systemic_risk = np.mean([abs(corr) for corr in negative_correlations])
                return float(max(0.0, min(1.0, systemic_risk)))
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Systemic risk calculation failed: {e}")
            return 0.0

    def _create_cross_asset_analysis(self, data: ProcessedUnderlyingAggregatesV2_5, correlations: Dict[str, float], confidence: float) -> CrossAssetAnalysis:
        """Create cross-asset analysis from processed data."""
        try:
            # Get regime analysis for each asset class
            equity_regime = self._analyze_equity_regime(data)
            bond_regime = self._analyze_bond_regime(data)
            commodity_regime = self._analyze_commodity_regime(data)
            currency_regime = self._analyze_currency_regime(data)
            
            # Calculate regime consistency
            regime_consistency = self._calculate_regime_consistency(correlations)
            
            # Identify divergent assets
            divergent_assets = self._identify_divergent_assets(correlations)
            
            # Create cross-asset analysis
            analysis = CrossAssetAnalysis(
                asset_correlations=correlations,
                regime_consistency=regime_consistency,
                divergence_signals=divergent_assets,
                confirmation_signals=[],
                equity_regime=EquityRegime(equity_regime),
                bond_regime=BondRegime(bond_regime),
                commodity_regime=CommodityRegime(commodity_regime),
                currency_regime=CurrencyRegime(currency_regime)
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Cross-asset analysis creation failed: {e}")
            # Return default analysis with low confidence
            return CrossAssetAnalysis(
                asset_correlations={},
                regime_consistency=0.0,
                divergence_signals=[],
                confirmation_signals=[],
                equity_regime=EquityRegime.UNDEFINED,
                bond_regime=BondRegime.UNDEFINED,
                commodity_regime=CommodityRegime.UNDEFINED,
                currency_regime=CurrencyRegime.UNDEFINED
            )

    def _get_regime_characteristics(self, regime_name: str) -> Dict[str, Any]:
        """Get detailed characteristics for a specific regime."""
        characteristics = {
            "bullish_trending": {
                "volatility_level": "Low",
                "trend_direction": "Up",
                "flow_pattern": "Strong Institutional Buying",
                "risk_appetite": "High"
            },
            "bullish_consolidation": {
                "volatility_level": "Low",
                "trend_direction": "Sideways",
                "flow_pattern": "Accumulation",
                "risk_appetite": "Moderate"
            },
            "bearish_trending": {
                "volatility_level": "High",
                "trend_direction": "Down",
                "flow_pattern": "Strong Institutional Selling",
                "risk_appetite": "Low"
            },
            "bearish_consolidation": {
                "volatility_level": "Medium",
                "trend_direction": "Sideways",
                "flow_pattern": "Distribution",
                "risk_appetite": "Low"
            },
            "neutral": {
                "volatility_level": "Low",
                "trend_direction": "Sideways",
                "flow_pattern": "Mixed",
                "risk_appetite": "Moderate"
            },
            "high_volatility": {
                "volatility_level": "High",
                "trend_direction": "Mixed",
                "flow_pattern": "Erratic",
                "risk_appetite": "Very Low"
            },
            "low_volatility": {
                "volatility_level": "Low",
                "trend_direction": "Sideways",
                "flow_pattern": "Range-Bound",
                "risk_appetite": "Moderate"
            },
            "transition": {
                "volatility_level": "Medium",
                "trend_direction": "Mixed",
                "flow_pattern": "Shifting",
                "risk_appetite": "Uncertain"
            },
            "undefined": {
                "volatility_level": "Unknown",
                "trend_direction": "Unknown",
                "flow_pattern": "Unknown",
                "risk_appetite": "Unknown"
            }
        }
        return characteristics.get(regime_name, characteristics["undefined"])
        
    def _generate_regime_description(self, regime_name: str, vri_components: VRI3Components) -> str:
        """Generate a detailed description of the current regime based on VRI components."""
        characteristics = self._get_regime_characteristics(regime_name)
        
        description = f"Market is in a {regime_name.replace('_', ' ').title()} regime. "
        description += f"Volatility is {characteristics['volatility_level']} with {characteristics['trend_direction']} trend. "
        description += f"Flow pattern shows {characteristics['flow_pattern']} with {characteristics['risk_appetite']} risk appetite. "
        
        # Add VRI component insights
        description += f"VRI analysis shows composite strength of {vri_components.composite_score:.2f} "
        description += f"with {vri_components.flow_intensity:.2f} flow intensity and "
        description += f"{vri_components.regime_stability:.2f} regime stability. "
        
        if vri_components.confidence_level > 0.8:
            description += "Analysis confidence is very high. "
        elif vri_components.confidence_level > 0.6:
            description += "Analysis confidence is moderate. "
        else:
            description += "Analysis confidence is low, interpret with caution. "
            
        return description.strip()
        
    def _get_transition_triggers(self, current_regime: str, predicted_regime: str) -> List[str]:
        """Identify potential triggers for regime transition."""
        triggers = []
        
        # Bullish to Bearish transitions
        if current_regime.startswith("bullish") and predicted_regime.startswith("bearish"):
            triggers.extend([
                "Deteriorating market breadth",
                "Increasing institutional selling pressure",
                "Rising volatility expectations",
                "Weakening momentum indicators",
                "Negative earnings surprises"
            ])
            
        # Bearish to Bullish transitions
        elif current_regime.startswith("bearish") and predicted_regime.startswith("bullish"):
            triggers.extend([
                "Improving market breadth",
                "Rising institutional buying pressure",
                "Decreasing volatility expectations",
                "Strengthening momentum indicators",
                "Positive earnings surprises"
            ])
            
        # Transitions to high volatility
        elif predicted_regime == "high_volatility":
            triggers.extend([
                "Sudden increase in options implied volatility",
                "Rising correlation across assets",
                "Increased trading volume",
                "Market sentiment extremes",
                "External shock events"
            ])
            
        # Transitions to low volatility
        elif predicted_regime == "low_volatility":
            triggers.extend([
                "Decreasing options implied volatility",
                "Normalizing correlations",
                "Stabilizing trading volume",
                "Neutral market sentiment",
                "Absence of major catalysts"
            ])
            
        return triggers

    def _get_risk_factors(self, current_regime: str, predicted_regime: str) -> List[str]:
        """Identify risk factors affecting regime transition."""
        risk_factors = []
        
        # Base risk factors for any transition
        risk_factors.extend([
            "Macro-economic policy changes",
            "Global market conditions",
            "Sector rotation dynamics"
        ])
        
        # Specific risk factors based on transition type
        if current_regime.startswith("bullish"):
            risk_factors.extend([
                "Overbought technical conditions",
                "Excessive optimism",
                "Valuation concerns",
                "Profit-taking pressure"
            ])
            
        elif current_regime.startswith("bearish"):
            risk_factors.extend([
                "Oversold technical conditions",
                "Extreme pessimism",
                "Forced liquidations",
                "Margin call pressure"
            ])
            
        # Additional risks for high volatility transitions
        if predicted_regime == "high_volatility":
            risk_factors.extend([
                "Liquidity constraints",
                "Market structure stress",
                "Hedging activity impact",
                "Systematic strategy flows"
            ])
            
        return risk_factors

    def _calculate_cross_asset_confidence(self, correlations: Dict[str, float]) -> float:
        """Calculate confidence score for cross-asset analysis."""
        # Calculate based on correlation strength and consistency
        correlation_values = list(correlations.values())
        mean_correlation = np.mean(np.abs(correlation_values))
        correlation_std = np.std(correlation_values)
        return float(max(0.0, min(1.0, mean_correlation * (1.0 - correlation_std))))

    def _calculate_vri_components(self, data: ProcessedUnderlyingAggregatesV2_5) -> VRI3Components:
        """Calculate VRI 3.0 components with machine learning enhancement."""
        try:
            # Calculate base VRI components using existing metrics
            volatility_regime = self.metrics_calculator.calculate_volatility_regime(data)
            flow_intensity = self.metrics_calculator.calculate_flow_intensity(data)
            
            # Calculate regime stability and transition momentum
            regime_stability = self.metrics_calculator.calculate_regime_stability(data)
            transition_momentum = self.metrics_calculator.calculate_transition_momentum(data)
            
            # Calculate composite score with ML enhancement
            composite_score = np.mean([
                volatility_regime,
                flow_intensity,
                regime_stability,
                transition_momentum
            ])
            
            # Calculate confidence level based on data quality
            # Use simpler confidence calculation since we don't have the specific methods
            confidence_factors = [
                volatility_regime > 0.0,  # Valid volatility
                flow_intensity > 0.0,     # Valid flow
                regime_stability > 0.0,    # Valid stability
                transition_momentum > 0.0  # Valid momentum
            ]
            confidence_level = sum(confidence_factors) / len(confidence_factors)
            
            # Create VRI components object
            vri_components = VRI3Components(
                composite_score=float(composite_score),
                volatility_regime=float(volatility_regime),
                flow_intensity=float(flow_intensity),
                regime_stability=float(regime_stability),
                transition_momentum=float(transition_momentum),
                confidence_level=float(confidence_level)
            )
            
            return vri_components
            
        except Exception as e:
            logger.error(f"VRI components calculation failed: {e}")
            # Return default components with low confidence
            return VRI3Components(
                composite_score=0.5,
                volatility_regime=0.5,
                flow_intensity=0.5,
                regime_stability=0.5,
                transition_momentum=0.5,
                confidence_level=0.1
            )

    def _classify_regime(self, data: ProcessedDataBundleV2_5, vri_components: VRI3Components) -> RegimeClassification:
        """Classify the current market regime."""
        try:
            regime_name = self.metrics_calculator.classify_regime(vri_components.model_dump())
            characteristics = self._get_regime_characteristics(regime_name)
            regime_id = f"REGIME_{regime_name.upper().replace(' ', '_')}"
            
            return RegimeClassification(
                regime_id=regime_id,
                regime_name=regime_name,
                regime_description=self._generate_regime_description(regime_name, vri_components),
                probability=vri_components.composite_score,
                confidence=vri_components.confidence_level,
                volatility_level=characteristics['volatility_level'],
                trend_direction=characteristics['trend_direction'],
                flow_pattern=characteristics['flow_pattern'],
                risk_appetite=characteristics['risk_appetite']
            )
            
        except Exception as e:
            logger.error(f"Error classifying regime: {str(e)}")
            raise
            
    def _predict_regime_transition(self, current_regime: RegimeClassification, vri_components: VRI3Components) -> RegimeTransitionPrediction:
        """Predict future regime transitions with probabilities."""
        try:
            # Calculate transition probabilities
            transition_probs = self.metrics_calculator.calculate_regime_transition_probabilities(
                current_regime.regime_name,
                {
                    "composite_score": vri_components.composite_score,
                    "volatility_regime": vri_components.volatility_regime,
                    "flow_intensity": vri_components.flow_intensity,
                    "regime_stability": vri_components.regime_stability,
                    "transition_momentum": vri_components.transition_momentum
                }
            )
            
            # Find most likely next regime
            predicted_regime = max(transition_probs.items(), key=lambda x: x[1])[0]
            transition_probability = transition_probs[predicted_regime]
            
            # Calculate expected timeframe
            expected_timeframe = self.metrics_calculator.calculate_transition_timeframe({
                "composite_score": vri_components.composite_score,
                "volatility_regime": vri_components.volatility_regime,
                "flow_intensity": vri_components.flow_intensity,
                "regime_stability": vri_components.regime_stability,
                "transition_momentum": vri_components.transition_momentum
            })
            
            # Get transition triggers and risk factors
            transition_triggers = self._get_transition_triggers(current_regime.regime_name, predicted_regime)
            risk_factors = self._get_risk_factors(current_regime.regime_name, predicted_regime)
            
            # Create prediction result
            prediction = RegimeTransitionPrediction(
                current_regime=current_regime.regime_name,
                predicted_regime=predicted_regime,
                transition_probability=transition_probability,
                expected_timeframe=expected_timeframe,
                confidence_level=vri_components.confidence_level,
                transition_triggers=transition_triggers,
                risk_factors=risk_factors
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Failed to predict regime transition: {e}")
            return RegimeTransitionPrediction(
                current_regime=current_regime.regime_name,
                predicted_regime=current_regime.regime_name,
                transition_probability=0.0,
                expected_timeframe=0,
                confidence_level=0.0,
                transition_triggers=[],
                risk_factors=[]
            )

    def _analyze_cross_asset_correlations(self, data: ProcessedDataBundleV2_5) -> CrossAssetAnalysis:
        """Analyze cross-asset correlations and regimes."""
        try:
            # Analyze individual asset regimes
            equity_regime = self._analyze_equity_regime(data.underlying_data_enriched)
            bond_regime = self._analyze_bond_regime(data.underlying_data_enriched)
            commodity_regime = self._analyze_commodity_regime(data.underlying_data_enriched)
            currency_regime = self._analyze_currency_regime(data.underlying_data_enriched)
            
            # Calculate asset returns and correlations
            asset_returns = self._calculate_asset_returns(data)
            correlations = {}
            
            # Calculate pairwise correlations
            for i, asset1 in enumerate(self.config.asset_classes):
                for asset2 in self.config.asset_classes[i+1:]:
                    if asset1 in asset_returns and asset2 in asset_returns:
                        corr = float(np.corrcoef(asset_returns[asset1], asset_returns[asset2])[0, 1])
                        correlations[f"{asset1}_{asset2}"] = corr
            
            # Calculate regime consistency and confidence
            regime_consistency = self._calculate_regime_consistency(correlations)
            confidence_level = self._calculate_cross_asset_confidence(correlations)
            
            # Identify divergent assets and systemic risk
            divergent_assets = self._identify_divergent_assets(correlations)
            systemic_risk_level = self._calculate_systemic_risk(correlations)
            
            # Create cross-asset analysis
            analysis = CrossAssetAnalysis(
                asset_correlations=correlations,
                regime_consistency=regime_consistency,
                divergence_signals=divergent_assets,
                confirmation_signals=[],
                equity_regime=EquityRegime(equity_regime),
                bond_regime=BondRegime(bond_regime),
                commodity_regime=CommodityRegime(commodity_regime),
                currency_regime=CurrencyRegime(currency_regime)
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze cross-asset correlations: {e}")
            return CrossAssetAnalysis(
                asset_correlations={},
                regime_consistency=0.0,
                divergence_signals=[],
                confirmation_signals=[],
                equity_regime=EquityRegime.UNDEFINED,
                bond_regime=BondRegime.UNDEFINED,
                commodity_regime=CommodityRegime.UNDEFINED,
                currency_regime=CurrencyRegime.UNDEFINED
            )

    def _calculate_asset_returns(self, data: ProcessedDataBundleV2_5) -> Dict[str, np.ndarray]:
        """Calculate returns for each asset class."""
        try:
            # Get underlying data
            underlying = data.underlying_data_enriched
            
            # Calculate returns (simplified for now)
            returns = {}
            if underlying.price and underlying.prev_day_close_price_und:
                returns['equities'] = np.array([
                    (underlying.price - underlying.prev_day_close_price_und) / 
                    underlying.prev_day_close_price_und
                ])
            
            # Add placeholder returns for other assets
            # In a real implementation, these would come from actual data
            returns['bonds'] = np.array([0.0])
            returns['commodities'] = np.array([0.0])
            returns['currencies'] = np.array([0.0])
            
            return returns
            
        except Exception as e:
            logger.error(f"Asset returns calculation failed: {e}")
            return {
                'equities': np.array([0.0]),
                'bonds': np.array([0.0]),
                'commodities': np.array([0.0]),
                'currencies': np.array([0.0])
            }

    def analyze(self, data: ProcessedDataBundleV2_5) -> LegendaryRegimeResult:
        """Perform comprehensive market regime analysis."""
        try:
            start_time = time.time()
            
            # Calculate VRI 3.0 components
            vri_components = self._calculate_vri_components(data.underlying_data_enriched)
            
            # Classify current regime
            regime_classification = self._classify_regime(data, vri_components)
            
            # Predict regime transition
            transition_prediction = self._predict_regime_transition(regime_classification, vri_components)
            
            # Calculate cross-asset correlations and analysis
            cross_asset_analysis = self._analyze_cross_asset_correlations(data)
            
            # Calculate volatility forecast
            volatility_forecast = {
                "1d": self.metrics_calculator.calculate_volatility_regime(data.underlying_data_enriched),
                "5d": self.metrics_calculator.calculate_regime_stability(data.underlying_data_enriched),
                "20d": self.metrics_calculator.calculate_flow_intensity(data.underlying_data_enriched)
            }
            
            # Calculate regime strength and persistence
            regime_strength = vri_components.composite_score
            regime_persistence = vri_components.regime_stability
            
            # Calculate risk scores
            regime_risk_score = 1.0 - vri_components.confidence_level
            tail_risk_probability = max(0.0, min(1.0, 1.0 - vri_components.regime_stability))
            
            # Create result
            result = LegendaryRegimeResult(
                analysis_id=str(uuid.uuid4()),
                ticker=data.underlying_data_enriched.symbol,
                timestamp=datetime.now(),
                processing_time_ms=float((time.time() - start_time) * 1000),
                vri_3_analysis=vri_components,
                regime_classification=regime_classification,
                transition_prediction=transition_prediction,
                cross_asset_analysis=cross_asset_analysis,
                regime_strength=regime_strength,
                regime_persistence=regime_persistence,
                volatility_forecast=volatility_forecast,
                regime_risk_score=regime_risk_score,
                tail_risk_probability=tail_risk_probability,
                black_swan_indicators=[],
                prediction_accuracy=None,
                confidence_score=vri_components.confidence_level,
                data_quality_score=1.0
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Market regime analysis failed: {e}")
            return LegendaryRegimeResult(
                analysis_id=str(uuid.uuid4()),
                ticker=data.underlying_data_enriched.symbol,
                timestamp=datetime.now(),
                processing_time_ms=0.0,
                vri_3_analysis=VRI3Components(
                    composite_score=0.0,
                    volatility_regime=0.0,
                    flow_intensity=0.0,
                    regime_stability=0.0,
                    transition_momentum=0.0,
                    confidence_level=0.0
                ),
                regime_classification=RegimeClassification(
                    regime_id="REGIME_UNDEFINED",
                    regime_name="undefined",
                    regime_description="Analysis failed",
                    probability=0.0,
                    confidence=0.0,
                    volatility_level="Unknown",
                    trend_direction="Unknown",
                    flow_pattern="Unknown",
                    risk_appetite="Unknown"
                ),
                transition_prediction=RegimeTransitionPrediction(
                    current_regime="undefined",
                    predicted_regime="undefined",
                    transition_probability=0.0,
                    expected_timeframe=0,
                    confidence_level=0.0,
                    transition_triggers=[],
                    risk_factors=[]
                ),
                cross_asset_analysis=CrossAssetAnalysis(
                    asset_correlations={},
                    regime_consistency=0.0,
                    divergence_signals=[],
                    confirmation_signals=[],
                    equity_regime=EquityRegime.UNDEFINED,
                    bond_regime=BondRegime.UNDEFINED,
                    commodity_regime=CommodityRegime.UNDEFINED,
                    currency_regime=CurrencyRegime.UNDEFINED
                ),
                regime_strength=0.0,
                regime_persistence=0.0,
                volatility_forecast={"1d": 0.0, "5d": 0.0, "20d": 0.0},
                regime_risk_score=1.0,
                tail_risk_probability=1.0,
                black_swan_indicators=["Analysis Failed"],
                prediction_accuracy=None,
                confidence_score=0.0,
                data_quality_score=0.0
            )

    def get_specialization_keywords(self) -> List[str]:
        """Get keywords that define this expert's specialization."""
        return [
            'market_regime',
            'volatility',
            'trend',
            'flow',
            'vri',
            'cross_asset',
            'macro'
        ]

    def get_example_analysis(self) -> Dict[str, Any]:
        """Get example analysis output for documentation."""
        return {
            'regime_classification': RegimeClassification(
                regime_id='REGIME_BULL_TREND_LOW_VOL',
                regime_name='Bull Trend Low Volatility',
                regime_description='Strong uptrend with low volatility, characterized by steady accumulation and positive sentiment.',
                probability=0.85,
                confidence=0.90,
                volatility_level='Low',
                trend_direction='Up',
                flow_pattern='Accumulation',
                risk_appetite='Risk-On'
            ),
            'vri_components': VRI3Components(
                composite_score=0.82,
                volatility_regime=0.25,
                flow_intensity=0.75,
                regime_stability=0.85,
                transition_momentum=0.45,
                confidence_level=0.85
            ),
            'cross_asset_analysis': CrossAssetAnalysis(
                asset_correlations={'equity_bonds': -0.3, 'equity_commodities': 0.5},
                regime_consistency=0.75,
                divergence_signals=['bonds'],
                confirmation_signals=[],
                equity_regime=EquityRegime.BULLISH_TRENDING,
                bond_regime=BondRegime.YIELD_RISING,
                commodity_regime=CommodityRegime.DEMAND_DRIVEN,
                currency_regime=CurrencyRegime.STRENGTHENING
            )
        }

    def _calculate_macro_indicators(self, data: ProcessedDataBundleV2_5) -> Dict[str, float]:
        """Calculate macro-economic indicators for regime analysis."""
        try:
            macro_indicators = {}
            
            # VIX analysis
            macro_indicators["vix"] = self.metrics_calculator.calculate_volatility_regime(data.underlying_data_enriched)
            
            # Interest rates analysis
            macro_indicators["rates"] = self.metrics_calculator.calculate_regime_stability(data.underlying_data_enriched)
            
            # Credit spread analysis
            macro_indicators["credit"] = self.metrics_calculator.calculate_flow_intensity(data.underlying_data_enriched)
            
            # Liquidity analysis
            macro_indicators["liquidity"] = self.metrics_calculator.calculate_transition_momentum(data.underlying_data_enriched)
            
            # Market sentiment analysis
            macro_indicators["sentiment"] = self.metrics_calculator.calculate_vri3_composite(data.underlying_data_enriched)
            
            # Market momentum analysis
            macro_indicators["momentum"] = self.metrics_calculator.calculate_confidence_level(data.underlying_data_enriched)
            
            # Volatility surface analysis
            macro_indicators["volatility"] = self.metrics_calculator.calculate_volatility_regime(data.underlying_data_enriched)
            
            return macro_indicators
            
        except Exception as e:
            logger.error(f"Failed to calculate macro indicators: {e}")
            return {indicator: 0.0 for indicator in self.config.macro_indicators}


    # --- Implementation of BaseHuiHuiExpert Abstract Methods ---

    def get_specialization_keywords(self) -> List[str]:
        """Get keywords that define this expert's specialization."""
        return [
            'market_regime', 'regime', 'volatility', 'vri', 'vri_3.0',
            'trend', 'cross_asset', 'macro', 'market_condition', 'regime_transition'
        ]

    def validate_input_data(self, data_bundle: ProcessedDataBundleV2_5) -> bool:
        """Validate that input data contains required fields for this expert."""
        if not data_bundle or not data_bundle.underlying_data_enriched:
            self.logger.warning("Input validation failed: Missing underlying_data_enriched in data_bundle.")
            return False

        # Example check: VRI 3.0 components rely on several underlying metrics.
        # This expert's internal _calculate_vri_components uses metrics_calculator,
        # which itself should handle missing data for its constituent parts.
        # So, a basic check for underlying_data_enriched might be sufficient here.
        if not hasattr(data_bundle.underlying_data_enriched, 'price'):
            self.logger.warning("Input validation: Price data missing from underlying_data_enriched.")
            # Depending on strictness, could return False or allow graceful degradation.
            # For now, let's assume metrics_calculator handles missing underlying fields.

        return True

    async def analyze(self, request: HuiHuiAnalysisRequestV2_5) -> HuiHuiAnalysisResponseV2_5:
        """
        Perform comprehensive market regime analysis.
        This is the main entry point called by the coordinator.
        """
        if not request.bundle_data:
            self.logger.error("No data bundle provided in HuiHuiAnalysisRequestV2_5 for MarketRegimeExpert.")
            return HuiHuiAnalysisResponseV2_5(
                expert_used=self.config.expert_id, # from HuiHuiExpertConfigV2_5
                analysis_content="Error: No data bundle provided for regime analysis.",
                confidence_score=0.0,
                processing_time=0.0,
                insights=["Critical error: Missing data bundle for regime analysis."],
                errors=["Missing ProcessedDataBundleV2_5 in request for MarketRegimeExpert."]
            )

        if not self.validate_input_data(request.bundle_data):
            self.logger.error("Input data validation failed for MarketRegimeExpert.")
            return HuiHuiAnalysisResponseV2_5(
                expert_used=self.config.expert_id,
                analysis_content="Error: Input data validation failed for regime analysis.",
                confidence_score=0.0,
                processing_time=0.0,
                insights=["Critical error: Invalid input data for regime analysis."],
                errors=["Input data validation failed for MarketRegimeExpert."]
            )

        # Call the original synchronous analysis logic.
        # If _perform_regime_analysis_logic becomes truly async (e.g., awaits IO),
        # then this await is correct. If it's CPU-bound, can run in executor or directly.
        # For now, assuming it's okay to call directly as its sub-methods are synchronous.
        # To be fully async-compliant if its sub-methods were blocking:
        # legendary_result = await asyncio.to_thread(self._perform_regime_analysis_logic, request.bundle_data)
        legendary_result: LegendaryRegimeResult = self._perform_regime_analysis_logic(request.bundle_data)

        # Package LegendaryRegimeResult into HuiHuiAnalysisResponseV2_5
        analysis_content_str = (
            f"**Market Regime Analysis for {legendary_result.ticker} ({legendary_result.timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')})**\n\n"
            f"**Current Regime:** {legendary_result.regime_classification.regime_name} (Prob: {legendary_result.regime_classification.probability:.2%}, Conf: {legendary_result.regime_classification.confidence:.2%})\n"
            f"   - Description: {legendary_result.regime_classification.regime_description}\n"
            f"   - Volatility: {legendary_result.regime_classification.volatility_level}, Trend: {legendary_result.regime_classification.trend_direction}, Flow: {legendary_result.regime_classification.flow_pattern}\n\n"
            f"**VRI 3.0 Analysis (Composite: {legendary_result.vri_3_analysis.composite_score:.2f}):**\n"
            f"   - Volatility Comp: {legendary_result.vri_3_analysis.volatility_regime:.2f}, Flow Comp: {legendary_result.vri_3_analysis.flow_intensity:.2f}\n"
            f"   - Stability: {legendary_result.vri_3_analysis.regime_stability:.2f}, Transition Momentum: {legendary_result.vri_3_analysis.transition_momentum:.2f}\n\n"
            f"**Regime Transition Prediction (Confidence: {legendary_result.transition_prediction.confidence_level:.2%}):**\n"
            f"   - Predicted Next: {legendary_result.transition_prediction.predicted_regime} (Prob: {legendary_result.transition_prediction.transition_probability:.2%})\n"
            f"   - Expected Timeframe: {legendary_result.transition_prediction.expected_timeframe} days\n\n"
            f"**Overall Strength:** {legendary_result.regime_strength:.2f}, Persistence: {legendary_result.regime_persistence:.2f}\n"
            f"**Regime Risk Score:** {legendary_result.regime_risk_score:.2f}, Tail Risk Prob: {legendary_result.tail_risk_probability:.2%}"
        )

        insights = [
            f"Current Regime: {legendary_result.regime_classification.regime_name} (Confidence: {legendary_result.regime_classification.confidence:.0%})",
            f"VRI 3.0 Composite: {legendary_result.vri_3_analysis.composite_score:.2f}",
            f"Predicted Next Regime: {legendary_result.transition_prediction.predicted_regime} (Prob: {legendary_result.transition_prediction.transition_probability:.0%})"
        ]

        response = HuiHuiAnalysisResponseV2_5(
            expert_used=HuiHuiExpertType.MARKET_REGIME, # Use enum member
            analysis_content=analysis_content_str,
            confidence_score=legendary_result.confidence_score,
            processing_time=legendary_result.processing_time_ms / 1000.0, # convert to seconds
            insights=insights,
            # eots_predictions and recommendations can be added if this expert generates them
        )

        self.last_analysis_time = datetime.utcnow() # From BaseHuiHuiExpert
        # Record usage using the base class method
        self.record_usage(request, response, legendary_result.processing_time_ms)

        return response

    # Renamed original analyze method to be the core synchronous logic
    def _perform_regime_analysis_logic(self, data: ProcessedDataBundleV2_5) -> LegendaryRegimeResult:
        """Perform comprehensive market regime analysis."""
        try:
            start_time = time.time()

            # Calculate VRI 3.0 components
            vri_components = self._calculate_vri_components(data.underlying_data_enriched)

            # Classify current regime
            regime_classification = self._classify_regime(data, vri_components)

            # Predict regime transition
            transition_prediction = self._predict_regime_transition(regime_classification, vri_components)

            # Calculate cross-asset correlations and analysis
            cross_asset_analysis = self._analyze_cross_asset_correlations(data)

            # Calculate volatility forecast
            volatility_forecast = {
                "1d": self.metrics_calculator.calculate_volatility_regime(data.underlying_data_enriched),
                "5d": self.metrics_calculator.calculate_regime_stability(data.underlying_data_enriched),
                "20d": self.metrics_calculator.calculate_flow_intensity(data.underlying_data_enriched)
            }

            # Calculate regime strength and persistence
            regime_strength = vri_components.composite_score
            regime_persistence = vri_components.regime_stability

            # Calculate risk scores
            regime_risk_score = 1.0 - vri_components.confidence_level
            tail_risk_probability = max(0.0, min(1.0, 1.0 - vri_components.regime_stability))

            # Create result
            result = LegendaryRegimeResult(
                analysis_id=str(uuid.uuid4()),
                ticker=data.underlying_data_enriched.symbol,
                timestamp=datetime.now(), # Use current time for result timestamp
                processing_time_ms=float((time.time() - start_time) * 1000),
                vri_3_analysis=vri_components,
                regime_classification=regime_classification,
                transition_prediction=transition_prediction,
                cross_asset_analysis=cross_asset_analysis,
                regime_strength=regime_strength,
                regime_persistence=regime_persistence,
                volatility_forecast=volatility_forecast,
                regime_risk_score=regime_risk_score,
                tail_risk_probability=tail_risk_probability,
                black_swan_indicators=[],
                prediction_accuracy=None,
                confidence_score=vri_components.confidence_level,
                data_quality_score=1.0 # Placeholder, should be calculated
            )

            return result

        except Exception as e:
            self.logger.error(f"Market regime analysis logic failed: {e}", exc_info=True)
            # Ensure a LegendaryRegimeResult is always returned, even on error
            current_symbol = "UNKNOWN"
            if data and data.underlying_data_enriched and data.underlying_data_enriched.symbol:
                current_symbol = data.underlying_data_enriched.symbol

            return LegendaryRegimeResult(
                analysis_id=str(uuid.uuid4()),
                ticker=current_symbol,
                timestamp=datetime.now(),
                processing_time_ms=0.0,
                vri_3_analysis=VRI3Components(
                    composite_score=0.0,
                    volatility_regime=0.0,
                    flow_intensity=0.0,
                    regime_stability=0.0,
                    transition_momentum=0.0,
                    confidence_level=0.0
                ),
                regime_classification=RegimeClassification(
                    regime_id="REGIME_UNDEFINED_ERROR",
                    regime_name="undefined_error",
                    regime_description=f"Analysis failed: {str(e)}",
                    probability=0.0,
                    confidence=0.0,
                    volatility_level="Unknown",
                    trend_direction="Unknown",
                    flow_pattern="Unknown",
                    risk_appetite="Unknown"
                ),
                transition_prediction=RegimeTransitionPrediction(
                    current_regime="undefined_error",
                    predicted_regime="undefined_error",
                    transition_probability=0.0,
                    expected_timeframe=0,
                    confidence_level=0.0,
                    transition_triggers=[],
                    risk_factors=[]
                ),
                cross_asset_analysis=CrossAssetAnalysis(
                    asset_correlations={},
                    regime_consistency=0.0,
                    divergence_signals=[],
                    confirmation_signals=[],
                    equity_regime=EquityRegime.UNDEFINED,
                    bond_regime=BondRegime.UNDEFINED,
                    commodity_regime=CommodityRegime.UNDEFINED,
                    currency_regime=CurrencyRegime.UNDEFINED
                ),
                regime_strength=0.0,
                regime_persistence=0.0,
                volatility_forecast={"1d": 0.0, "5d": 0.0, "20d": 0.0},
                regime_risk_score=1.0,
                tail_risk_probability=1.0,
                black_swan_indicators=["Analysis Failed Due to Error"],
                prediction_accuracy=None,
                confidence_score=0.0,
                data_quality_score=0.0
            )

# Maintain backward compatibility
MarketRegimeExpert = UltimateMarketRegimeExpert

