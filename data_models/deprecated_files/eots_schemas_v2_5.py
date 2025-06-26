"""DEPRECATED: This module is kept for backward compatibility only.

All models have been moved to a more organized package structure:
- data_models.raw: For raw API data models
- data_models.processed: For processed and enriched data models
- data_models.analytics: For analytics and metrics models

Please update your imports to use the new module structure.
"""
import warnings
from typing import List, Dict, Any, Optional, Union, TypeVar
from datetime import datetime

# Re-export from the new package structure
from ..base import BaseModel, Field, field_validator, model_validator, ConfigDict, FieldValidationInfo
from ..base import T, PandasDataFrame, PathLike, Timestamp, validate_timestamp

# Re-export models from the new structure
from ..raw import (
    RawOptionsContractV2_5,
    ConsolidatedUnderlyingDataV2_5,
    UnprocessedDataBundleV2_5
)

from ..processed import (
    ProcessedUnderlyingAggregatesV2_5,
    ProcessedContractMetricsV2_5,
    ProcessedStrikeLevelMetricsV2_5
)

from ..analytics import (
    MarketRegimeState,
    DynamicThresholdsV2_5
)

# Issue deprecation warning
warnings.warn(
    "eots_schemas_v2_5.py is deprecated and will be removed in a future version. "
    "Please update your imports to use the new module structure.",
    DeprecationWarning,
    stacklevel=2
)

# Market Regime States for HuiHui Integration
class MarketRegimeState(str, Enum):
    """Market regime states for the expert system.
    
    Attributes:
        BULLISH_TREND: Sustained upward price movement
        BEARISH_TREND: Sustained downward price movement
        SIDEWAYS: No clear trend, price oscillating in a range
        VOLATILITY_EXPANSION: Increasing price volatility
        VOLATILITY_CONTRACTION: Decreasing price volatility
        BULLISH_REVERSAL: Potential reversal from downtrend to uptrend
        BEARISH_REVERSAL: Potential reversal from uptrend to downtrend
        DISTRIBUTION: Smart money distributing positions
        ACCUMULATION: Smart money accumulating positions
        CAPITULATION: Panic selling
        EUPHORIA: Extreme bullish sentiment
        PANIC: Extreme bearish sentiment
        CONSOLIDATION: Price moving in a tight range
        BREAKOUT: Price breaking out of a range
        BREAKDOWN: Price breaking down from a range
        CHOPPY: Erratic price action
        TRENDING: Clear directional movement
        RANGE_BOUND: Price contained within support/resistance
        UNDEFINED: Default/unknown state
    """
    BULLISH_TREND = "bullish_trend"
    BEARISH_TREND = "bearish_trend"
    SIDEWAYS = "sideways"
    VOLATILITY_EXPANSION = "volatility_expansion"
    VOLATILITY_CONTRACTION = "volatility_contraction"
    BULLISH_REVERSAL = "bullish_reversal"
    BEARISH_REVERSAL = "bearish_reversal"
    DISTRIBUTION = "distribution"
    ACCUMULATION = "accumulation"
    CAPITULATION = "capitulation"
    EUPHORIA = "euphoria"
    PANIC = "panic"
    CONSOLIDATION = "consolidation"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    CHOPPY = "choppy"
    TRENDING = "trending"
    RANGE_BOUND = "range_bound"
    UNDEFINED = "undefined"

# Placeholder for pandas DataFrame if we decide to type hint it (not true Pydantic validation for content)
PandasDataFrame = Any # Or pd.DataFrame, but then Pydantic won't validate its contents


# --- Canonical Parameter Lists from ConvexValue ---
# For reference and ensuring Raw models are comprehensive.
UNDERLYING_REQUIRED_PARAMS_CV = [
    "price", "volatility", "day_volume", "call_gxoi", "put_gxoi",
    "gammas_call_buy", "gammas_call_sell", "gammas_put_buy", "gammas_put_sell",
    "deltas_call_buy", "deltas_call_sell", "deltas_put_buy", "deltas_put_sell",
    "vegas_call_buy", "vegas_call_sell", "vegas_put_buy", "vegas_put_sell",
    "thetas_call_buy", "thetas_call_sell", "thetas_put_buy", "thetas_put_sell",
    "call_vxoi", "put_vxoi", "value_bs", "volm_bs", "deltas_buy", "deltas_sell",
    "vegas_buy", "vegas_sell", "thetas_buy", "thetas_sell", "volm_call_buy",
    "volm_put_buy", "volm_call_sell", "volm_put_sell", "value_call_buy",
    "value_put_buy", "value_call_sell", "value_put_sell", "vflowratio",
    "dxoi", "gxoi", "vxoi", "txoi", "call_dxoi", "put_dxoi"
]

OPTIONS_CHAIN_REQUIRED_PARAMS_CV = [
    "price", "volatility", "multiplier", "oi", "delta", "gamma", "theta", "vega",
    "vanna", "vomma", "charm", "dxoi", "gxoi", "vxoi", "txoi", "vannaxoi",
    "vommaxoi", "charmxoi", "dxvolm", "gxvolm", "vxvolm", "txvolm", "vannaxvolm",
    "vommaxvolm", "charmxvolm", "value_bs", "volm_bs", "deltas_buy", "deltas_sell",
    "gammas_buy", "gammas_sell", "vegas_buy", "vegas_sell", "thetas_buy", "thetas_sell",
    "valuebs_5m", "volmbs_5m", "valuebs_15m", "volmbs_15m",
    "valuebs_30m", "volmbs_30m", "valuebs_60m", "volmbs_60m",
    "volm", "volm_buy", "volm_sell", "value_buy", "value_sell"
]
# End Canonical Parameter Lists


class RawOptionsContractV2_5(BaseModel):
    """Represents a raw options contract with all available market data.
    
    This model serves as the foundation for options data processing and validation.
    It includes fields for greeks, open interest, volume, and other market metrics.
    """
    
    contract_symbol: str = Field(..., description="Unique identifier for the options contract")
    strike: float = Field(..., gt=0, description="Strike price of the option")
    opt_kind: Literal["call", "put"] = Field(..., description="Option type (call/put)")
    dte_calc: Optional[float] = Field(
        None, 
        ge=0, 
        description="Days to expiration (calculated if not provided)"
    )

    # Fields corresponding to OPTIONS_CHAIN_REQUIRED_PARAMS_CV
    # Existing fields (comments denote mapping if name differs)
    open_interest: Optional[float] = Field(None, description="Open interest for the contract (maps to CV 'oi')")
    iv: Optional[float] = Field(None, description="Implied Volatility for the contract (maps to CV 'volatility')")
    raw_price: Optional[float] = Field(None, description="Raw price of the option contract from CV (maps to CV 'price')") # Explicit for CV 'price'
    delta_contract: Optional[float] = Field(None, description="Delta per contract (maps to CV 'delta')")
    gamma_contract: Optional[float] = Field(None, description="Gamma per contract (maps to CV 'gamma')")
    theta_contract: Optional[float] = Field(None, description="Theta per contract (maps to CV 'theta')")
    vega_contract: Optional[float] = Field(None, description="Vega per contract (maps to CV 'vega')")
    rho_contract: Optional[float] = Field(None, description="Rho per contract") # Rho is standard, not explicitly in CV list but good to have
    vanna_contract: Optional[float] = Field(None, description="Vanna per contract (maps to CV 'vanna')")
    vomma_contract: Optional[float] = Field(None, description="Vomma per contract (maps to CV 'vomma')")
    charm_contract: Optional[float] = Field(None, description="Charm per contract (maps to CV 'charm')")

    # Greeks OI (Open Interest based Greeks, if provided directly)
    dxoi: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (dxoi)")
    gxoi: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (gxoi)")
    vxoi: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (vxoi)")
    txoi: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (txoi)")
    vannaxoi: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (vannaxoi)")
    vommaxoi: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (vommaxoi)")
    charmxoi: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (charmxoi)")

    # Greek-Volume Proxies (some may be redundant if direct signed Greek flows are available)
    dxvolm: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (dxvolm)")
    gxvolm: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (gxvolm)")
    vxvolm: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (vxvolm)")
    txvolm: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (txvolm)")
    vannaxvolm: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (vannaxvolm)")
    vommaxvolm: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (vommaxvolm)")
    charmxvolm: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (charmxvolm)")

    # Transaction Data
    value_bs: Optional[float] = Field(None, description="Day Sum of Buy Value minus Sell Value Traded (maps to CV 'value_bs')")
    volm_bs: Optional[float] = Field(None, description="Volume of Buys minus Sells (maps to CV 'volm_bs')")
    volm: Optional[float] = Field(None, description="Total volume for the contract (maps to CV 'volm')")

    # Rolling Flows
    valuebs_5m: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (valuebs_5m)")
    volmbs_5m: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (volmbs_5m)")
    valuebs_15m: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (valuebs_15m)")
    volmbs_15m: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (volmbs_15m)")
    valuebs_30m: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (valuebs_30m)")
    volmbs_30m: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (volmbs_30m)")
    valuebs_60m: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (valuebs_60m)")
    volmbs_60m: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (volmbs_60m)")

    # Bid/Ask for liquidity calculations and advanced options metrics
    bid: Optional[float] = Field(None, description="Bid price of the option (ConvexValue 'bid')")
    ask: Optional[float] = Field(None, description="Ask price of the option (ConvexValue 'ask')")
    bid_size: Optional[float] = Field(None, description="Bid size/volume of the option (ConvexValue 'bid_size')")
    ask_size: Optional[float] = Field(None, description="Ask size/volume of the option (ConvexValue 'ask_size')")
    bid_price: Optional[float] = Field(None, description="Legacy bid price field")
    ask_price: Optional[float] = Field(None, description="Legacy ask price field")
    mid_price: Optional[float] = Field(None, description="Midpoint price of the option")
    theo: Optional[float] = Field(None, description="Theoretical fair value of the option (ConvexValue 'theo')")
    spread: Optional[float] = Field(None, description="Bid-ask spread (ConvexValue 'spread')")

    # New fields from OPTIONS_CHAIN_REQUIRED_PARAMS_CV
    multiplier: Optional[float] = Field(None, description="Option contract multiplier (e.g., 100) (maps to CV 'multiplier')")
    deltas_buy: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (deltas_buy)")
    deltas_sell: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (deltas_sell)")
    gammas_buy: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (gammas_buy)")
    gammas_sell: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (gammas_sell)")
    vegas_buy: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (vegas_buy)")
    vegas_sell: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (vegas_sell)")
    thetas_buy: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (thetas_buy)")
    thetas_sell: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (thetas_sell)")
    volm_buy: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (volm_buy)")
    volm_sell: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (volm_sell)")
    value_buy: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (value_buy)")
    value_sell: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (value_sell)")

class UnprocessedDataBundleV2_5(BaseModel):
    options_contracts: List[RawOptionsContractV2_5] = Field(default_factory=list)
    underlying_data: ConsolidatedUnderlyingDataV2_5
    fetch_timestamp: datetime
    errors: List[str] = Field(default_factory=list)

class ProcessedContractMetricsV2_5(RawOptionsContractV2_5):
    """Enhanced options contract metrics with additional calculated fields.
    
    This model extends RawOptionsContractV2_5 with derived metrics and analytics
    for options trading strategies and risk management.
    """
    
    # 0DTE-specific metrics
    vri_0dte_contract: Optional[float] = Field(
        None,
        ge=0.0,
        description="Volume-Volatility Ratio for 0DTE contracts"
    )
    vfi_0dte_contract: Optional[float] = Field(
        None,
        ge=0.0,
        description="Volume Flow Index for 0DTE contracts"
    )
    vvr_0dte_contract: Optional[float] = Field(
        None,
        ge=0.0,
        description="Volume-Volatility Ratio for 0DTE contracts"
    )
    
    @classmethod
    def validate_positive_float(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("Value must be non-negative")
        return v
    
    @classmethod
    def validate_probability(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("VRI must be between 0.0 and 1.0")
        return v

class ProcessedStrikeLevelMetricsV2_5(BaseModel):
    """Comprehensive strike-level metrics for options analysis.
    
    This model contains aggregated metrics at each strike price level, including
    greeks exposure, flow analysis, and various derived indicators for trading signals.
    """
    
    # Core strike information
    strike: float = Field(..., gt=0, description="Strike price level")
    
    # Open interest metrics
    total_dxoi_at_strike: Optional[float] = Field(
        None, 
        description="Total Delta Open Interest at strike"
    )
    total_gxoi_at_strike: Optional[float] = Field(
        None, 
        description="Total Gamma Open Interest at strike"
    )
    total_vxoi_at_strike: Optional[float] = Field(
        None, 
        description="Total Vega Open Interest at strike"
    )
    total_txoi_at_strike: Optional[float] = Field(
        None, 
        description="Total Theta Open Interest at strike"
    )
    
    # Advanced greeks exposure
    total_charmxoi_at_strike: Optional[float] = Field(
        None, 
        description="Total Charm Open Interest at strike"
    )
    total_vannaxoi_at_strike: Optional[float] = Field(
        None, 
        description="Total Vanna Open Interest at strike"
    )
    total_vommaxoi_at_strike: Optional[float] = Field(
        None, 
        description="Total Vomma Open Interest at strike"
    )
    
    # Customer flow metrics
    net_cust_delta_flow_at_strike: Optional[float] = Field(
        None, 
        description="Net customer delta flow at strike"
    )
    net_cust_gamma_flow_at_strike: Optional[float] = Field(
        None, 
        description="Net customer gamma flow at strike"
    )
    net_cust_vega_flow_at_strike: Optional[float] = Field(
        None, 
        description="Net customer vega flow at strike"
    )
    net_cust_theta_flow_at_strike: Optional[float] = Field(
        None, 
        description="Net customer theta flow at strike"
    )
    net_cust_charm_flow_proxy_at_strike: Optional[float] = Field(
        None, 
        description="Net customer charm flow proxy at strike"
    )
    net_cust_vanna_flow_proxy_at_strike: Optional[float] = Field(
        None, 
        description="Net customer vanna flow proxy at strike"
    )
    
    # Advanced metrics
    nvp_at_strike: Optional[float] = Field(
        None, 
        description="Net vega position at strike"
    )
    nvp_vol_at_strike: Optional[float] = Field(
        None, 
        description="Net vega position volatility at strike"
    )
    
    # DAG (Directional Analysis Group) metrics
    a_dag_strike: Optional[float] = Field(
        None, 
        description="Adaptive Directional Analysis Group score"
    )
    a_dag_exposure: Optional[float] = Field(
        None, 
        description="DAG exposure metric"
    )
    a_dag_adaptive_alpha: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0,
        description="Adaptive alpha parameter for DAG"
    )
    
    # SDAG (Sentiment Directional Analysis Group) metrics
    e_sdag_mult_strike: Optional[float] = Field(
        None, 
        description="SDAG multiplier score at strike"
    )
    e_sdag_dir_strike: Optional[float] = Field(
        None, 
        ge=-1.0, 
        le=1.0,
        description="SDAG directional score at strike"
    )
    
    # Volatility and flow indicators
    vri_2_0_strike: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0,
        description="Volatility Risk Index 2.0 at strike"
    )
    
    # Concentration indices
    gci_strike: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0,
        description="Gamma Concentration Index at strike level"
    )
    dci_strike: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0,
        description="Delta Concentration Index at strike level"
    )
    
    # 0DTE specific metrics
    vri_0dte: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0,
        description="0DTE Volatility Risk Index"
    )
    vfi_0dte: Optional[float] = Field(
        None, 
        ge=0.0,
        description="0DTE Volume Flow Index"
    )
    vvr_0dte: Optional[float] = Field(
        None, 
        ge=0.0,
        description="0DTE Volume-Volatility Ratio"
    )
    
    # Additional metrics with validation
    a_mspi_strike: Optional[float] = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Market Sentiment and Positioning Index at strike"
    )
    
    # Validators
    @classmethod
    def validate_strike_metrics(cls, values):
        """Validate cross-field relationships and constraints."""
        # Example validation: If DCI is high, we might expect certain patterns in delta flows
        dci_strike = values.get('dci_strike')
        net_cust_delta_flow_at_strike = values.get('net_cust_delta_flow_at_strike')
        strike = values.get('strike')
        if dci_strike is not None and dci_strike > 0.8:
            if net_cust_delta_flow_at_strike is not None and abs(net_cust_delta_flow_at_strike) < 1000:
                logger.warning(
                    "High DCI (%f) with low delta flow (%f) at strike %f",
                    dci_strike,
                    net_cust_delta_flow_at_strike or 0.0,
                    strike
                )
        return values
    
    @classmethod
    def validate_probability_range(cls, v: Optional[float]) -> Optional[float]:
        """Validate that values are within 0.0 to 1.0 range."""
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("Value must be between 0.0 and 1.0")
        return v
    
    @classmethod
    def validate_neg_one_to_one_range(cls, v: Optional[float]) -> Optional[float]:
        """Validate that values are within -1.0 to 1.0 range."""
        if v is not None and not (-1.0 <= v <= 1.0):
            raise ValueError("Value must be between -1.0 and 1.0")
        return v

class DynamicThresholdsV2_5(BaseModel):
    """Dynamic thresholds for market regime detection and signal generation.
    
    This model defines configurable thresholds used throughout the system for:
    - Market regime classification
    - Signal generation and filtering
    - Risk management
    - Data quality assessment
    
    All thresholds are designed to be dynamically adjustable based on market conditions.
    """
    
    # ===== VAPI-FA (Volume-At-Price Imbalance - Flow Adjusted) Thresholds =====
    vapi_fa_bullish_thresh: float = Field(
        default=1.5, 
        gt=0.0,
        description="Z-score threshold for bullish VAPI-FA signals"
    )
    vapi_fa_bearish_thresh: float = Field(
        default=-1.5, 
        lt=0.0,
        description="Z-score threshold for bearish VAPI-FA signals"
    )

    # ===== Volatility Risk Index (VRI) Thresholds =====
    vri_bullish_thresh: float = Field(
        default=0.6, 
        gt=0.0,
        lt=1.0,
        description="VRI threshold for bullish regime classification"
    )
    vri_bearish_thresh: float = Field(
        default=-0.6, 
        gt=-1.0,
        lt=0.0,
        description="VRI threshold for bearish regime classification"
    )

    # ===== General Signal Thresholds =====
    negative_thresh_default: float = Field(
        default=-0.5, 
        lt=0.0,
        description="Default negative threshold for signal classification"
    )
    positive_thresh_default: float = Field(
        default=0.5, 
        gt=0.0,
        description="Default positive threshold for signal classification"
    )

    # ===== Significant Value Thresholds =====
    significant_pos_thresh: float = Field(
        default=1000.0, 
        gt=0,
        description="Threshold for significant positive values (e.g., OI, volume)"
    )
    significant_neg_thresh: float = Field(
        default=-1000.0, 
        lt=0,
        description="Threshold for significant negative values (e.g., net flows)"
    )

    # ===== Net Vega Position (NVP) Thresholds =====
    mid_high_nvp_thresh_pos: float = Field(
        default=5000.0, 
        gt=0,
        description="Mid-high threshold for Net Vega Position"
    )
    high_nvp_thresh_pos: float = Field(
        default=10000.0,
        gt=0,
        description="High threshold for Net Vega Position"
    )

    # ===== Advanced Regime Detection Thresholds =====
    dwfd_strong_thresh: float = Field(
        default=1.5, 
        gt=0.0,
        description="Threshold for strong DWFD (Daily/Weekly Flow Divergence) signal"
    )
    tw_laf_strong_thresh: float = Field(
        default=1.2, 
        gt=0.0,
        description="Threshold for strong TW-LAF (Time-Weighted Liquidity-Adjusted Flow) signal"
    )
    volatility_expansion_thresh: float = Field(
        default=0.8, 
        gt=0.0,
        le=1.0,
        description="Threshold for volatility expansion detection (0-1 scale)"
    )
    hedging_pressure_thresh: float = Field(
        default=500.0, 
        gt=0,
        description="Threshold for significant hedging pressure"
    )

    # ===== Confidence and Quality Thresholds =====
    high_confidence_thresh: float = Field(
        default=0.8, 
        gt=0.0,
        le=1.0,
        description="Minimum score for high confidence classification (0-1 scale)"
    )
    moderate_confidence_thresh: float = Field(
        default=0.6, 
        gt=0.0,
        le=1.0,
        description="Minimum score for moderate confidence classification (0-1 scale)"
    )
    data_quality_thresh: float = Field(
        default=0.7, 
        gt=0.0,
        le=1.0,
        description="Minimum acceptable data quality score (0-1 scale)"
    )
    
    # ===== Validation Methods =====
    @classmethod
    def validate_thresholds(cls, values):
        """Validate cross-field dependencies and business logic."""
        # Ensure bullish threshold is greater than bearish for VAPI-FA
        if values['vapi_fa_bullish_thresh'] <= abs(values['vapi_fa_bearish_thresh']):
            raise ValueError(
                f"Bullish threshold ({values['vapi_fa_bullish_thresh']}) must be greater than "
                f"absolute bearish threshold ({abs(values['vapi_fa_bearish_thresh'])})"
            )
        # Ensure high confidence threshold is greater than moderate
        if values['high_confidence_thresh'] <= values['moderate_confidence_thresh']:
            raise ValueError(
                f"High confidence threshold ({values['high_confidence_thresh']}) must be "
                f"greater than moderate threshold ({values['moderate_confidence_thresh']})"
            )
        # Ensure significant positive threshold is greater than negative
        if values['significant_pos_thresh'] <= abs(values['significant_neg_thresh']):
            raise ValueError(
                f"Significant positive threshold ({values['significant_pos_thresh']}) must be "
                f"greater than absolute negative threshold ({abs(values['significant_neg_thresh'])})"
            )
        return values
    
    @field_validator('vri_bullish_thresh', 'vri_bearish_thresh', mode='before')
    @classmethod
    def validate_vri_thresholds(cls, v: float, info: FieldValidationInfo) -> float:
        """Validate VRI thresholds are within -1 to 1 range."""
        if not (-1.0 <= v <= 1.0):
            raise ValueError(f"VRI threshold must be between -1.0 and 1.0, got {v}")
        return v
    
    @field_validator('high_confidence_thresh', 'moderate_confidence_thresh', 'data_quality_thresh', mode='before')
    @classmethod
    def validate_probability_thresholds(cls, v: float) -> float:
        """Validate that probability thresholds are between 0 and 1."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {v}")
        return v

class ConsolidatedUnderlyingDataV2_5(BaseModel):
    """Consolidated underlying data model - replaces RawUnderlyingDataV2_5, RawUnderlyingDataCombinedV2_5, and ProcessedUnderlyingAggregatesV2_5."""
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(default_factory=datetime.now, description="Data timestamp")
    price: Optional[float] = Field(None, description="Current price")
    price_change_abs: Optional[float] = Field(None, description="Absolute price change")
    price_change_pct: Optional[float] = Field(None, description="Percentage price change")
    day_open: Optional[float] = Field(None, description="Day open price")
    day_high: Optional[float] = Field(None, description="Day high price")
    day_low: Optional[float] = Field(None, description="Day low price")
    prev_close: Optional[float] = Field(None, description="Previous close price")
    day_volume: Optional[int] = Field(None, description="Day volume")
    call_gxoi: Optional[float] = Field(None, description="Call gamma exposure")
    put_gxoi: Optional[float] = Field(None, description="Put gamma exposure")
    net_delta_flow: Optional[float] = Field(None, description="Net delta flow")
    net_gamma_flow: Optional[float] = Field(None, description="Net gamma flow")
    net_vega_flow: Optional[float] = Field(None, description="Net vega flow")
    net_theta_flow: Optional[float] = Field(None, description="Net theta flow")



    
    # === Market Sentiment Metrics ===
    market_sentiment_score: Optional[float] = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Aggregated market sentiment score (-1.0 to 1.0)"
    )
    
    # === Volume Profile Metrics ===
    volume_profile_std: Optional[float] = Field(
        None,
        ge=0.0,
        description="Standard deviation of volume profile"
    )
    
    # === Volatility Metrics ===
    implied_volatility_percentile: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Implied volatility percentile (0-100)"
    )
    
    # === Liquidity Metrics ===
    liquidity_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Liquidity score (0-1 scale)"
    )
    
    # === Flow Metrics ===
    net_options_flow_usd: Optional[float] = Field(
        None,
        description="Net options flow in USD (positive for net buying pressure)"
    )
    
    # === Derived Analytics ===
    trend_strength: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Normalized trend strength (0-1 scale)"
    )
    
    # === Validation Methods ===
    @model_validator(mode='after')
    def validate_underlying_metrics(cls, values):
        """Validate cross-field relationships and derived metrics."""
        # Example: If we have both raw gamma and delta, we can validate their relationship
        if hasattr(cls, 'gib_raw_gamma_units') and hasattr(cls, 'net_delta'):
            # Add validation logic here if needed
            pass
            
        return values
    
    @field_validator('market_sentiment_score', 'liquidity_score', 'trend_strength', mode='before')
    @classmethod
    def validate_normalized_metrics(cls, v: Optional[float]) -> Optional[float]:
        """Ensure normalized metrics are within 0-1 or -1-1 range."""
        if v is not None:
            field_name = getattr(cls, '_current_field_name', 'field')
            if 'sentiment' in field_name and not (-1.0 <= v <= 1.0):
                raise ValueError(f"{field_name} must be between -1.0 and 1.0, got {v}")
            elif not (0.0 <= v <= 1.0):
                raise ValueError(f"{field_name} must be between 0.0 and 1.0, got {v}")
        return v
    

class ProcessedDataBundleV2_5(BaseModel):
    """Container for all processed market data and analytics.
    
    This is the top-level container that holds all processed market data,
    including options contracts, strike-level metrics, and underlying asset
    analytics. It enforces strict typing and validation of all contained data.
    
    Attributes:
        options_data_with_metrics: List of processed options contracts with metrics
        strike_level_data_with_metrics: Aggregated metrics at each strike price
        underlying_data_enriched: Enriched metrics for the underlying asset
        processing_timestamp: When the data processing occurred
        errors: Any errors or warnings generated during processing
        metadata: Additional metadata about the processing run
    """
    
    # Core data containers
    options_data_with_metrics: List[ProcessedContractMetricsV2_5] = Field(
        default_factory=list,
        description="List of processed options contracts with calculated metrics"
    )
    
    strike_level_data_with_metrics: List[ProcessedStrikeLevelMetricsV2_5] = Field(
        default_factory=list,
        description="Aggregated metrics at each strike price level"
    )
    
    underlying_data_enriched: ProcessedUnderlyingAggregatesV2_5 = Field(
        ...,
        description="Enriched metrics and analytics for the underlying asset"
    )
    
    # Metadata
    processing_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the data processing was completed"
    )
    
    errors: List[str] = Field(
        default_factory=list,
        description="Any errors or warnings generated during processing"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the processing run"
    )
    
    # ===== Validation Methods =====
    @model_validator(mode='after')
    def validate_data_consistency(cls, values):
        """Validate cross-model consistency and data integrity."""
        # Ensure we have data to work with
        if not values['options_data_with_metrics'] and not values['strike_level_data_with_metrics']:
            values['errors'].append("No options or strike level data provided")
            
        # Check timestamp is not in the future
        if values['processing_timestamp'] > datetime.utcnow() + timedelta(minutes=5):
            logger.warning(
                "Processing timestamp is more than 5 minutes in the future: %s",
                values['processing_timestamp']
            )
            
        # Validate underlying data if present
        if values['underlying_data_enriched']:
            cls._validate_underlying_data(values)
            
        return values
    
    def _validate_underlying_data(self, values):
        """Validate underlying data consistency."""
        # Add specific validations for underlying data
        pass
    
    # ===== Helper Methods =====
    def add_error(self, error_msg: str) -> None:
        """Add an error message to the errors list."""
        self.errors.append(f"{datetime.utcnow().isoformat()}: {error_msg}")
    
    def add_warning(self, warning_msg: str) -> None:
        """Add a warning message to the errors list."""
        self.add_error(f"WARNING: {warning_msg}")
    
    def is_valid(self) -> bool:
        """Check if the bundle is valid (no critical errors)."""
        # Filter out warnings (messages starting with 'WARNING:')
        critical_errors = [e for e in self.errors if not e.startswith("WARNING:")]
        return not bool(critical_errors)
    
    def summary(self) -> Dict[str, Any]:
        """Generate a summary of the data bundle."""
        return {
            "processing_timestamp": self.processing_timestamp,
            "options_contracts_count": len(self.options_data_with_metrics),
            "strike_levels_count": len(self.strike_level_data_with_metrics),
            "has_underlying_data": self.underlying_data_enriched is not None,
            "error_count": len([e for e in self.errors if not e.startswith("WARNING:")]),
            "warning_count": len([e for e in self.errors if e.startswith("WARNING:")]),
        }

class AdvancedOptionsMetricsV2_5(BaseModel):
    """
    Advanced options metrics for price action analysis based on liquidity and volatility.

    These metrics are derived from the "Options Contract Metrics for Price Action Analysis"
    document and provide sophisticated insights into market dynamics.
    """
    lwpai: Optional[float] = None  # Liquidity-Weighted Price Action Indicator
    vabai: Optional[float] = None  # Volatility-Adjusted Bid/Ask Imbalance
    aofm: Optional[float] = None   # Aggressive Order Flow Momentum
    lidb: Optional[float] = None   # Liquidity-Implied Directional Bias

    # Supporting metrics
    bid_ask_spread_percentage: Optional[float] = None
    total_liquidity_size: Optional[float] = None
    spread_to_volatility_ratio: Optional[float] = None
    theoretical_price_deviation: Optional[float] = None

    # Metadata
    valid_contracts_count: Optional[int] = None
    calculation_timestamp: Optional[datetime] = None
    confidence_score: Optional[float] = None  # 0-1 based on data quality
    data_quality_score: Optional[float] = None  # Additional data quality metric
    contracts_analyzed: Optional[int] = None  # Number of contracts analyzed

class TickerContextDictV2_5(BaseModel):
    is_0dte: Optional[bool] = None
    is_1dte: Optional[bool] = None
    is_spx_mwf_expiry_type: Optional[bool] = None
    is_spy_eom_expiry: Optional[bool] = None
    is_quad_witching_week_flag: Optional[bool] = None
    days_to_nearest_0dte: Optional[int] = None
    days_to_monthly_opex: Optional[int] = None
    is_fomc_meeting_day: Optional[bool] = None
    is_fomc_announcement_imminent: Optional[bool] = None
    post_fomc_drift_period_active: Optional[bool] = None
    vix_spy_price_divergence_strong_negative: Optional[bool] = None
    active_intraday_session: Optional[str] = None
    is_near_auction_period: Optional[bool] = None
    ticker_liquidity_profile_flag: Optional[str] = None
    ticker_volatility_state_flag: Optional[str] = None
    # NEW: Missing regime detection fields
    is_SPX_0DTE_Friday_EOD: Optional[bool] = None
    a_mspi_und_summary_score: Optional[float] = None
    nvp_by_strike: Optional[Dict[str, float]] = None
    hp_eod_und: Optional[float] = None
    trend_threshold: Optional[float] = None
    # NEW: Advanced options metrics integration
    advanced_options_metrics: Optional[AdvancedOptionsMetricsV2_5] = None
    earnings_approaching_flag: Optional[bool] = None
    days_to_earnings: Optional[int] = None

class SignalPayloadV2_5(BaseModel):
    signal_id: str
    signal_name: str
    symbol: str
    timestamp: datetime
    signal_type: str
    direction: Optional[str] = None
    strength_score: float
    strike_impacted: Optional[float] = None
    regime_at_signal_generation: Optional[str] = None
    supporting_metrics: Dict[str, Any] = Field(default_factory=dict)

class KeyLevelV2_5(BaseModel):
    level_price: float
    level_type: str
    conviction_score: float
    contributing_metrics: List[str] = Field(default_factory=list)
    source_identifier: Optional[str] = None

class KeyLevelsDataV2_5(BaseModel):
    supports: List[KeyLevelV2_5] = Field(default_factory=list)
    resistances: List[KeyLevelV2_5] = Field(default_factory=list)
    pin_zones: List[KeyLevelV2_5] = Field(default_factory=list)
    vol_triggers: List[KeyLevelV2_5] = Field(default_factory=list)
    major_walls: List[KeyLevelV2_5] = Field(default_factory=list)
    timestamp: datetime

class ATIFSituationalAssessmentProfileV2_5(BaseModel):
    bullish_assessment_score: float = 0.0
    bearish_assessment_score: float = 0.0
    vol_expansion_score: float = 0.0
    vol_contraction_score: float = 0.0
    mean_reversion_likelihood: float = 0.0
    timestamp: datetime

class ATIFStrategyDirectivePayloadV2_5(BaseModel):
    selected_strategy_type: str
    target_dte_min: int
    target_dte_max: int
    target_delta_long_leg_min: Optional[float] = None
    target_delta_long_leg_max: Optional[float] = None
    target_delta_short_leg_min: Optional[float] = None
    target_delta_short_leg_max: Optional[float] = None
    underlying_price_at_decision: float
    final_conviction_score_from_atif: float
    supportive_rationale_components: Dict[str, Any] = Field(default_factory=dict)
    assessment_profile: ATIFSituationalAssessmentProfileV2_5

class ATIFManagementDirectiveV2_5(BaseModel):
    recommendation_id: str
    action: str
    reason: str
    new_stop_loss: Optional[float] = None
    new_target_1: Optional[float] = None
    new_target_2: Optional[float] = None
    exit_price_type: Optional[str] = None
    percentage_to_manage: Optional[float] = None

class TradeParametersV2_5(BaseModel):
    """Defines the core parameters for an options trade with validation.
    
    This model enforces business rules for trade parameters including:
    - Strike price validation
    - Price consistency (stop loss < entry < targets)
    - Option symbol formatting
    - Expiration date parsing
    """
    
    # Core trade identifiers
    option_symbol: str = Field(
        ...,
        description="Option symbol in standard format (e.g., 'SPY_011923C420')",
        pattern=r'^[A-Z0-9_]+$',
        min_length=6,
        max_length=50
    )
    
    option_type: Literal['CALL', 'PUT'] = Field(
        ...,
        description="Option type, either 'CALL' or 'PUT'"
    )
    
    strike: float = Field(
        ...,
        gt=0,
        description="Strike price of the option"
    )
    
    expiration_str: str = Field(
        ...,
        description="Expiration date in YYYY-MM-DD format",
        pattern=r'^\d{4}-\d{2}-\d{2}$'
    )
    
    # Price levels
    entry_price_suggested: float = Field(
        ...,
        gt=0,
        description="Suggested entry price for the trade"
    )
    
    stop_loss_price: float = Field(
        ...,
        gt=0,
        description="Stop loss price level"
    )
    
    target_1_price: float = Field(
        ...,
        gt=0,
        description="Primary target price level"
    )
    
    target_2_price: Optional[float] = Field(
        None,
        gt=0,
        description="Secondary target price level (optional)"
    )
    
    target_3_price: Optional[float] = Field(
        None,
        gt=0,
        description="Tertiary target price level (optional)"
    )
    
    target_rationale: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Detailed rationale for the target prices"
    )
    
    # ===== Validation =====
    @model_validator(mode='after')
    def validate_price_levels(cls, values):
        """Ensure price levels are logically consistent."""
        option_type = values.get('option_type')
        entry_price_suggested = values.get('entry_price_suggested')
        stop_loss_price = values.get('stop_loss_price')
        target_1_price = values.get('target_1_price')
        target_2_price = values.get('target_2_price')
        target_3_price = values.get('target_3_price')
        if option_type == 'CALL':
            if stop_loss_price >= entry_price_suggested:
                raise ValueError("For CALLS: Stop loss must be below entry price")
            if target_1_price <= entry_price_suggested:
                raise ValueError("For CALLS: Target 1 must be above entry price")
        else:  # PUT
            if stop_loss_price <= entry_price_suggested:
                raise ValueError("For PUTS: Stop loss must be above entry price")
            if target_1_price >= entry_price_suggested:
                raise ValueError("For PUTS: Target 1 must be below entry price")
        if target_2_price is not None:
            if option_type == 'CALL':
                if target_2_price <= target_1_price:
                    raise ValueError("For CALLS: Target 2 must be greater than Target 1")
            else:
                if target_2_price >= target_1_price:
                    raise ValueError("For PUTS: Target 2 must be less than Target 1")
        if target_3_price is not None:
            if target_2_price is None:
                raise ValueError("Cannot have Target 3 without Target 2")
            if option_type == 'CALL':
                if target_3_price <= target_2_price:
                    raise ValueError("For CALLS: Target 3 must be greater than Target 2")
            else:
                if target_3_price >= target_2_price:
                    raise ValueError("For PUTS: Target 3 must be less than Target 2")
        return values
    
    # ===== Helper Methods =====
    @property
    def expiration_date(self) -> date:
        """Parse expiration string into date object."""
        return datetime.strptime(self.expiration_str, '%Y-%m-%d').date()
    
    @property
    def days_to_expiration(self) -> int:
        """Calculate days until expiration."""
        return (self.expiration_date - date.today()).days
    
    def risk_reward_ratio(self, target_num: int = 1) -> float:
        """Calculate risk/reward ratio for the specified target."""
        if target_num == 1:
            target = self.target_1_price
        elif target_num == 2 and self.target_2_price is not None:
            target = self.target_2_price
        elif target_num == 3 and self.target_3_price is not None:
            target = self.target_3_price
        else:
            raise ValueError(f"Invalid target number: {target_num}")
        
        risk = abs(self.entry_price_suggested - self.stop_loss_price)
        reward = abs(target - self.entry_price_suggested)
        return reward / risk if risk != 0 else float('inf')

class ActiveRecommendationPayloadV2_5(BaseModel):
    """Tracks the complete lifecycle of a trading recommendation.
    
    This model represents a trading recommendation from generation to completion,
    including all relevant parameters, status updates, and performance metrics.
    It enforces business rules for trade management and tracks the evolution of
    the trade through its lifecycle.
    """
    
    # Core identifiers
    recommendation_id: str = Field(
        ...,
        description="Unique identifier for this recommendation",
        min_length=8,
        max_length=64,
        pattern=r'^[a-zA-Z0-9_-]+$'
    )
    
    symbol: str = Field(
        ...,
        description="Trading symbol (e.g., 'SPY')",
        pattern=r'^[A-Z0-9.-]+$',
        max_length=20
    )
    
    timestamp_issued: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the recommendation was initially generated"
    )
    
    # Strategy details
    strategy_type: str = Field(
        ...,
        description="Type of trading strategy (e.g., 'Iron Condor', 'Bull Call Spread')",
        min_length=2,
        max_length=50
    )
    
    selected_option_details: List[Dict[str, Any]] = Field(
        ...,
        description="Detailed parameters for each leg of the options strategy"
    )
    
    # Trade parameters
    trade_bias: Literal['BULLISH', 'BEARISH', 'NEUTRAL', 'VOLATILITY'] = Field(
        ...,
        description="Overall bias of the trade"
    )
    
    entry_price_initial: float = Field(
        ...,
        gt=0,
        description="Initial entry price when the recommendation was created"
    )
    
    stop_loss_initial: float = Field(
        ...,
        gt=0,
        description="Initial stop loss level"
    )
    
    target_1_initial: float = Field(
        ...,
        gt=0,
        description="Initial primary target level"
    )
    
    target_2_initial: Optional[float] = Field(
        None,
        gt=0,
        description="Initial secondary target level (if any)"
    )
    
    target_3_initial: Optional[float] = Field(
        None,
        gt=0,
        description="Initial tertiary target level (if any)"
    )
    
    # Current trade state
    entry_price_actual: Optional[float] = Field(
        None,
        gt=0,
        description="Actual fill price (if entered)"
    )
    
    stop_loss_current: float = Field(
        ...,
        gt=0,
        description="Current stop loss level (may be adjusted)"
    )
    
    target_1_current: float = Field(
        ...,
        gt=0,
        description="Current primary target level (may be adjusted)"
    )
    
    target_2_current: Optional[float] = Field(
        None,
        gt=0,
        description="Current secondary target level (if any)"
    )
    
    target_3_current: Optional[float] = Field(
        None,
        gt=0,
        description="Current tertiary target level (if any)"
    )
    
    target_rationale: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Detailed rationale for the target prices and strategy"
    )
    
    # Status tracking
    status: Literal[
        'PENDING', 'ENTERED', 'TARGET_1_HIT', 'TARGET_2_HIT', 'TARGET_3_HIT',
        'STOPPED_OUT', 'CLOSED', 'EXPIRED', 'CANCELLED'
    ] = Field(
        default='PENDING',
        description="Current status of the recommendation"
    )
    
    status_update_reason: Optional[str] = Field(
        None,
        description="Reason for the most recent status update",
        max_length=500
    )
    
    # Analytics and metrics
    atif_conviction_score_at_issuance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Conviction score from ATIF when the recommendation was issued"
    )
    
    triggering_signals_summary: Optional[str] = Field(
        None,
        description="Summary of signals that triggered this recommendation",
        max_length=1000
    )
    
    regime_at_issuance: str = Field(
        ...,
        description="Market regime when the recommendation was issued"
    )
    
    # Exit information (if applicable)
    exit_timestamp: Optional[datetime] = Field(
        None,
        description="When the position was exited (if applicable)"
    )
    
    exit_price: Optional[float] = Field(
        None,
        gt=0,
        description="Exit price (if position was closed)"
    )
    
    pnl_percentage: Optional[float] = Field(
        None,
        description="Realized PnL as a percentage"
    )
    
    pnl_absolute: Optional[float] = Field(
        None,
        description="Realized PnL in absolute terms"
    )
    
    exit_reason: Optional[str] = Field(
        None,
        description="Reason for exiting the position",
        max_length=500
    )
    
    # ===== Validation Methods =====
    @model_validator(mode='after')
    def validate_recommendation(cls, values):
        """Validate the recommendation's internal consistency."""
        status = values.get('status')
        entry_price_actual = values.get('entry_price_actual')
        exit_timestamp = values.get('exit_timestamp')
        exit_price = values.get('exit_price')
        pnl_percentage = values.get('pnl_percentage')
        pnl_absolute = values.get('pnl_absolute')
        stop_loss_current = values.get('stop_loss_current')
        stop_loss_initial = values.get('stop_loss_initial')
        trade_bias = values.get('trade_bias')
        target_1_current = values.get('target_1_current')
        target_1_initial = values.get('target_1_initial')
        # Ensure entry price is set if status is beyond PENDING
        if status != 'PENDING' and entry_price_actual is None:
            raise ValueError("entry_price_actual must be set for non-pending recommendations")
        # Ensure exit fields are consistent
        if exit_timestamp is not None:
            if exit_price is None:
                raise ValueError("exit_price must be set when exit_timestamp is provided")
            if pnl_percentage is None or pnl_absolute is None:
                raise ValueError("PnL metrics must be set when exit_timestamp is provided")
        # Ensure current targets are not worse than initial
        if stop_loss_current is not None and stop_loss_initial is not None and trade_bias in ['BULLISH', 'VOLATILITY']:
            if stop_loss_current > stop_loss_initial:
                pass  # logger.warning("Stop loss moved in unfavorable direction for %s bias", trade_bias)
        if target_1_current is not None and target_1_initial is not None and trade_bias == 'BULLISH':
            if target_1_current < target_1_initial:
                pass  # logger.warning("Target 1 moved down for BULLISH bias")
        return values
    
    # ===== Helper Methods =====
    def calculate_risk_reward(self) -> Dict[str, float]:
        """Calculate risk/reward metrics for the recommendation."""
        if self.entry_price_actual is None:
            entry = self.entry_price_initial
        else:
            entry = self.entry_price_actual
        
        risk = abs(entry - self.stop_loss_current)
        
        # Calculate reward for each target
        rewards = {}
        for i, target in enumerate([self.target_1_current, self.target_2_current, self.target_3_current], 1):
            if target is not None:
                reward = abs(target - entry)
                rewards[f'target_{i}_rr'] = reward / risk if risk > 0 else float('inf')
        
        return rewards

# ===== AI PREDICTIONS MODELS =====

class AIPredictionV2_5(BaseModel):
    """Represents an AI-generated prediction for market analysis and trading decisions.
    
    This model captures all aspects of a market prediction, including the prediction itself,
    confidence levels, time horizons, and post-prediction validation. It's designed to work
    with various prediction types and includes comprehensive validation to ensure data integrity.
    
    Key Features:
    - Supports multiple prediction types (price, direction, volatility, etc.)
    - Tracks prediction accuracy and performance
    - Maintains market context for better interpretation
    - Includes timestamps for temporal analysis
    - Enforces data validation rules
    """
    
    # Core prediction identifiers
    id: Optional[int] = Field(
        None,
        description="Auto-generated database ID",
        ge=1
    )
    
    symbol: str = Field(
        ...,
        description="Trading symbol (e.g., 'SPY', 'QQQ')",
        pattern=r'^[A-Z0-9.-]+$',
        max_length=20
    )
    
    prediction_type: Literal['price', 'direction', 'volatility', 'eots_direction', 'sentiment'] = Field(
        ...,
        description="Type of prediction being made"
    )
    
    # Prediction details
    prediction_value: Optional[float] = Field(
        None,
        description="Predicted numerical value (for price, volatility, etc.)"
    )
    
    prediction_direction: Literal['UP', 'DOWN', 'NEUTRAL'] = Field(
        ...,
        description="Predicted market direction"
    )
    
    confidence_score: float = Field(
        ...,
        description="Model's confidence in the prediction (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    # Time-related fields
    time_horizon: str = Field(
        ...,
        description="Time frame for the prediction (e.g., '1H', '4H', '1D', '1W')",
        pattern=r'^\d+[mhdwMqy]$'
    )
    
    prediction_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the prediction was generated"
    )
    
    target_timestamp: datetime = Field(
        ...,
        description="When the prediction should be evaluated"
    )
    
    # Actual outcomes (filled in later)
    actual_value: Optional[float] = Field(
        None,
        description="Actual observed value (for validation)"
    )
    
    actual_direction: Optional[Literal['UP', 'DOWN', 'NEUTRAL']] = Field(
        None,
        description="Actual observed direction (for validation)"
    )
    
    # Performance metrics
    prediction_accurate: Optional[bool] = Field(
        None,
        description="Whether the prediction was accurate (for direction predictions)"
    )
    
    accuracy_score: Optional[float] = Field(
        None,
        description="Quantitative accuracy score (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    # Metadata
    model_version: str = Field(
        default="v2.5",
        description="Version of the prediction model used"
    )
    
    model_name: Optional[str] = Field(
        None,
        description="Name/identifier of the specific model used"
    )
    
    market_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Market conditions and features used for the prediction"
    )
    
    # Audit fields
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this record was created"
    )
    
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this record was last updated"
    )
    
    # ===== Validation Methods =====
    @model_validator(mode='after')
    def validate_prediction(self) -> 'AIPredictionV2_5':
        """Validate the prediction's internal consistency."""
        # Ensure target timestamp is in the future of prediction timestamp
        if self.target_timestamp <= self.prediction_timestamp:
            raise ValueError("target_timestamp must be after prediction_timestamp")
        
        # For price predictions, ensure we have a prediction value
        if self.prediction_type == 'price' and self.prediction_value is None:
            raise ValueError("prediction_value is required for price predictions")
        
        # For direction predictions, ensure direction is set
        if self.prediction_type == 'direction' and not self.prediction_direction:
            raise ValueError("prediction_direction is required for direction predictions")
        
        # If actual values are provided, validate them
        if self.actual_value is not None and self.actual_direction is None:
            # Auto-set direction based on value if possible
            if self.prediction_value is not None:
                if self.actual_value > self.prediction_value * 1.001:  # Small threshold for equality
                    self.actual_direction = 'UP'
                elif self.actual_value < self.prediction_value * 0.999:
                    self.actual_direction = 'DOWN'
                else:
                    self.actual_direction = 'NEUTRAL'
        
        # Calculate accuracy if we have both prediction and actual
        if self.actual_direction is not None and self.prediction_direction is not None:
            self.prediction_accurate = (self.actual_direction == self.prediction_direction)
            
            # For numerical predictions, calculate accuracy score
            if self.prediction_value is not None and self.actual_value is not None:
                error = abs(self.prediction_value - self.actual_value)
                # Simple accuracy score (1.0 for perfect prediction, decreasing with error)
                scale = max(abs(self.prediction_value) * 0.1, 0.01)  # Prevent division by zero
                self.accuracy_score = max(0.0, 1.0 - (error / scale))
        
        return self
    
    # ===== Helper Methods =====
    def update_with_actual(
        self, 
        actual_value: Optional[float] = None, 
        actual_direction: Optional[str] = None
    ) -> 'AIPredictionV2_5':
        """Update the prediction with actual outcomes and calculate accuracy."""
        if actual_value is not None:
            self.actual_value = actual_value
        if actual_direction is not None:
            self.actual_direction = actual_direction.upper() if actual_direction else None
        
        # Re-validate to update accuracy metrics
        self.validate_prediction()
        self.updated_at = datetime.utcnow()
        return self
    
    def is_expired(self) -> bool:
        """Check if the prediction's target time has passed."""
        return datetime.utcnow() > self.target_timestamp
    
    def time_until_target(self) -> timedelta:
        """Calculate time remaining until the target timestamp."""
        return self.target_timestamp - datetime.utcnow()
    
    def get_confidence_category(self) -> str:
        """Categorize the confidence level."""
        if self.confidence_score >= 0.8:
            return "HIGH"
        elif self.confidence_score >= 0.6:
            return "MEDIUM"
        elif self.confidence_score >= 0.4:
            return "LOW"
        return "VERY_LOW"

class AIPredictionPerformanceV2_5(BaseModel):
    """Tracks and analyzes the performance of AI predictions over time.
    
    This model provides comprehensive metrics for evaluating the performance of
    AI prediction models, including accuracy, confidence levels, and learning trends.
    It's designed to help monitor model health and guide model improvement efforts.
    
    Key Features:
    - Tracks prediction accuracy and success rates
    - Monitors model confidence and learning progress
    - Identifies performance trends over time
    - Supports model comparison and selection
    - Enables data-driven model improvement
    """
    
    # Core identifiers
    symbol: str = Field(
        ...,
        description="Trading symbol being analyzed",
        pattern=r'^[A-Z0-9.-]+$',
        max_length=20
    )
    
    model_name: str = Field(
        default="default",
        description="Name/identifier of the prediction model"
    )
    
    # Time period
    time_period_days: int = Field(
        ...,
        description="Number of days included in the performance analysis",
        ge=1,
        le=3650  # ~10 years
    )
    
    # Prediction counts
    total_predictions: int = Field(
        ...,
        description="Total number of predictions made in the period",
        ge=0
    )
    
    correct_predictions: int = Field(
        ...,
        description="Number of correct predictions (for categorical outcomes)",
        ge=0
    )
    
    incorrect_predictions: int = Field(
        ...,
        description="Number of incorrect predictions (for categorical outcomes)",
        ge=0
    )
    
    pending_predictions: int = Field(
        ...,
        description="Number of predictions awaiting verification",
        ge=0
    )
    
    # Performance metrics
    success_rate: float = Field(
        ...,
        description="Ratio of correct predictions to total verifiable predictions",
        ge=0.0,
        le=1.0
    )
    
    avg_confidence: float = Field(
        ...,
        description="Average confidence score across all predictions",
        ge=0.0,
        le=1.0
    )
    
    avg_accuracy_score: float = Field(
        ...,
        description="Average accuracy score (for numerical predictions)",
        ge=0.0,
        le=1.0
    )
    
    learning_score: float = Field(
        ...,
        description="Score representing model's learning progress (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    performance_trend: Literal['IMPROVING', 'STABLE', 'DECLINING', 'UNKNOWN'] = Field(
        ...,
        description="Trend of prediction performance over time"
    )
    
    # Additional metrics
    precision: Optional[float] = Field(
        None,
        description="Precision metric (true positives / (true positives + false positives))",
        ge=0.0,
        le=1.0
    )
    
    recall: Optional[float] = Field(
        None,
        description="Recall metric (true positives / (true positives + false negatives))",
        ge=0.0,
        le=1.0
    )
    
    f1_score: Optional[float] = Field(
        None,
        description="F1 score (harmonic mean of precision and recall)",
        ge=0.0,
        le=1.0
    )
    
    # Timestamp
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this performance record was last updated"
    )
    
    # ===== Validation Methods =====
    @model_validator(mode='after')
    def validate_performance_metrics(self) -> 'AIPredictionPerformanceV2_5':
        """Validate the consistency of performance metrics."""
        # Ensure prediction counts add up
        if self.total_predictions != (self.correct_predictions + 
                                   self.incorrect_predictions + 
                                   self.pending_predictions):
            raise ValueError("Prediction counts do not sum to total_predictions")
        
        # Validate success rate if we have verifiable predictions
        verifiable = self.correct_predictions + self.incorrect_predictions
        if verifiable > 0:
            calculated_success = self.correct_predictions / verifiable
            if not math.isclose(self.success_rate, calculated_success, abs_tol=0.01):
                raise ValueError(f"Success rate {self.success_rate} doesn't match "
                               f"calculated value {calculated_success}")
        
        # Ensure learning score is within bounds
        if not (0.0 <= self.learning_score <= 1.0):
            raise ValueError("learning_score must be between 0.0 and 1.0")
        
        return self
    
    # ===== Helper Methods =====
    def get_verifiable_predictions(self) -> int:
        """Get the number of predictions with known outcomes."""
        return self.correct_predictions + self.incorrect_predictions
    
    def get_verification_rate(self) -> float:
        """Calculate the percentage of predictions that have been verified."""
        if self.total_predictions == 0:
            return 0.0
        return self.get_verifiable_predictions() / self.total_predictions
    
    def get_confidence_ratio(self) -> float:
        """Calculate the ratio of average confidence to success rate."""
        if self.success_rate == 0:
            return 0.0
        return self.avg_confidence / self.success_rate
    
    def is_performing_well(self, min_success_rate: float = 0.6) -> bool:
        """Check if the model meets minimum performance criteria."""
        return (self.success_rate >= min_success_rate and 
                self.get_verification_rate() > 0.5 and
                self.performance_trend in ['IMPROVING', 'STABLE'])
    
    def to_metrics_dict(self) -> Dict[str, float]:
        """Convert performance metrics to a dictionary for logging/analysis."""
        return {
            'success_rate': self.success_rate,
            'avg_confidence': self.avg_confidence,
            'accuracy_score': self.avg_accuracy_score,
            'learning_score': self.learning_score,
            'verification_rate': self.get_verification_rate(),
            'confidence_ratio': self.get_confidence_ratio(),
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'verifiable_predictions': self.get_verifiable_predictions(),
            'is_performing_well': self.is_performing_well()
        }

class AIPredictionRequestV2_5(BaseModel):
    """Request model for creating new AI predictions.
    
    This model defines the structure for submitting prediction requests to the AI system.
    It includes all necessary fields for different types of market predictions with
    appropriate validation and documentation.
    
    Key Features:
    - Supports various prediction types (price, direction, volatility, etc.)
    - Flexible input format for different prediction scenarios
    - Comprehensive validation of all input parameters
    - Integration with market context for richer predictions
    """
    
    # Core prediction parameters
    symbol: str = Field(
        ...,
        description="Trading symbol (e.g., 'SPY', 'QQQ')",
        pattern=r'^[A-Z0-9.-]+$',
        max_length=20,
        examples=["SPY", "QQQ", "AAPL"]
    )
    
    prediction_type: Literal['price', 'direction', 'volatility', 'eots_direction', 'sentiment'] = Field(
        ...,
        description="Type of prediction being requested"
    )
    
    # Prediction details (at least one of value or direction should be provided)
    prediction_value: Optional[float] = Field(
        None,
        description="Predicted numerical value (required for 'price' and 'volatility' types)",
        ge=0.0
    )
    
    prediction_direction: Optional[Literal['UP', 'DOWN', 'NEUTRAL']] = Field(
        None,
        description="Predicted market direction (required for 'direction' and 'eots_direction' types)"
    )
    
    # Model and confidence
    confidence_score: float = Field(
        default=0.5,
        description="Model's confidence in the prediction (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    model_name: Optional[str] = Field(
        None,
        description="Name/version of the prediction model to use",
        max_length=100
    )
    
    
    # Time parameters
    time_horizon: str = Field(
        ...,
        description="Time frame for the prediction (e.g., '1H', '4H', '1D', '1W')",
        pattern=r'^\d+[mhdwMqy]$',
        examples=["1H", "4H", "1D", "1W"]
    )
    
    target_timestamp: datetime = Field(
        ...,
        description="When the prediction should be evaluated"
    )
    
    # Context and metadata
    market_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional market context and features for the prediction"
    )
    
    request_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the prediction request"
    )
    
    # ===== Validation Methods =====
    @model_validator(mode='after')
    def validate_prediction_request(self) -> 'AIPredictionRequestV2_5':
        """Validate the prediction request's internal consistency."""
        # Ensure at least one of prediction_value or prediction_direction is provided
        if self.prediction_value is None and self.prediction_direction is None:
            raise ValueError("At least one of prediction_value or prediction_direction must be provided")
        
        # Type-specific validation
        if self.prediction_type in ['price', 'volatility'] and self.prediction_value is None:
            raise ValueError(f"prediction_value is required for {self.prediction_type} predictions")
            
        if self.prediction_type in ['direction', 'eots_direction', 'sentiment'] and self.prediction_direction is None:
            raise ValueError(f"prediction_direction is required for {self.prediction_type} predictions")
        
        # Validate target timestamp is in the future
        if self.target_timestamp <= datetime.utcnow():
            raise ValueError("target_timestamp must be in the future")
        
        return self
    
    # ===== Helper Methods =====
    def get_time_horizon_minutes(self) -> int:
        """Convert time_horizon string to minutes."""
        unit = self.time_horizon[-1]
        value = int(self.time_horizon[:-1])
        
        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 1440  # 24 * 60
        elif unit == 'w':
            return value * 10080  # 7 * 24 * 60
        elif unit == 'M':
            return value * 43800  # 30.42 * 24 * 60 (avg month)
        elif unit == 'q':
            return value * 131400  # 3 * 30.42 * 24 * 60
        elif unit == 'y':
            return value * 525600  # 365 * 24 * 60
        return 0
    
    def to_prediction_dict(self) -> Dict[str, Any]:
        """Convert the request to a dictionary suitable for creating an AIPredictionV2_5."""
        return {
            'symbol': self.symbol,
            'prediction_type': self.prediction_type,
            'prediction_value': self.prediction_value,
            'prediction_direction': self.prediction_direction,
            'confidence_score': self.confidence_score,
            'time_horizon': self.time_horizon,
            'target_timestamp': self.target_timestamp,
            'market_context': self.market_context,
            'model_name': self.model_name
        }

# ===== AI ADAPTATIONS MODELS =====

class AIAdaptationV2_5(BaseModel):
    """Tracks AI model adaptations and adjustments for market conditions.
    
    This model represents adaptations made to AI models to improve their performance
    in different market conditions. It tracks the type, effectiveness, and implementation
    status of each adaptation.
    
    Key Features:
    - Tracks adaptation performance metrics
    - Manages implementation lifecycle
    - Records market context for adaptations
    - Supports A/B testing of different adaptations
    """
    
    # Core identifiers
    id: Optional[int] = Field(
        None,
        description="Auto-generated database ID",
        ge=1
    )
    
    # Adaptation details
    adaptation_type: Literal['signal_enhancement', 'threshold_adjustment', 'model_calibration', 'feature_engineering'] = Field(
        ...,
        description="Type of adaptation being applied"
    )
    
    adaptation_name: str = Field(
        ...,
        description="Human-readable name for the adaptation",
        min_length=3,
        max_length=100,
        pattern=r'^[a-zA-Z0-9_\- ]+$'
    )
    
    adaptation_description: Optional[str] = Field(
        None,
        description="Detailed description of the adaptation and its purpose",
        max_length=1000
    )
    
    # Performance metrics
    confidence_score: float = Field(
        default=0.0,
        description="Confidence in adaptation effectiveness (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    success_rate: float = Field(
        default=0.0,
        description="Historical success rate of this adaptation (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    adaptation_score: float = Field(
        default=0.0,
        description="Overall adaptation performance score (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    # Implementation details
    implementation_status: Literal['PENDING', 'ACTIVE', 'INACTIVE', 'DEPRECATED', 'TESTING'] = Field(
        default="PENDING",
        description="Current status of the adaptation"
    )
    
    # Context and metadata
    market_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Market conditions when this adaptation was created"
    )
    
    performance_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed performance tracking metrics"
    )
    
    # References and relationships
    parent_model_id: Optional[str] = Field(
        None,
        description="ID of the model this adaptation applies to"
    )
    
    # Audit fields
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this adaptation was created"
    )
    
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this adaptation was last updated"
    )
    
    # ===== Validation Methods =====
    @model_validator(mode='after')
    def validate_adaptation(self) -> 'AIAdaptationV2_5':
        """Validate the adaptation's internal consistency."""
        # Ensure adaptation score is consistent with confidence and success rate
        if self.adaptation_score > max(self.confidence_score, self.success_rate):
            self.adaptation_score = max(self.confidence_score, self.success_rate)
        
        # If active, ensure we have sufficient confidence
        if self.implementation_status == 'ACTIVE' and self.confidence_score < 0.7:
            warnings.warn("Active adaptation has low confidence score", UserWarning)
            
        return self
    
    # ===== Helper Methods =====
    def is_active(self) -> bool:
        """Check if this adaptation is currently active."""
        return self.implementation_status == 'ACTIVE'
    
    def update_performance(
        self, 
        success: bool, 
        impact_score: float,
        market_context: Optional[Dict[str, Any]] = None
    ) -> 'AIAdaptationV2_5':
        """Update adaptation performance metrics."""
        # Update metrics
        total_tests = self.performance_metrics.get('total_tests', 0)
        successful_tests = self.performance_metrics.get('successful_tests', 0)
        
        total_tests += 1
        if success:
            successful_tests += 1
        
        # Update metrics
        self.success_rate = successful_tests / total_tests if total_tests > 0 else 0.0
        self.performance_metrics.update({
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'last_test_result': success,
            'last_test_time': datetime.utcnow(),
            'impact_score': impact_score
        })
        
        # Update market context if provided
        if market_context:
            self.market_context.update(market_context)
        
        # Update timestamps
        self.updated_at = datetime.utcnow()
        
        return self
    
    def to_metrics_dict(self) -> Dict[str, Any]:
        """Convert adaptation to a metrics dictionary for analysis."""
        return {
            'adaptation_id': self.id,
            'adaptation_type': self.adaptation_type,
            'confidence_score': self.confidence_score,
            'success_rate': self.success_rate,
            'adaptation_score': self.adaptation_score,
            'status': self.implementation_status,
            'total_tests': self.performance_metrics.get('total_tests', 0),
            'successful_tests': self.performance_metrics.get('successful_tests', 0),
            'last_test_result': self.performance_metrics.get('last_test_result'),
            'impact_score': self.performance_metrics.get('impact_score', 0.0)
        }

    model_config = ConfigDict(extra='allow')

class AIAdaptationPerformanceV2_5(BaseModel):
    """Tracks performance metrics for AI model adaptations over time.
    
    This model provides detailed performance tracking for AI adaptations, including
    success rates, improvement metrics, and performance trends. It helps evaluate
    the effectiveness of different adaptations in various market conditions.
    
    Key Features:
    - Tracks adaptation success rates and improvements
    - Monitors performance trends over time
    - Provides metrics for adaptation evaluation
    - Supports performance-based adaptation selection
    """
    model_config = ConfigDict(
        extra='forbid',
        frozen=False,
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
        },
        json_schema_extra={
            'examples': [{
                'adaptation_id': 'adapt_12345',
                'symbol': 'SPY',
                'time_period_days': 30,
                'total_applications': 100,
                'successful_applications': 75,
                'success_rate': 0.75,
                'avg_improvement': 0.15,
                'adaptation_score': 0.45,
                'performance_trend': 'IMPROVING',
                'avg_processing_time_ms': 250.5,
                'market_conditions': {},
                'last_updated': '2023-01-01T12:00:00Z'
            }]
        }
    )
    
    # Core identifiers
    adaptation_id: int = Field(
        ...,
        description="Reference to the adaptation ID this performance data belongs to",
        ge=1
    )
    
    symbol: str = Field(
        ...,
        description="Trading symbol this adaptation applies to",
        pattern=r'^[A-Z0-9.-]+$',
        max_length=20
    )
    
    # Time period
    time_period_days: int = Field(
        ...,
        description="Number of days this performance data covers",
        ge=1,
        le=3650  # ~10 years
    )
    
    # Application metrics
    total_applications: int = Field(
        ...,
        description="Total number of times this adaptation was applied",
        ge=0
    )
    
    successful_applications: int = Field(
        ...,
        description="Number of successful applications (met or exceeded performance threshold)",
        ge=0
    )
    
    # Performance metrics
    success_rate: float = Field(
        ...,
        description="Ratio of successful applications to total applications (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    avg_improvement: float = Field(
        ...,
        description="Average improvement in the target metric (e.g., accuracy, profit factor)",
        ge=0.0
    )
    
    adaptation_score: float = Field(
        ...,
        description="Overall performance score combining success rate and improvement (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    performance_trend: Literal['IMPROVING', 'STABLE', 'DECLINING', 'UNKNOWN'] = Field(
        ...,
        description="Trend of adaptation performance over recent evaluations"
    )
    
    # Additional metrics
    avg_processing_time_ms: float = Field(
        default=0.0,
        description="Average processing time in milliseconds",
        ge=0.0
    )
    
    market_conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Market conditions during this performance period"
    )
    
    # Timestamp
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this performance record was last updated"
    )
    
    # ===== Validation Methods =====
    @model_validator(mode='after')
    def validate_performance_metrics(self) -> 'AIAdaptationPerformanceV2_5':
        """Validate the performance metrics' internal consistency."""
        # Ensure successful_applications doesn't exceed total_applications
        if self.successful_applications > self.total_applications:
            raise ValueError(
                f"successful_applications ({self.successful_applications}) cannot exceed "
                f"total_applications ({self.total_applications})"
            )
        
        # Calculate success rate if we have applications
        if self.total_applications > 0:
            calculated_rate = self.successful_applications / self.total_applications
            
            # Allow small floating point differences
            if not math.isclose(self.success_rate, calculated_rate, abs_tol=0.001):
                raise ValueError(
                    f"success_rate {self.success_rate} doesn't match "
                    f"calculated rate {calculated_rate}"
                )
        
        # Ensure adaptation_score is consistent with success_rate and avg_improvement
        # Normalize avg_improvement to 0-1 range (assuming max improvement of 100%)
        normalized_improvement = min(self.avg_improvement, 1.0)
        expected_score = (self.success_rate + normalized_improvement) / 2
        
        if not math.isclose(self.adaptation_score, expected_score, abs_tol=0.1):
            # Auto-correct if the difference is small, otherwise raise error
            if abs(self.adaptation_score - expected_score) < 0.2:
                self.adaptation_score = expected_score
            else:
                raise ValueError(
                    f"adaptation_score {self.adaptation_score} is inconsistent with "
                    f"success_rate {self.success_rate} and avg_improvement {self.avg_improvement}"
                )
        
        return self
    
    # ===== Helper Methods =====
    def is_improving(self) -> bool:
        """Check if the adaptation performance is improving."""
        return self.performance_trend == 'IMPROVING'
    
    def is_above_threshold(self, min_success_rate: float = 0.6) -> bool:
        """Check if the adaptation meets minimum performance criteria."""
        return self.success_rate >= min_success_rate
    
    def get_effectiveness_category(self) -> str:
        """Categorize the adaptation's effectiveness."""
        if self.adaptation_score >= 0.8:
            return "HIGHLY_EFFECTIVE"
        elif self.adaptation_score >= 0.6:
            return "EFFECTIVE"
        elif self.adaptation_score >= 0.4:
            return "MODERATELY_EFFECTIVE"
        return "INEFFECTIVE"
    
    def to_metrics_dict(self) -> Dict[str, Any]:
        """Convert performance metrics to a dictionary for analysis."""
        return {
            'adaptation_id': self.adaptation_id,
            'symbol': self.symbol,
            'time_period_days': self.time_period_days,
            'total_applications': self.total_applications,
            'successful_applications': self.successful_applications,
            'success_rate': self.success_rate,
            'avg_improvement': self.avg_improvement,
            'adaptation_score': self.adaptation_score,
            'performance_trend': self.performance_trend,
            'effectiveness_category': self.get_effectiveness_category(),
            'is_improving': self.is_improving(),
            'last_updated': self.last_updated.isoformat()
        }



class AIAdaptationRequestV2_5(BaseModel):
    """Request model for creating new AI model adaptations.
    
    This model defines the structure for submitting adaptation requests to the AI system.
    It includes all necessary fields to define and configure a new adaptation, including
    type, name, description, and initial configuration.
    
    Key Features:
    - Supports various adaptation types (signal enhancement, threshold adjustment, etc.)
    - Includes confidence scoring and market context
    - Validates adaptation parameters
    - Tracks request metadata
    """
    
    # Adaptation details
    adaptation_type: Literal['signal_enhancement', 'threshold_adjustment', 'model_calibration', 'feature_engineering'] = Field(
        ...,
        description="Type of adaptation being requested"
    )
    
    adaptation_name: str = Field(
        ...,
        description="Unique name for the adaptation (alphanumeric with spaces and underscores)",
        min_length=3,
        max_length=100,
        pattern=r'^[a-zA-Z0-9_\- ]+$',
        examples=["volatility_threshold_adjustment", "sentiment_signal_enhancement"]
    )
    
    adaptation_description: Optional[str] = Field(
        None,
        description="Detailed description of the adaptation and its intended purpose",
        max_length=1000
    )
    
    # Model and confidence
    confidence_score: float = Field(
        default=0.5,
        description="Initial confidence score (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    # Target model
    target_model_id: Optional[str] = Field(
        None,
        description="ID of the model this adaptation applies to (None for all models)",
        min_length=1,
        max_length=100
    )
    
    # Configuration
    adaptation_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration parameters specific to this adaptation type"
    )
    
    # Context and metadata
    market_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current market conditions and context for this adaptation"
    )
    
    request_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about this adaptation request"
    )
    
    # Timestamp
    requested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this adaptation was requested"
    )
    
    # ===== Validation Methods =====
    @model_validator(mode='after')
    def validate_adaptation_request(self) -> 'AIAdaptationRequestV2_5':
        """Validate the adaptation request's internal consistency."""
        # Validate adaptation parameters based on type
        if self.adaptation_type == 'threshold_adjustment':
            if 'threshold_name' not in self.adaptation_parameters:
                raise ValueError("threshold_adjustment requires 'threshold_name' in adaptation_parameters")
            if 'new_value' not in self.adaptation_parameters:
                raise ValueError("threshold_adjustment requires 'new_value' in adaptation_parameters")
        
        elif self.adaptation_type == 'signal_enhancement':
            if 'signal_name' not in self.adaptation_parameters:
                raise ValueError("signal_enhancement requires 'signal_name' in adaptation_parameters")
        
        # Validate confidence score is reasonable for the request
        if self.confidence_score > 0.8 and not self.adaptation_description:
            warnings.warn(
                "High confidence adaptation requested without detailed description. "
                "Please provide more context in adaptation_description.",
                UserWarning
            )
        
        return self
    
    # ===== Helper Methods =====
    def get_adaptation_key(self) -> str:
        """Generate a unique key for this adaptation request."""
        target = self.target_model_id or 'ALL_MODELS'
        return f"{self.adaptation_type}:{target}:{self.adaptation_name}"
    
    def to_adaptation_dict(self) -> Dict[str, Any]:
        """Convert the request to a dictionary for creating an AIAdaptationV2_5."""
        return {
            'adaptation_type': self.adaptation_type,
            'adaptation_name': self.adaptation_name,
            'adaptation_description': self.adaptation_description,
            'confidence_score': self.confidence_score,
            'market_context': self.market_context,
            'performance_metrics': {
                'initial_confidence': self.confidence_score,
                'created_from_request': self.request_metadata.get('request_id', 'unknown')
            }
        }
    
    def get_impact_assessment(self) -> Dict[str, Any]:
        """Assess the potential impact of this adaptation."""
        impact = {
            'estimated_impact': 'MODERATE',
            'risk_level': 'MEDIUM',
            'affected_models': [self.target_model_id] if self.target_model_id else 'ALL',
            'confidence': self.confidence_score,
            'requires_approval': self.confidence_score < 0.7
        }
        
        # Adjust impact based on adaptation type
        if self.adaptation_type == 'threshold_adjustment':
            impact['estimated_impact'] = 'LOW'
            impact['risk_level'] = 'LOW'
        elif self.adaptation_type == 'model_calibration':
            impact['estimated_impact'] = 'HIGH'
            impact['risk_level'] = 'HIGH'
        
        return impact

    model_config = ConfigDict(extra='forbid')

# ===== HUIHUI PYDANTIC-FIRST SCHEMAS =====

class HuiHuiExpertType(str, Enum):
    """HuiHui expert types for EOTS system."""
    MARKET_REGIME = "market_regime"
    OPTIONS_FLOW = "options_flow"
    SENTIMENT = "sentiment"
    ORCHESTRATOR = "orchestrator"

class HuiHuiModelConfigV2_5(BaseModel):
    """PYDANTIC-FIRST: HuiHui model configuration."""
    expert_type: HuiHuiExpertType = Field(default=HuiHuiExpertType.ORCHESTRATOR)
    temperature: float = Field(ge=0.0, le=2.0, default=0.1)
    max_tokens: int = Field(ge=100, le=8000, default=4000)
    enable_eots_integration: bool = Field(default=True)
    context_budget: int = Field(ge=1000, le=8000, default=4000)
    timeout_seconds: int = Field(ge=30, le=300, default=90)

class HuiHuiExpertConfigV2_5(BaseModel):
    """PYDANTIC-FIRST: Individual expert configuration."""
    expert_name: str
    specialist_id: str
    temperature: float = Field(ge=0.0, le=2.0)
    keywords: List[str] = Field(default_factory=list)
    eots_metrics: List[str] = Field(default_factory=list)
    performance_weight: float = Field(ge=0.0, le=1.0, default=1.0)

class HuiHuiAnalysisRequestV2_5(BaseModel):
    """PYDANTIC-FIRST: HuiHui analysis request."""
    symbol: str
    analysis_type: str
    bundle_data: Optional['FinalAnalysisBundleV2_5'] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    expert_preference: Optional[HuiHuiExpertType] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class HuiHuiAnalysisResponseV2_5(BaseModel):
    """PYDANTIC-FIRST: HuiHui analysis response."""
    expert_used: HuiHuiExpertType
    analysis_content: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    processing_time: float = Field(ge=0.0)
    insights: List[str] = Field(default_factory=list)
    eots_predictions: Optional[List[AIPredictionV2_5]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class HuiHuiUsageRecordV2_5(BaseModel):
    """PYDANTIC-FIRST: HuiHui usage tracking."""
    expert_used: HuiHuiExpertType
    symbol: str
    processing_time: float = Field(ge=0.0)
    success: bool
    error_message: Optional[str] = None
    market_condition: str = Field(default="normal")
    timestamp: datetime = Field(default_factory=datetime.now)

class HuiHuiPerformanceMetricsV2_5(BaseModel):
    """PYDANTIC-FIRST: HuiHui performance metrics."""
    expert_type: HuiHuiExpertType
    total_requests: int = Field(ge=0, default=0)
    successful_requests: int = Field(ge=0, default=0)
    average_processing_time: float = Field(ge=0.0, default=0.0)
    success_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    last_updated: datetime = Field(default_factory=datetime.now)

class FinalAnalysisBundleV2_5(BaseModel):
    processed_data_bundle: ProcessedDataBundleV2_5
    scored_signals_v2_5: Dict[str, List[SignalPayloadV2_5]] = Field(default_factory=dict)
    key_levels_data_v2_5: KeyLevelsDataV2_5
    active_recommendations_v2_5: List[ActiveRecommendationPayloadV2_5] = Field(default_factory=list)
    bundle_timestamp: datetime
    target_symbol: str
    system_status_messages: List[str] = Field(default_factory=list)
    atif_recommendations_v2_5: Optional[List[ATIFStrategyDirectivePayloadV2_5]] = None
    news_intelligence_v2_5: Optional[Dict[str, Any]] = Field(None, description="Diabolical news intelligence analysis")
    ai_predictions_v2_5: Optional[List[AIPredictionV2_5]] = Field(None, description="AI predictions for this analysis")
    """
    The main analysis bundle for the dashboard. Now includes ATIF recommendations, diabolical news intelligence, and AI predictions for apex predator analysis.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

class TimeOfDayDefinitions(BaseModel):
    market_open: str = Field("09:30:00", description="Market open time in HH:MM:SS format")
    market_close: str = Field("16:00:00", description="Market close time in HH:MM:SS format")
    pre_market_start: str = Field("04:00:00", description="Pre-market start time in HH:MM:SS format")
    after_hours_end: str = Field("20:00:00", description="After hours end time in HH:MM:SS format")
    eod_pressure_calc_time: str = Field("15:00:00", description="Time for end-of-day pressure calculations in HH:MM:SS format")

    model_config = ConfigDict(extra='forbid')

# Dashboard Mode Configuration Models
class DashboardModeSettings(BaseModel):
    """Defines a single dashboard mode configuration"""
    label: str = Field(..., description="Display label for the mode")
    module_name: str = Field(..., description="Python module name to import for this mode")
    charts: List[str] = Field(default_factory=list, description="List of chart/component names to display in this mode")
    
    model_config = ConfigDict(extra='forbid')

class MainDashboardDisplaySettings(BaseModel):
    """Settings specific to main dashboard display components"""
    regime_indicator: Dict[str, Any] = Field(default_factory=lambda: {
        "title": "Market Regime",
        "regime_colors": {
            "default": "secondary",
            "bullish": "success", 
            "bearish": "danger",
            "neutral": "info",
            "unclear": "warning"
        }
    })
    flow_gauge: Dict[str, Any] = Field(default_factory=lambda: {
        "height": 200,
        "indicator_font_size": 16,
        "number_font_size": 24,
        "axis_range": [-3, 3],
        "threshold_line_color": "white",
        "margin": {"t": 60, "b": 40, "l": 20, "r": 20},
        "steps": [
            {"range": [-3, -2], "color": "#d62728"},
            {"range": [-2, -0.5], "color": "#ff9896"},
            {"range": [-0.5, 0.5], "color": "#aec7e8"},
            {"range": [0.5, 2], "color": "#98df8a"},
            {"range": [2, 3], "color": "#2ca02c"}
        ]
    })
    gib_gauge: Dict[str, Any] = Field(default_factory=lambda: {
        "height": 180,
        "indicator_font_size": 14,
        "number_font_size": 20,
        "axis_range": [-1, 1],
        "dollar_axis_range": [-1000000, 1000000],
        "threshold_line_color": "white",
        "margin": {"t": 50, "b": 30, "l": 15, "r": 15},
        "steps": [
            {"range": [-1, -0.5], "color": "#d62728"},
            {"range": [-0.5, -0.1], "color": "#ff9896"},
            {"range": [-0.1, 0.1], "color": "#aec7e8"},
            {"range": [0.1, 0.5], "color": "#98df8a"},
            {"range": [0.5, 1], "color": "#2ca02c"}
        ],
        "dollar_steps": [
            {"range": [-1000000, -500000], "color": "#d62728"},
            {"range": [-500000, -100000], "color": "#ff9896"},
            {"range": [-100000, 100000], "color": "#aec7e8"},
            {"range": [100000, 500000], "color": "#98df8a"},
            {"range": [500000, 1000000], "color": "#2ca02c"}
        ]
    })
    mini_heatmap: Dict[str, Any] = Field(default_factory=lambda: {
        "height": 150,
        "colorscale": "RdYlGn",
        "margin": {"t": 50, "b": 30, "l": 40, "r": 40}
    })
    recommendations_table: Dict[str, Any] = Field(default_factory=lambda: {
        "title": "ATIF Recommendations",
        "max_rationale_length": 50,
        "page_size": 5,
        "style_cell": {"textAlign": "left", "padding": "5px", "minWidth": "80px", "width": "auto", "maxWidth": "200px"},
        "style_header": {"backgroundColor": "rgb(30, 30, 30)", "fontWeight": "bold", "color": "white"},
        "style_data": {"backgroundColor": "rgb(50, 50, 50)", "color": "white"}
    })
    ticker_context: Dict[str, Any] = Field(default_factory=lambda: {
        "title": "Ticker Context"
    })
    
    model_config = ConfigDict(extra='forbid')

class DashboardModeCollection(BaseModel):
    """Collection of all dashboard modes"""
    main: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Main Dashboard",
        module_name="main_dashboard_display_v2_5",
        charts=["regime_display", "flow_gauges", "gib_gauges", "recommendations_table"]
    ))
    flow: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Flow Analysis",
        module_name="flow_mode_display_v2_5",
        charts=["net_value_heatmap_viz", "net_cust_delta_flow_viz", "net_cust_gamma_flow_viz", "net_cust_vega_flow_viz"]
    ))
    structure: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Structure & Positioning",
        module_name="structure_mode_display_v2_5",
        charts=["mspi_components", "sai_ssi_displays"]
    ))
    timedecay: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Time Decay & Pinning",
        module_name="time_decay_mode_display_v2_5",
        charts=["tdpi_displays", "vci_strike_charts"]
    ))
    advanced: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Advanced Flow Metrics",
        module_name="advanced_flow_mode_display_v2_5",
        charts=["vapi_gauges", "dwfd_gauges", "tw_laf_gauges"]
    ))
    volatility: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Volatility Deep Dive",
        module_name="volatility_mode_display_v2_5",
        charts=["vri_2_0_strike_profile", "volatility_gauges", "volatility_surface_heatmap"]
    ))
    ai: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="AI Intelligence Hub",
        module_name="ai_dashboard.ai_dashboard_display_v2_5",
        charts=["ai_market_analysis", "ai_recommendations", "ai_insights", "ai_regime_context", "ai_performance_tracker"]
    ))

    model_config = ConfigDict(extra='forbid')



class RegimeRule(BaseModel):
    """Individual rule for market regime evaluation."""
    metric: str = Field(..., description="Metric name to evaluate")
    operator: str = Field(..., description="Comparison operator (_gt, _lt, _eq, etc.)")
    value: Union[str, float, int, bool] = Field(..., description="Target value for comparison")
    selector: Optional[str] = Field(None, description="Optional selector like @ATM")
    aggregator: Optional[str] = Field(None, description="Optional aggregation method")

    model_config = ConfigDict(extra='forbid')

class MarketRegimeEngineSettings(BaseModel):
    default_regime: str = Field(default="REGIME_UNCLEAR_OR_TRANSITIONING", description="Default market regime")
    regime_evaluation_order: List[str] = Field(
        default=[
            "REGIME_SPX_0DTE_FRIDAY_EOD_VANNA_CASCADE_POTENTIAL_BULLISH",
            "REGIME_SPY_PRE_FOMC_VOL_COMPRESSION_WITH_DWFD_ACCUMULATION",
            "REGIME_HIGH_VAPI_FA_BULLISH_MOMENTUM_UNIVERSAL",
            "REGIME_ADAPTIVE_STRUCTURE_BREAKDOWN_WITH_DWFD_CONFIRMATION_BEARISH_UNIVERSAL",
            "REGIME_VOL_EXPANSION_IMMINENT_VRI0DTE_BULLISH",
            "REGIME_VOL_EXPANSION_IMMINENT_VRI0DTE_BEARISH",
            "REGIME_NVP_STRONG_BUY_IMBALANCE_AT_KEY_STRIKE",
            "REGIME_EOD_HEDGING_PRESSURE_BUY",
            "REGIME_EOD_HEDGING_PRESSURE_SELL",
            "REGIME_SIDEWAYS_MARKET",
            "REGIME_HIGH_VOLATILITY"
        ],
        description="Order in which to evaluate market regimes"
    )
    regime_rules: Dict[str, List[RegimeRule]] = Field(
        default_factory=dict,
        description="Rules for different market regimes - each regime has a list of rules (AND logic)"
    )

    model_config = ConfigDict(extra='forbid')

class SystemSettings(BaseModel):
    project_root_override: Optional[str] = Field(None, description="Absolute path to override the auto-detected project root. Use null for auto-detection.")
    logging_level: str = Field("INFO", description="The minimum level of logs to record.")
    log_to_file: bool = Field(True, description="If true, logs will be written to the file specified in log_file_path.")
    log_file_path: str = Field("logs/eots_v2_5.log", description="Relative path from project root for the log file.", pattern="\\.log$")
    max_log_file_size_bytes: int = Field(10485760, description="Maximum size of a single log file in bytes before rotation.", ge=1024)
    backup_log_count: int = Field(5, description="Number of old log files to keep after rotation.", ge=0)
    live_mode: bool = Field(True, description="If true, system will only use live data sources and fail fast on errors.")
    fail_fast_on_errors: bool = Field(True, description="If true, system will halt on any data quality or API errors.")
    enable_ai_intelligence: bool = Field(True, description="Enable AI intelligence features including news analysis and unified AI orchestration.")
    enable_unified_orchestrator: bool = Field(True, description="Enable unified AI orchestrator with MCP servers and enhanced intelligence.")
    enable_multi_database: bool = Field(True, description="Enable multi-database support for enhanced data storage.")
    metrics_for_dynamic_threshold_distribution_tracking: List[str] = Field(
        default=["GIB_OI_based_Und", "VAPI_FA_Z_Score_Und", "DWFD_Z_Score_Und", "TW_LAF_Z_Score_Und"],
        description="List of underlying aggregate metric names to track historically for dynamic threshold calculations."
    )
    signal_activation: Dict[str, Any] = Field(default_factory=lambda: {"EnableAllSignals": True}, description="Toggles for enabling or disabling specific signal generation routines.")

class ConvexValueAuthSettings(BaseModel):
    use_env_variables: bool = Field(True, description="Whether to use environment variables for authentication.")
    auth_method: str = Field("email_password", description="Authentication method for ConvexValue API.")

class DataFetcherSettings(BaseModel):
    convexvalue_auth: ConvexValueAuthSettings = Field(default_factory=lambda: ConvexValueAuthSettings(
        use_env_variables=True,
        auth_method="email_password"
    ), description="Authentication settings for ConvexValue.")
    tradier_api_key: str = Field(..., description="API Key for Tradier.")
    tradier_account_id: str = Field(..., description="Account ID for Tradier.")
    max_retries: int = Field(3, description="Maximum number of retry attempts for a failing API call.", ge=0)
    retry_delay_seconds: float = Field(5, description="Base delay in seconds between API call retries.", ge=0)

class DataManagementSettings(BaseModel):
    data_cache_dir: str = Field("data_cache_v2_5", description="Directory for caching data.")
    historical_data_store_dir: str = Field("data_cache_v2_5/historical_data_store", description="Directory for historical data storage.")
    performance_data_store_dir: str = Field("data_cache_v2_5/performance_data_store", description="Directory for performance data storage.")

class EnhancedFlowMetricSettings(BaseModel):
    vapi_fa_params: Dict[str, Any] = Field(default_factory=dict)
    acceleration_calculation_intervals: List[str] = Field(default_factory=list)
    dwfd_params: Dict[str, Any] = Field(default_factory=dict)
    tw_laf_params: Dict[str, Any] = Field(default_factory=dict)
    # Isolated configuration parameters
    z_score_window: int = Field(20, description="Window size for Z-score calculations in enhanced flow metrics", ge=5, le=200)
    time_intervals: List[int] = Field(default_factory=lambda: [5, 15, 30], description="Time intervals for TW-LAF calculations")
    liquidity_weight: float = Field(0.3, description="Liquidity weight for TW-LAF calculations", ge=0.0, le=1.0)
    divergence_threshold: float = Field(1.5, description="Threshold for DWFD divergence detection", ge=0.1, le=10.0)
    lookback_periods: List[int] = Field(default_factory=lambda: [5, 10, 20], description="Lookback periods for VAPI-FA calculations")

    model_config = ConfigDict(extra='forbid')

class StrategySettings(BaseModel):
    model_config = ConfigDict(extra='allow')

class LearningParams(BaseModel):
    performance_tracker_query_lookback: int = Field(90, description="Number of days of historical performance data to consider for learning.", ge=1)
    learning_rate_for_signal_weights: float = Field(0.05, description="How aggressively ATIF adjusts signal weights based on new performance data (0-1 scale).", ge=0, le=1)
    learning_rate_for_target_adjustments: float = Field(0.02, description="How aggressively ATIF adjusts target parameters based on performance data (0-1 scale).", ge=0, le=1)
    min_trades_for_statistical_significance: int = Field(20, description="Minimum number of trades required for statistical significance in learning adjustments.", ge=1)

    model_config = ConfigDict(extra='forbid')

class AdaptiveTradeIdeaFrameworkSettings(BaseModel):
    min_conviction_to_initiate_trade: float = Field(2.5, description="The minimum ATIF conviction score (0-5 scale) required to generate a new trade recommendation.", ge=0, le=5)
    signal_integration_params: Dict[str, Any] = Field(default_factory=dict)
    regime_context_weight_multipliers: Dict[str, float] = Field(default_factory=dict)
    conviction_mapping_params: Dict[str, Any] = Field(default_factory=dict)
    strategy_specificity_rules: List[Dict[str, Any]] = Field(default_factory=list)
    intelligent_recommendation_management_rules: Dict[str, Any] = Field(default_factory=dict)
    learning_params: LearningParams = Field(default_factory=lambda: LearningParams(
        performance_tracker_query_lookback=90,
        learning_rate_for_signal_weights=0.05,
        learning_rate_for_target_adjustments=0.02,
        min_trades_for_statistical_significance=20
    ))

    model_config = ConfigDict(extra='forbid')

class AdvancedOptionsMetricsSettings(BaseModel):
    """Configuration for advanced options metrics calculation."""
    enabled: bool = True
    min_contracts_for_calculation: int = 5
    min_bid_ask_size: float = 1.0
    max_spread_percentage: float = 50.0
    default_implied_volatility: float = 0.20
    lwpai_settings: Dict[str, Any] = Field(default_factory=lambda: {
        "weight_threshold": 0.1,
        "outlier_filter_enabled": True,
        "outlier_threshold_std": 2.0
    })
    vabai_settings: Dict[str, Any] = Field(default_factory=lambda: {
        "volatility_floor": 0.05,
        "volatility_ceiling": 2.0,
        "imbalance_threshold": 0.1
    })
    aofm_settings: Dict[str, Any] = Field(default_factory=lambda: {
        "momentum_window": 5,
        "smoothing_factor": 0.3,
        "significant_change_threshold": 1000.0
    })
    lidb_settings: Dict[str, Any] = Field(default_factory=lambda: {
        "neutral_zone": 0.05,
        "strong_bias_threshold": 0.3,
        "extreme_bias_threshold": 0.45
    })
    confidence_scoring: Dict[str, Any] = Field(default_factory=lambda: {
        "min_valid_contracts": 10,
        "data_quality_weight": 0.4,
        "spread_quality_weight": 0.3,
        "volume_quality_weight": 0.3
    })

    model_config = ConfigDict(extra='forbid')

class TickerContextAnalyzerSettings(BaseModel):
    lookback_days: int = 252  # 1 year
    correlation_window: int = 60
    volatility_windows: List[int] = Field(default_factory=lambda: [1, 5, 20])
    volume_threshold: int = 1000000
    use_yahoo_finance: bool = False
    yahoo_finance_rate_limit_seconds: float = 2.0
    SPY: Dict[str, Any] = Field(default_factory=dict)
    DEFAULT_TICKER_PROFILE: Dict[str, Any] = Field(default_factory=dict)
    advanced_options_metrics: AdvancedOptionsMetricsSettings = Field(default_factory=lambda: AdvancedOptionsMetricsSettings())

    model_config = ConfigDict(extra='forbid')

class KeyLevelIdentifierSettings(BaseModel):
    lookback_periods: int = 20
    min_touches: int = 2
    level_tolerance: float = 0.005  # 0.5%
    volume_threshold: float = 1.5
    oi_threshold: int = 1000
    gamma_threshold: float = 0.1
    nvp_support_quantile: float = 0.95
    nvp_resistance_quantile: float = 0.95

    model_config = ConfigDict(extra='forbid')

class HeatmapGenerationSettings(BaseModel):
    ugch_params: Dict[str, Any] = Field(default_factory=dict)
    sgdhp_params: Dict[str, Any] = Field(default_factory=dict)
    # Isolated configuration parameters
    flow_normalization_window: int = Field(100, description="Window size for flow normalization in enhanced heatmap calculations", ge=10, le=500)

    model_config = ConfigDict(extra='forbid')

class PerformanceTrackerSettingsV2_5(BaseModel):
    performance_data_directory: str = Field(..., description="Directory for storing performance tracking data.")
    historical_window_days: int = Field(365, description="Number of days to consider for historical performance.", ge=1)
    weight_smoothing_factor: float = Field(0.1, description="Smoothing factor for performance weights (0-1 scale).", ge=0, le=1)
    min_sample_size: int = Field(10, description="Minimum number of samples required for reliable performance metrics.", ge=1)
    confidence_threshold: float = Field(0.75, description="Minimum confidence level required for performance-based adjustments.", ge=0, le=1)
    update_interval_seconds: int = Field(3600, description="Interval in seconds between performance data updates.", ge=1)
    tracking_enabled: bool = Field(True, description="Whether performance tracking is enabled.")
    metrics_to_track: List[str] = Field(default_factory=lambda: ["returns", "sharpe_ratio", "max_drawdown", "win_rate"], description="List of metrics to track.")
    reporting_frequency: str = Field("daily", description="Frequency of performance reporting.")
    benchmark_symbol: str = Field("SPY", description="Symbol to use as a benchmark for performance comparison.")

    model_config = ConfigDict(extra='forbid')

class AdaptiveMetricParameters(BaseModel):
    a_mspi_settings: Dict[str, Any] = Field(default_factory=dict)
    a_dag_settings: Dict[str, Any] = Field(default_factory=dict)
    e_sdag_settings: Dict[str, Any] = Field(default_factory=dict)
    d_tdpi_settings: Dict[str, Any] = Field(default_factory=dict)
    vri_2_0_settings: Dict[str, Any] = Field(default_factory=dict)
    # Isolated configuration parameters for enhanced heatmap
    enhanced_heatmap_settings: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra='forbid')

class DagAlphaCoeffs(BaseModel):
    aligned: float = Field(default=1.35, description="Coefficient for aligned market conditions")
    opposed: float = Field(default=0.65, description="Coefficient for opposed market conditions")
    neutral: float = Field(default=1.0, description="Coefficient for neutral market conditions")

    model_config = ConfigDict(extra='forbid')

class TdpiBetaCoeffs(BaseModel):
    aligned: float = Field(default=1.35, description="Coefficient for aligned market conditions")
    opposed: float = Field(default=0.65, description="Coefficient for opposed market conditions")
    neutral: float = Field(default=1.0, description="Coefficient for neutral market conditions")

    model_config = ConfigDict(extra='forbid')

class VriGammaCoeffs(BaseModel):
    aligned: float = Field(default=1.35, description="Coefficient for aligned market conditions")
    opposed: float = Field(default=0.65, description="Coefficient for opposed market conditions")
    neutral: float = Field(default=1.0, description="Coefficient for neutral market conditions")

    model_config = ConfigDict(extra='forbid')

class CoefficientsSettings(BaseModel):
    dag_alpha: DagAlphaCoeffs = Field(default_factory=DagAlphaCoeffs)
    tdpi_beta: TdpiBetaCoeffs = Field(default_factory=TdpiBetaCoeffs)
    vri_gamma: VriGammaCoeffs = Field(default_factory=VriGammaCoeffs)

    model_config = ConfigDict(extra='forbid')

class DataProcessorSettings(BaseModel):
    factors: Dict[str, Any] = Field(default_factory=dict)
    coefficients: CoefficientsSettings = Field(default_factory=CoefficientsSettings)
    iv_context_parameters: Dict[str, Any] = Field(default_factory=dict)
    vci_0dte_parameters: Dict[str, Any] = Field(default_factory=lambda: {
        "high_concentration_threshold": 0.6,
        "extreme_concentration_threshold": 0.8,
        "min_vanna_oi_threshold": 1000.0,
        "top_strikes_percentile": 0.8
    })

    model_config = ConfigDict(extra='forbid')  # Strict validation - no extra fields allowed

class VisualizationSettings(BaseModel):
    dashboard_refresh_interval_seconds: int = Field(60, description="Interval in seconds between dashboard refreshes")
    max_table_rows_signals_insights: int = Field(10, description="Maximum number of rows to display in signals and insights tables")
    dashboard: Dict[str, Any] = Field(default_factory=lambda: {
        "host": "localhost",
        "port": 8050,
        "debug": False,
        "auto_refresh_seconds": 30,
        "timestamp_format": "%Y-%m-%d %H:%M:%S %Z",
        "defaults": {
            "symbol": "SPX",
            "refresh_interval_seconds": 30
        },
        "modes_detail_config": {
            "main": {
                "label": "Main Dashboard",
                "module_name": "main_dashboard_display_v2_5",
                "charts": ["regime_display", "flow_gauges", "gib_gauges", "recommendations_table"]
            },
            "flow": {
                "label": "Flow Analysis",
                "module_name": "flow_mode_display_v2_5",
                "charts": ["net_value_heatmap_viz", "net_cust_delta_flow_viz", "net_cust_gamma_flow_viz", "net_cust_vega_flow_viz"]
            },
            "structure": {
                "label": "Structure & Positioning",
                "module_name": "structure_mode_display_v2_5",
                "charts": ["mspi_components", "sai_ssi_displays"]
            },
            "timedecay": {
                "label": "Time Decay & Pinning",
                "module_name": "time_decay_mode_display_v2_5",
                "charts": ["tdpi_displays", "vci_strike_charts"]
            },
            "advanced": {
                "label": "Advanced Flow Metrics",
                "module_name": "advanced_flow_mode_v2_5",
                "charts": ["vapi_gauges", "dwfd_gauges", "tw_laf_gauges"]
            }
        },
        "main_dashboard_settings": {
            "regime_indicator": {
                "title": "Market Regime",
                "regime_colors": {
                    "default": "secondary",
                    "bullish": "success",
                    "bearish": "danger",
                    "neutral": "info",
                    "unclear": "warning"
                }
            },
            "flow_gauge": {
                "height": 200,
                "indicator_font_size": 16,
                "number_font_size": 24,
                "axis_range": [-3, 3],
                "threshold_line_color": "white",
                "margin": {"t": 60, "b": 40, "l": 20, "r": 20},
                "steps": [
                    {"range": [-3, -2], "color": "#d62728"},
                    {"range": [-2, -0.5], "color": "#ff9896"},
                    {"range": [-0.5, 0.5], "color": "#aec7e8"},
                    {"range": [0.5, 2], "color": "#98df8a"},
                    {"range": [2, 3], "color": "#2ca02c"}
                ]
            },
            "gib_gauge": {
                "height": 180,
                "indicator_font_size": 14,
                "number_font_size": 20,
                "axis_range": [-1, 1],
                "dollar_axis_range": [-1000000, 1000000],
                "threshold_line_color": "white",
                "margin": {"t": 50, "b": 30, "l": 15, "r": 15},
                "steps": [
                    {"range": [-1, -0.5], "color": "#d62728"},
                    {"range": [-0.5, -0.1], "color": "#ff9896"},
                    {"range": [-0.1, 0.1], "color": "#aec7e8"},
                    {"range": [0.1, 0.5], "color": "#98df8a"},
                    {"range": [0.5, 1], "color": "#2ca02c"}
                ],
                "dollar_steps": [
                    {"range": [-1000000, -500000], "color": "#d62728"},
                    {"range": [-500000, -100000], "color": "#ff9896"},
                    {"range": [-100000, 100000], "color": "#aec7e8"},
                    {"range": [100000, 500000], "color": "#98df8a"},
                    {"range": [500000, 1000000], "color": "#2ca02c"}
                ]
            },
            "mini_heatmap": {
                "height": 150,
                "colorscale": "RdYlGn",
                "margin": {"t": 50, "b": 30, "l": 40, "r": 40}
            },
            "recommendations_table": {
                "title": "ATIF Recommendations",
                "max_rationale_length": 50,
                "page_size": 5,
                "style_cell": {"textAlign": "left", "padding": "5px", "minWidth": "80px", "width": "auto", "maxWidth": "200px"},
                "style_header": {"backgroundColor": "rgb(30, 30, 30)", "fontWeight": "bold", "color": "white"},
                "style_data": {"backgroundColor": "rgb(50, 50, 50)", "color": "white"}
            },
            "ticker_context": {
                "title": "Ticker Context"
            }
        }
    })

    model_config = ConfigDict(extra='forbid')

class SymbolDefaultOverridesStrategySettingsTargets(BaseModel):
    target_atr_stop_loss_multiplier: float = 1.5

    model_config = ConfigDict(extra='forbid')

class SymbolDefaultOverridesStrategySettings(BaseModel):
    targets: SymbolDefaultOverridesStrategySettingsTargets = Field(default_factory=SymbolDefaultOverridesStrategySettingsTargets)

    model_config = ConfigDict(extra='forbid')

class SymbolDefaultOverrides(BaseModel):
    strategy_settings: Optional[SymbolDefaultOverridesStrategySettings] = Field(default_factory=SymbolDefaultOverridesStrategySettings)

    model_config = ConfigDict(extra='forbid')

class SymbolSpecificOverrides(BaseModel):
    DEFAULT: Optional[SymbolDefaultOverrides] = Field(default_factory=SymbolDefaultOverrides)
    SPY: Optional[Dict[str, Any]] = Field(default_factory=dict)
    AAPL: Optional[Dict[str, Any]] = Field(default_factory=dict)

    model_config = ConfigDict(extra='forbid')



class DatabaseSettings(BaseModel):
    host: str = Field(..., description="Database host address")
    port: int = Field(5432, description="Database port number")
    database: str = Field(..., description="Database name")
    user: str = Field(..., description="Database username")
    password: str = Field(..., description="Database password")
    min_connections: int = Field(1, description="Minimum number of database connections")
    max_connections: int = Field(10, description="Maximum number of database connections")

    model_config = ConfigDict(extra='forbid')

# ===== AI PERFORMANCE TRACKER MODELS =====

class AIPerformanceDataV2_5(BaseModel):
    """Comprehensive Pydantic model for AI Performance Tracker dashboard data."""
    # Time series data
    dates: List[str] = Field(default_factory=list, description="List of dates for performance tracking")
    accuracy: List[float] = Field(default_factory=list, description="Daily accuracy percentages")
    confidence: List[float] = Field(default_factory=list, description="Daily confidence scores")
    learning_curve: List[float] = Field(default_factory=list, description="Cumulative learning improvement")

    # Summary statistics
    total_predictions: int = Field(0, description="Total number of predictions made", ge=0)
    successful_predictions: int = Field(0, description="Number of successful predictions", ge=0)
    success_rate: float = Field(0.0, description="Overall success rate", ge=0.0, le=1.0)
    avg_confidence: float = Field(0.0, description="Average confidence score", ge=0.0, le=1.0)
    improvement_rate: float = Field(0.0, description="Rate of improvement over time")
    learning_score: float = Field(0.0, description="Overall learning effectiveness score", ge=0.0, le=10.0)

    # Data source and metadata
    data_source: str = Field("Unknown", description="Source of performance data")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    symbol: str = Field("SPY", description="Primary symbol for performance tracking")

    # Enhanced intelligence features
    diabolical_intelligence_active: bool = Field(False, description="Whether diabolical intelligence is active")
    sentiment_regime: str = Field("NEUTRAL", description="Current sentiment regime")
    intelligence_confidence: str = Field("50.0%", description="Intelligence confidence level")
    diabolical_insight: str = Field("ðŸ˜ˆ Apex predator analyzing...", description="Diabolical AI insight")

    model_config = ConfigDict(extra='allow')

class AILearningStatsV2_5(BaseModel):
    """Pydantic model for AI learning statistics in 6-metric comprehensive format."""
    patterns_learned: int = Field(247, description="Number of patterns learned", ge=0)
    success_rate: float = Field(0.73, description="Learning success rate", ge=0.0, le=1.0)
    adaptation_score: float = Field(8.4, description="Adaptation effectiveness score", ge=0.0, le=10.0)
    memory_nodes: int = Field(1432, description="Number of active memory nodes", ge=0)
    active_connections: int = Field(3847, description="Number of active neural connections", ge=0)
    learning_velocity: float = Field(0.85, description="Rate of learning velocity", ge=0.0, le=1.0)

    # Metadata
    data_source: str = Field("Database", description="Source of learning statistics")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")

    model_config = ConfigDict(extra='allow')

class LearningStatusV2_5(BaseModel):
    """Pydantic model for comprehensive learning system status tracking."""
    system_status: str = Field(default="ACTIVE", description="Current learning system status")
    last_learning_cycle: datetime = Field(default_factory=datetime.now, description="Timestamp of last learning cycle")
    active_patterns: int = Field(default=0, description="Number of active learning patterns", ge=0)
    success_rate: float = Field(default=0.0, description="Current learning success rate", ge=0.0, le=1.0)
    learning_cycles_completed: int = Field(default=0, description="Total learning cycles completed", ge=0)
    parameters_optimized: int = Field(default=0, description="Number of parameters optimized", ge=0)
    confidence_score: float = Field(default=0.0, description="Learning confidence score", ge=0.0, le=1.0)
    next_scheduled_learning: Optional[datetime] = Field(default=None, description="Next scheduled learning cycle")
    performance_trend: str = Field(default="STABLE", description="Performance trend indicator")
    learning_insights: List[str] = Field(default_factory=list, description="Recent learning insights")

    model_config = ConfigDict(extra='allow')

class LearningMetricsV2_5(BaseModel):
    """Pydantic model for detailed learning metrics and analytics."""
    total_patterns_discovered: int = Field(default=0, description="Total patterns discovered", ge=0)
    patterns_validated: int = Field(default=0, description="Patterns successfully validated", ge=0)
    patterns_rejected: int = Field(default=0, description="Patterns rejected during validation", ge=0)
    learning_accuracy: float = Field(default=0.0, description="Overall learning accuracy", ge=0.0, le=1.0)
    adaptation_velocity: float = Field(default=0.0, description="Speed of adaptation to new patterns", ge=0.0, le=1.0)
    memory_efficiency: float = Field(default=0.0, description="Memory usage efficiency", ge=0.0, le=1.0)
    prediction_confidence: float = Field(default=0.0, description="Average prediction confidence", ge=0.0, le=1.0)
    last_learning_session: datetime = Field(default_factory=datetime.now, description="Last learning session timestamp")

    model_config = ConfigDict(extra='allow')

class AdaptiveLearningConfigV2_5(BaseModel):
    """Pydantic model for adaptive learning system configuration."""
    learning_enabled: bool = Field(True, description="Enable adaptive learning")
    learning_rate: float = Field(0.01, description="Base learning rate", ge=0.0, le=1.0)
    pattern_discovery_threshold: float = Field(0.7, description="Threshold for pattern discovery", ge=0.0, le=1.0)
    validation_window_days: int = Field(30, description="Days for pattern validation", ge=1)
    max_patterns_per_symbol: int = Field(100, description="Maximum patterns per symbol", ge=1)
    learning_schedule: str = Field("daily", description="Learning schedule frequency")
    auto_adaptation: bool = Field(True, description="Enable automatic adaptation")

    model_config = ConfigDict(extra='allow')

class AIMCPStatusV2_5(BaseModel):
    """Pydantic model for MCP (Model Context Protocol) server status."""
    memory_server_active: bool = Field(False, description="Memory MCP server status")
    sequential_thinking_active: bool = Field(False, description="Sequential thinking MCP server status")
    exa_search_active: bool = Field(False, description="Exa search MCP server status")
    context7_active: bool = Field(False, description="Context7 MCP server status")

    # Connection details
    total_servers: int = Field(4, description="Total number of MCP servers", ge=0)
    active_servers: int = Field(0, description="Number of active MCP servers", ge=0)
    connection_health: float = Field(0.0, description="Overall connection health", ge=0.0, le=1.0)

    # Status messages
    status_message: str = Field("MCP servers initializing...", description="Current status message")
    last_check: datetime = Field(default_factory=datetime.now, description="Last status check timestamp")

    model_config = ConfigDict(extra='allow')

class AISystemHealthV2_5(BaseModel):
    """Comprehensive Pydantic model for AI system health monitoring."""
    # Database connectivity
    database_connected: bool = Field(default=False, description="Database connection status")
    ai_tables_available: bool = Field(default=False, description="AI tables availability status")

    # Component health
    predictions_manager_healthy: bool = Field(default=False, description="AI Predictions Manager health")
    learning_system_healthy: bool = Field(default=False, description="AI Learning System health")
    adaptation_engine_healthy: bool = Field(default=False, description="AI Adaptation Engine health")

    # Performance metrics
    overall_health_score: float = Field(default=0.0, description="Overall system health score", ge=0.0, le=1.0)
    response_time_ms: float = Field(default=0.0, description="Average response time in milliseconds", ge=0.0)
    error_rate: float = Field(default=0.0, description="System error rate", ge=0.0, le=1.0)

    # Status details
    status_message: str = Field(default="System initializing...", description="Current system status message")
    last_health_check: datetime = Field(default_factory=datetime.now, description="Last health check timestamp")

    # Detailed component status
    component_status: Dict[str, Any] = Field(default_factory=dict, description="Detailed component status information")

    model_config = ConfigDict(extra='allow')

# ===== CONSOLIDATED SETTINGS MODELS =====
# Note: Settings models consolidated into logical groups for better organization

class DataManagementConfigV2_5(BaseModel):
    """Consolidated data management configuration - replaces DataFetcherSettings + DataManagementSettings."""

    # ConvexValue Authentication
    convexvalue_auth: ConvexValueAuthSettings = Field(default_factory=lambda: ConvexValueAuthSettings(
        use_env_variables=True,
        auth_method="email_password"
    ))

    # API Configuration
    tradier_api_key: str = Field(..., description="API Key for Tradier")
    tradier_account_id: str = Field(..., description="Account ID for Tradier")
    alpha_vantage_api_key: str = Field(default="", description="API Key for Alpha Vantage")

    # Retry and Timeout Configuration
    retry_attempts: int = Field(default=3, description="Maximum number of retry attempts", ge=0)
    retry_delay: float = Field(default=5.0, description="Base delay between retries", ge=0)
    timeout: float = Field(default=30.0, description="Request timeout in seconds", ge=0)

    # Data Storage Configuration
    cache_directory: str = Field(default="data_cache_v2_5", description="Directory for caching data")
    data_store_directory: str = Field(default="data_cache_v2_5/data_store", description="Data storage directory")
    cache_expiry_hours: float = Field(default=24.0, description="Cache expiry time in hours", ge=0)

    model_config = ConfigDict(extra='forbid')

class AnalyticsEngineConfigV2_5(BaseModel):
    """Consolidated analytics configuration - replaces EnhancedFlowMetricSettings + AdaptiveMetricParameters + HeatmapGenerationSettings."""

    # Enhanced Flow Metrics
    z_score_window: int = Field(default=20, description="Window size for Z-score calculations", ge=5, le=200)
    time_intervals: List[int] = Field(default_factory=lambda: [5, 15, 30], description="Time intervals for TW-LAF calculations")
    liquidity_weight: float = Field(default=0.3, description="Liquidity weight for TW-LAF", ge=0.0, le=1.0)
    divergence_threshold: float = Field(default=1.5, description="Threshold for DWFD divergence detection", ge=0.1, le=10.0)
    lookback_periods: List[int] = Field(default_factory=lambda: [5, 10, 20], description="Lookback periods for VAPI-FA")

    # Heatmap Generation
    flow_normalization_window: int = Field(default=100, description="Window size for flow normalization", ge=10, le=500)

    # Adaptive Metric Parameters
    vapi_fa_params: Dict[str, Any] = Field(default_factory=dict)
    acceleration_calculation_intervals: List[str] = Field(default_factory=list)
    dwfd_params: Dict[str, Any] = Field(default_factory=dict)
    tw_laf_params: Dict[str, Any] = Field(default_factory=dict)
    ugch_params: Dict[str, Any] = Field(default_factory=dict)
    sgdhp_params: Dict[str, Any] = Field(default_factory=dict)
    a_mspi_settings: Dict[str, Any] = Field(default_factory=dict)
    a_dag_settings: Dict[str, Any] = Field(default_factory=dict)
    e_sdag_settings: Dict[str, Any] = Field(default_factory=dict)
    d_tdpi_settings: Dict[str, Any] = Field(default_factory=dict)
    vri_2_0_settings: Dict[str, Any] = Field(default_factory=dict)
    enhanced_heatmap_settings: Dict[str, Any] = Field(default_factory=dict)

    # Learning Configuration
    learning_enabled: bool = Field(default=True, description="Enable learning features")
    learning_rate: float = Field(default=0.01, description="Learning rate for adaptive algorithms", ge=0.0001, le=1.0)
    pattern_discovery_threshold: float = Field(default=0.75, description="Threshold for pattern discovery", ge=0.0, le=1.0)
    validation_window_days: int = Field(default=30, description="Validation window in days", ge=1, le=365)
    max_patterns_per_symbol: int = Field(default=100, description="Maximum patterns to track per symbol", ge=1, le=1000)
    learning_schedule: str = Field(default="continuous", description="Learning schedule type")
    auto_adaptation: bool = Field(default=True, description="Enable automatic parameter adaptation")

    model_config = ConfigDict(extra='forbid')

class AIConfigV2_5(BaseModel):
    """Consolidated AI configuration - replaces AIPredictionsSettings + AIPerformanceTrackerSettings."""

    # AI Predictions Configuration
    predictions_enabled: bool = Field(default=True, description="Enable AI predictions generation")
    auto_create_predictions: bool = Field(default=True, description="Automatically create predictions during analysis")
    default_time_horizon: str = Field(default="4H", description="Default prediction time horizon")
    min_confidence_threshold: float = Field(default=0.5, description="Minimum confidence to create prediction", ge=0.0, le=1.0)
    max_predictions_per_symbol: int = Field(default=10, description="Maximum active predictions per symbol", ge=1)
    evaluation_frequency_minutes: int = Field(default=60, description="How often to evaluate predictions (minutes)", ge=1)
    auto_evaluation_enabled: bool = Field(default=True, description="Automatically evaluate predictions when target time reached")
    performance_tracking_days: int = Field(default=30, description="Days to track for performance metrics", ge=1)
    prediction_types: List[str] = Field(default_factory=lambda: [
        "eots_direction", "price_target", "volatility_forecast", "regime_transition"
    ], description="Supported prediction types")
    confidence_calibration: Dict[str, float] = Field(default_factory=lambda: {
        "strong_signal_threshold": 3.0,
        "moderate_signal_threshold": 1.5,
        "weak_signal_threshold": 0.5
    }, description="Confidence calibration thresholds")

    # Performance Tracker Configuration
    tracker_title: str = Field(default="ðŸ" " AI Performance Tracker", description="Display title for performance tracker")
    tracker_height: int = Field(default=250, description="Chart height in pixels", ge=100)
    lookback_days: int = Field(default=30, description="Number of days to look back for performance data", ge=1)
    show_learning_curve: bool = Field(default=True, description="Whether to show learning curve visualization")
    refresh_interval: int = Field(default=30, description="Refresh interval in seconds", ge=5)
    max_data_points: int = Field(default=100, description="Maximum data points to display", ge=10)
    excellent_threshold: float = Field(default=0.85, description="Threshold for excellent performance", ge=0.0, le=1.0)
    good_threshold: float = Field(default=0.70, description="Threshold for good performance", ge=0.0, le=1.0)
    poor_threshold: float = Field(default=0.50, description="Threshold for poor performance", ge=0.0, le=1.0)
    show_confidence_bands: bool = Field(default=True, description="Show confidence bands on charts")
    show_trend_analysis: bool = Field(default=True, description="Show trend analysis")
    enable_real_time_updates: bool = Field(default=True, description="Enable real-time performance updates")

    model_config = ConfigDict(extra='forbid')

class TradingConfigV2_5(BaseModel):
    """Consolidated trading configuration - replaces AdaptiveTradeIdeaFrameworkSettings + StrategySettings."""

    # ATIF Configuration
    min_conviction_to_initiate_trade: float = Field(default=2.5, description="Minimum ATIF conviction score (0-5 scale)", ge=0, le=5)
    signal_integration_params: Dict[str, Any] = Field(default_factory=dict)
    regime_context_weight_multipliers: Dict[str, float] = Field(default_factory=dict)
    conviction_mapping_params: Dict[str, Any] = Field(default_factory=dict)
    strategy_specificity_rules: List[Dict[str, Any]] = Field(default_factory=list)
    intelligent_recommendation_management_rules: Dict[str, Any] = Field(default_factory=dict)

    # Learning Parameters
    performance_tracker_query_lookback: int = Field(default=90, description="Days of historical performance data for learning", ge=1)
    learning_rate_for_signal_weights: float = Field(default=0.05, description="Signal weights learning rate (0-1 scale)", ge=0, le=1)
    learning_rate_for_target_adjustments: float = Field(default=0.02, description="Target parameters learning rate (0-1 scale)", ge=0, le=1)
    min_trades_for_statistical_significance: int = Field(default=20, description="Minimum trades for statistical significance", ge=1)

    # Strategy Settings (extensible)
    strategy_params: Dict[str, Any] = Field(default_factory=dict, description="Additional strategy parameters")

    model_config = ConfigDict(extra='allow')  # Allow additional strategy parameters

class DashboardConfigV2_5(BaseModel):
    """Consolidated dashboard configuration - replaces VisualizationSettings + DashboardModeSettings + MainDashboardDisplaySettings."""

    # Core Dashboard Settings
    refresh_interval_seconds: int = Field(default=30, description="Interval between dashboard refreshes")
    host: str = Field(default="localhost", description="Dashboard host")
    port: int = Field(default=8050, description="Dashboard port")
    debug: bool = Field(default=False, description="Debug mode")
    timestamp_format: str = Field(default="%Y-%m-%d %H:%M:%S %Z", description="Timestamp format")
    max_table_rows_signals_insights: int = Field(default=10, description="Maximum table rows for signals/insights")

    # Dashboard Modes
    modes_detail_config: Dict[str, DashboardModeSettings] = Field(default_factory=lambda: {
        "main": DashboardModeSettings(
            label="Main Dashboard",
            module_name="main_dashboard_display_v2_5",
            charts=["regime_display", "flow_gauges", "gib_gauges", "recommendations_table"]
        ),
        "flow": DashboardModeSettings(
            label="Flow Analysis",
            module_name="flow_mode_display_v2_5",
            charts=["net_value_heatmap_viz", "net_cust_delta_flow_viz", "net_cust_gamma_flow_viz", "net_cust_vega_flow_viz"]
        ),
        "structure": DashboardModeSettings(
            label="Structure & Positioning",
            module_name="structure_mode_display_v2_5",
            charts=["mspi_components", "sai_ssi_displays"]
        ),
        "timedecay": DashboardModeSettings(
            label="Time Decay & Pinning",
            module_name="time_decay_mode_display_v2_5",
            charts=["tdpi_displays", "vci_strike_charts"]
        ),
        "advanced": DashboardModeSettings(
            label="Advanced Flow Metrics",
            module_name="advanced_flow_mode_v2_5",
            charts=["vapi_gauges", "dwfd_gauges", "tw_laf_gauges"]
        ),
        "volatility": DashboardModeSettings(
            label="Volatility Deep Dive",
            module_name="volatility_mode_display_v2_5",
            charts=["vri_2_0_strike_profile", "volatility_gauges", "volatility_surface_heatmap"]
        ),
        "ai": DashboardModeSettings(
            label="AI Intelligence Hub",
            module_name="ai_dashboard.ai_dashboard_display_v2_5",
            charts=["ai_market_analysis", "ai_recommendations", "ai_insights", "ai_regime_context", "ai_performance_tracker"]
        )
    })

    # Main Dashboard Display Settings
    regime_indicator: Dict[str, Any] = Field(default_factory=lambda: {
        "title": "Market Regime",
        "regime_colors": {
            "default": "secondary",
            "bullish": "success",
            "bearish": "danger",
            "neutral": "info",
            "unclear": "warning"
        }
    })
    flow_gauge: Dict[str, Any] = Field(default_factory=lambda: {
        "height": 200,
        "indicator_font_size": 16,
        "number_font_size": 24,
        "axis_range": [-3, 3],
        "threshold_line_color": "white",
        "margin": {"t": 60, "b": 40, "l": 20, "r": 20},
        "steps": [
            {"range": [-3, -2], "color": "#d62728"},
            {"range": [-2, -0.5], "color": "#ff9896"},
            {"range": [-0.5, 0.5], "color": "#aec7e8"},
            {"range": [0.5, 2], "color": "#98df8a"},
            {"range": [2, 3], "color": "#2ca02c"}
        ]
    })
    gib_gauge: Dict[str, Any] = Field(default_factory=lambda: {
        "height": 180,
        "indicator_font_size": 14,
        "number_font_size": 20,
        "axis_range": [-1, 1],
        "dollar_axis_range": [-1000000, 1000000],
        "threshold_line_color": "white",
        "margin": {"t": 50, "b": 30, "l": 15, "r": 15},
        "steps": [
            {"range": [-1, -0.5], "color": "#d62728"},
            {"range": [-0.5, -0.1], "color": "#ff9896"},
            {"range": [-0.1, 0.1], "color": "#aec7e8"},
            {"range": [0.1, 0.5], "color": "#98df8a"},
            {"range": [0.5, 1], "color": "#2ca02c"}
        ],
        "dollar_steps": [
            {"range": [-1000000, -500000], "color": "#d62728"},
            {"range": [-500000, -100000], "color": "#ff9896"},
            {"range": [-100000, 100000], "color": "#aec7e8"},
            {"range": [100000, 500000], "color": "#98df8a"},
            {"range": [500000, 1000000], "color": "#2ca02c"}
        ]
    })
    mini_heatmap: Dict[str, Any] = Field(default_factory=lambda: {
        "height": 150,
        "colorscale": "RdYlGn",
        "margin": {"t": 50, "b": 30, "l": 40, "r": 40}
    })
    recommendations_table: Dict[str, Any] = Field(default_factory=lambda: {
        "title": "ATIF Recommendations",
        "max_rationale_length": 50,
        "page_size": 5,
        "style_cell": {"textAlign": "left", "padding": "5px", "minWidth": "80px", "width": "auto", "maxWidth": "200px"},
        "style_header": {"backgroundColor": "rgb(30, 30, 30)", "fontWeight": "bold", "color": "white"},
        "style_data": {"backgroundColor": "rgb(50, 50, 50)", "color": "white"}
    })
    ticker_context: Dict[str, Any] = Field(default_factory=lambda: {
        "title": "Ticker Context"
    })

    model_config = ConfigDict(extra='forbid')

class UnifiedPerformanceMetricsV2_5(BaseModel):
    """Consolidated performance tracking for all EOTS components - replaces 7+ performance models."""

    # Core Performance Metrics (using existing models)
    ai_predictions_performance: Dict[str, Any] = Field(default_factory=lambda: {
        "total_predictions": 0,
        "correct_predictions": 0,
        "incorrect_predictions": 0,
        "pending_predictions": 0,
        "success_rate": 0.0,
        "avg_confidence": 0.0,
        "performance_trend": "STABLE"
    }, description="AI predictions performance metrics")

    ai_adaptations_performance: Dict[str, Any] = Field(default_factory=lambda: {
        "total_applications": 0,
        "successful_applications": 0,
        "success_rate": 0.0,
        "avg_improvement": 0.0,
        "adaptation_score": 0.0,
        "performance_trend": "STABLE"
    }, description="AI adaptations performance metrics")

    huihui_experts_performance: Dict[str, Any] = Field(default_factory=lambda: {
        "total_requests": 0,
        "successful_requests": 0,
        "average_processing_time": 0.0,
        "success_rate": 0.0,
        "expert_efficiency": 0.0
    }, description="HuiHui experts performance metrics")

    orchestrator_performance: Dict[str, Any] = Field(default_factory=lambda: {
        "total_decisions": 0,
        "successful_decisions": 0,
        "avg_execution_time_ms": 0.0,
        "success_rate": 0.0,
        "resource_efficiency": 0.0,
        "learning_impact": 0.0
    }, description="Orchestrator performance metrics")

    system_health: Dict[str, Any] = Field(default_factory=lambda: {
        "overall_status": "operational",
        "component_statuses": {},
        "uptime_hours": 0.0,
        "error_rate": 0.0,
        "memory_usage_mb": 0.0,
        "cpu_usage_percent": 0.0
    }, description="System health metrics")

    learning_statistics: Dict[str, Any] = Field(default_factory=lambda: {
        "total_learning_cycles": 0,
        "successful_adaptations": 0,
        "learning_efficiency": 0.0,
        "knowledge_retention_rate": 0.0,
        "improvement_rate": 0.0
    }, description="Learning and adaptation statistics")

    # Performance Summary
    overall_performance_score: float = Field(default=0.0, description="Overall system performance score (0.0-1.0)", ge=0.0, le=1.0)
    performance_trend: str = Field(default="STABLE", description="Performance trend direction")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last performance update timestamp")

    model_config = ConfigDict(extra='forbid')

# ===== AI PREDICTIONS SETTINGS =====

class AIPerformanceTrackerSettings(BaseModel):
    """Configuration settings for AI Performance Tracker dashboard component."""
    title: str = Field(default="ðŸ" " AI Performance Tracker", description="Display title for performance tracker")
    height: int = Field(default=250, description="Chart height in pixels", ge=100)
    lookback_days: int = Field(default=30, description="Number of days to look back for performance data", ge=1)
    show_learning_curve: bool = Field(default=True, description="Whether to show learning curve visualization")
    refresh_interval: int = Field(default=30, description="Refresh interval in seconds", ge=5)
    max_data_points: int = Field(default=100, description="Maximum data points to display", ge=10)

    # Performance thresholds
    excellent_threshold: float = Field(default=0.85, description="Threshold for excellent performance", ge=0.0, le=1.0)
    good_threshold: float = Field(default=0.70, description="Threshold for good performance", ge=0.0, le=1.0)
    poor_threshold: float = Field(default=0.50, description="Threshold for poor performance", ge=0.0, le=1.0)

    # Display options
    show_confidence_bands: bool = Field(default=True, description="Show confidence bands on charts")
    show_trend_analysis: bool = Field(default=True, description="Show trend analysis")
    enable_real_time_updates: bool = Field(default=True, description="Enable real-time performance updates")

    model_config = ConfigDict(extra='allow')

class AIPredictionsSettings(BaseModel):
    """Configuration settings for AI predictions system."""
    enabled: bool = Field(default=True, description="Enable AI predictions generation")
    auto_create_predictions: bool = Field(default=True, description="Automatically create predictions during analysis")
    default_time_horizon: str = Field(default="4H", description="Default prediction time horizon")
    min_confidence_threshold: float = Field(default=0.5, description="Minimum confidence to create prediction", ge=0.0, le=1.0)
    max_predictions_per_symbol: int = Field(default=10, description="Maximum active predictions per symbol", ge=1)
    evaluation_frequency_minutes: int = Field(default=60, description="How often to evaluate predictions (minutes)", ge=1)
    auto_evaluation_enabled: bool = Field(default=True, description="Automatically evaluate predictions when target time reached")
    performance_tracking_days: int = Field(default=30, description="Days to track for performance metrics", ge=1)
    prediction_types: List[str] = Field(default_factory=lambda: [
        "eots_direction", "price_target", "volatility_forecast", "regime_transition"
    ], description="Supported prediction types")
    confidence_calibration: Dict[str, float] = Field(default_factory=lambda: {
        "strong_signal_threshold": 3.0,
        "moderate_signal_threshold": 1.5,
        "max_confidence": 0.85,
        "min_confidence": 0.5
    }, description="Confidence score calibration parameters")

    model_config = ConfigDict(extra='forbid')

class IntradayMetricDataV2_5(BaseModel):
    """ðŸš€ PYDANTIC-FIRST: Model for intraday metric data stored in Redis."""
    values: Union[List[float], Dict[str, Any], float] = Field(description="Metric values - can be scalar, list, or dict")
    last_updated: datetime = Field(description="Timestamp when data was last updated")
    sample_count: int = Field(description="Number of samples in the data")

    model_config = ConfigDict(extra='forbid')

class IntradayCollectorSettings(BaseModel):
    watched_tickers: List[str] = Field(default_factory=lambda: [
        "SPY", "SPX", "QQQ", "IWM", "VIX", "TSLA", "AMZN", "AAPL", "META", "MSFT", "NVDA"
    ], description="List of tickers to collect intraday metrics for.")
    metrics: List[str] = Field(default_factory=lambda: [
        "vapi_fa", "dwfd", "tw_laf",
        "vapifa_zscore_history", "dwfd_zscore_history", "twlaf_zscore_history",
        "rolling_flows", "nvp_by_strike", "nvp_vol_by_strike", "strikes",
        "greek_flows", "flow_ratios"
    ], description="Gauge and advanced analytics metrics to collect for intraday dashboard display.")
    cache_dir: str = Field(default="cache/intraday_metrics", description="Directory for intraday metric cache files.")
    collection_interval_seconds: int = Field(default=5, description="Interval in seconds between metric collections.")
    market_open_time: str = Field(default="09:30:00", description="Market open time (HH:MM:SS).")
    market_close_time: str = Field(default="16:00:00", description="Market close time (HH:MM:SS).")
    reset_at_eod: bool = Field(default=True, description="Whether to wipe cache at end of day.")
    model_config = ConfigDict(extra='forbid')

class EOTSConfigV2_5(BaseModel):
    """CONSOLIDATED EOTS Configuration - Pydantic-first architecture with unified settings groups."""

    # Core System Configuration
    system_settings: SystemSettings = Field(default_factory=lambda: SystemSettings(
        project_root_override=None,
        logging_level="INFO",
        log_to_file=True,
        log_file_path="logs/eots_v2_5.log",
        max_log_file_size_bytes=10485760,
        backup_log_count=5,
        live_mode=True,
        fail_fast_on_errors=True,
        enable_ai_intelligence=True,
        enable_unified_orchestrator=True,
        enable_multi_database=True,
        metrics_for_dynamic_threshold_distribution_tracking=[],
        signal_activation={}
    ))

    # CONSOLIDATED CONFIGURATION GROUPS
    data_management: DataManagementConfigV2_5 = Field(default_factory=lambda: DataManagementConfigV2_5(
        tradier_api_key="REQUIRED_FROM_CONFIG",
        tradier_account_id="REQUIRED_FROM_CONFIG"
    ))
    analytics_engine: AnalyticsEngineConfigV2_5 = Field(default_factory=lambda: AnalyticsEngineConfigV2_5())
    ai_intelligence: AIConfigV2_5 = Field(default_factory=lambda: AIConfigV2_5())
    trading_strategy: TradingConfigV2_5 = Field(default_factory=lambda: TradingConfigV2_5())
    dashboard: DashboardConfigV2_5 = Field(default_factory=lambda: DashboardConfigV2_5())

    # Specialized Configuration (kept separate for specific functionality)
    database_settings: Optional[DatabaseSettings] = Field(None, description="Database connection settings")
    data_processor_settings: DataProcessorSettings = Field(default_factory=lambda: DataProcessorSettings())
    market_regime_engine_settings: MarketRegimeEngineSettings = Field(default_factory=lambda: MarketRegimeEngineSettings())
    ticker_context_analyzer_settings: TickerContextAnalyzerSettings = Field(default_factory=lambda: TickerContextAnalyzerSettings())
    key_level_identifier_settings: KeyLevelIdentifierSettings = Field(default_factory=lambda: KeyLevelIdentifierSettings())
    symbol_specific_overrides: SymbolSpecificOverrides = Field(default_factory=lambda: SymbolSpecificOverrides())
    time_of_day_definitions: TimeOfDayDefinitions = Field(default_factory=lambda: TimeOfDayDefinitions(
        market_open="09:30:00",
        market_close="16:00:00",
        pre_market_start="04:00:00",
        after_hours_end="20:00:00",
        eod_pressure_calc_time="15:00:00"
    ))
    intraday_collector_settings: IntradayCollectorSettings = Field(default_factory=lambda: IntradayCollectorSettings())
    huihui_model_config: HuiHuiModelConfigV2_5 = Field(default_factory=lambda: HuiHuiModelConfigV2_5())
    performance_tracker_settings_v2_5: PerformanceTrackerSettingsV2_5 = Field(default_factory=lambda: PerformanceTrackerSettingsV2_5(
        performance_data_directory="data_cache_v2_5/performance_data_store",
        historical_window_days=365,
        weight_smoothing_factor=0.1,
        min_sample_size=10,
        confidence_threshold=0.75,
        update_interval_seconds=3600,
        tracking_enabled=True,
        reporting_frequency="daily",
        benchmark_symbol="SPY"
    ))
    dte_filter: Optional[int] = Field(None, description="Days to expiration filter for options analysis", ge=0, le=365)

    model_config = ConfigDict(json_schema_extra={"$schema":"http://json-schema.org/draft-07/schema#","title":"EOTS_V2_5_Config_Schema","description":"Canonical schema for EOTS v2.5 configuration (config_v2_5.json). Defines all valid parameters, types, defaults, and descriptions for system operation."}, extra='allow')  # TEMPORARY: Allow extra fields during config migration

# ===== PYDANTIC AI ORCHESTRATOR MODELS =====

class OrchestratorDecisionContextV2_5(BaseModel):
    """Context information for AI orchestrator decision making."""
    symbol: str = Field(..., description="Trading symbol being analyzed")
    market_regime: Optional[str] = Field(None, description="Current market regime")
    data_quality_score: float = Field(default=0.0, description="Quality of available data (0.0-1.0)", ge=0.0, le=1.0)
    system_load: float = Field(default=0.0, description="Current system resource utilization", ge=0.0, le=1.0)
    historical_performance: Dict[str, float] = Field(default_factory=dict, description="Historical performance metrics")
    market_volatility: Optional[float] = Field(None, description="Current market volatility level")
    time_of_day: Optional[str] = Field(None, description="Market session timing context")
    recent_signals: List[str] = Field(default_factory=list, description="Recent signal types generated")
    active_recommendations_count: int = Field(default=0, description="Number of active recommendations", ge=0)
    timestamp: datetime = Field(default_factory=datetime.now, description="Context timestamp")

class OrchestratorDecisionV2_5(BaseModel):
    """AI orchestrator decision output."""
    decision_type: str = Field(..., description="Type of orchestration decision")
    priority_level: int = Field(..., description="Priority level (1=highest, 5=lowest)", ge=1, le=5)
    component_routing: Dict[str, bool] = Field(default_factory=dict, description="Which components to activate")
    parameter_adjustments: Dict[str, Any] = Field(default_factory=dict, description="Dynamic parameter adjustments")
    resource_allocation: Dict[str, float] = Field(default_factory=dict, description="Resource allocation percentages")
    confidence_score: float = Field(..., description="Confidence in decision (0.0-1.0)", ge=0.0, le=1.0)
    reasoning: str = Field(..., description="AI reasoning for the decision")
    expected_outcome: str = Field(..., description="Expected outcome from this decision")
    fallback_strategy: Optional[str] = Field(None, description="Fallback if primary decision fails")
    timestamp: datetime = Field(default_factory=datetime.now, description="Decision timestamp")

class OrchestratorPerformanceMetricsV2_5(BaseModel):
    """Performance tracking for AI orchestrator decisions."""
    decision_id: str = Field(..., description="Unique decision identifier")
    decision_type: str = Field(..., description="Type of decision made")
    execution_time_ms: float = Field(..., description="Time taken to execute decision", ge=0.0)
    success_rate: float = Field(..., description="Success rate of similar decisions", ge=0.0, le=1.0)
    resource_efficiency: float = Field(..., description="Resource utilization efficiency", ge=0.0, le=1.0)
    outcome_accuracy: Optional[float] = Field(None, description="Accuracy of predicted outcome", ge=0.0, le=1.0)
    learning_impact: float = Field(default=0.0, description="Impact on system learning", ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.now, description="Metrics creation timestamp")

class HuiHuiOrchestratorExpertV2_5(BaseModel):
    """Pydantic model for HuiHui Orchestrator Expert configuration."""
    expert_id: str = Field(default="huihui_orchestrator", description="Unique expert identifier")
    expert_name: str = Field(default="System Orchestrator Expert", description="Human-readable expert name")
    model_endpoint: str = Field(..., description="HuiHui model API endpoint")
    api_key: str = Field(..., description="API key for HuiHui access")
    temperature: float = Field(default=0.1, description="Model temperature for consistency", ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, description="Maximum tokens per response", ge=100, le=8000)
    timeout_seconds: int = Field(default=45, description="Request timeout in seconds", ge=5, le=120)
    retry_attempts: int = Field(default=3, description="Number of retry attempts", ge=1, le=5)
    decision_confidence_threshold: float = Field(default=0.7, description="Minimum confidence for decisions", ge=0.0, le=1.0)
    learning_rate: float = Field(default=0.1, description="Learning adaptation rate", ge=0.01, le=1.0)
    performance_tracking_enabled: bool = Field(default=True, description="Enable performance tracking")

class OrchestratorWorkflowStepV2_5(BaseModel):
    """Individual step in the orchestrator workflow."""
    step_id: str = Field(..., description="Unique step identifier")
    step_name: str = Field(..., description="Human-readable step name")
    component_name: str = Field(..., description="Component to execute")
    priority: int = Field(..., description="Execution priority", ge=1, le=10)
    dependencies: List[str] = Field(default_factory=list, description="Required previous steps")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Step-specific parameters")
    timeout_seconds: int = Field(default=30, description="Step timeout", ge=5, le=300)
    retry_on_failure: bool = Field(default=True, description="Whether to retry on failure")
    critical_step: bool = Field(default=False, description="Whether failure stops entire workflow")
    ai_optimizable: bool = Field(default=True, description="Whether AI can optimize this step")

class OrchestratorWorkflowV2_5(BaseModel):
    """Complete orchestrator workflow definition."""
    workflow_id: str = Field(..., description="Unique workflow identifier")
    workflow_name: str = Field(..., description="Human-readable workflow name")
    workflow_type: str = Field(..., description="Type of workflow (analysis, monitoring, etc.)")
    steps: List[OrchestratorWorkflowStepV2_5] = Field(..., description="Ordered list of workflow steps")
    total_estimated_time_seconds: int = Field(..., description="Estimated total execution time", ge=1)
    success_criteria: Dict[str, Any] = Field(default_factory=dict, description="Criteria for workflow success")
    failure_handling: Dict[str, str] = Field(default_factory=dict, description="Failure handling strategies")
    ai_optimization_enabled: bool = Field(default=True, description="Whether AI can optimize workflow")
    created_at: datetime = Field(default_factory=datetime.now, description="Workflow creation timestamp")
    last_optimized: Optional[datetime] = Field(None, description="Last AI optimization timestamp")

class ConsolidatedUnderlyingDataV2_5(BaseModel):
    """Consolidated underlying data model - replaces RawUnderlyingDataV2_5, RawUnderlyingDataCombinedV2_5, and ProcessedUnderlyingAggregatesV2_5."""
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(default_factory=datetime.now, description="Data timestamp")
    price: Optional[float] = Field(None, description="Current price")
    price_change_abs: Optional[float] = Field(None, description="Absolute price change")
    price_change_pct: Optional[float] = Field(None, description="Percentage price change")
    day_open: Optional[float] = Field(None, description="Day open price")
    day_high: Optional[float] = Field(None, description="Day high price")
    day_low: Optional[float] = Field(None, description="Day low price")
    prev_close: Optional[float] = Field(None, description="Previous close price")
    day_volume: Optional[int] = Field(None, description="Day volume")
    call_gxoi: Optional[float] = Field(None, description="Call gamma exposure")
    put_gxoi: Optional[float] = Field(None, description="Put gamma exposure")
    net_delta_flow: Optional[float] = Field(None, description="Net delta flow")
    net_gamma_flow: Optional[float] = Field(None, description="Net gamma flow")
    net_vega_flow: Optional[float] = Field(None, description="Net vega flow")
    net_theta_flow: Optional[float] = Field(None, description="Net theta flow")
    gib_oi_based: Optional[float] = Field(None, description="GIB OI-based metric")
    vapi_fa_z_score: Optional[float] = Field(None, description="VAPI FA Z-score")
    dwfd_z_score: Optional[float] = Field(None, description="DWFD Z-score")
    tw_laf_z_score: Optional[float] = Field(None, description="TW LAF Z-score")
    vri_0dte_sum: Optional[float] = Field(None, description="VRI 0DTE sum")
    vfi_0dte_sum: Optional[float] = Field(None, description="VFI 0DTE sum")
    vvr_0dte_avg: Optional[float] = Field(None, description="VVR 0DTE average")
    vci_0dte_agg: Optional[float] = Field(None, description="VCI 0DTE aggregate")
    a_mspi_summary_score: Optional[float] = Field(None, description="A-MSPI summary score")
    a_sai_avg: Optional[float] = Field(None, description="A-SAI average")
    a_ssi_avg: Optional[float] = Field(None, description="A-SSI average")
    vri_2_0_aggregate: Optional[float] = Field(None, description="VRI 2.0 aggregate")
    e_sdag_mult: Optional[float] = Field(None, description="E-SDAG multiplier")
    a_dag_total: Optional[float] = Field(None, description="A-DAG total")
    net_value_flow_5m: Optional[float] = Field(None, description="5-minute net value flow")
    net_value_flow_15m: Optional[float] = Field(None, description="15-minute net value flow")
    net_value_flow_30m: Optional[float] = Field(None, description="30-minute net value flow")
    net_value_flow_60m: Optional[float] = Field(None, description="60-minute net value flow")
    total_nvp: Optional[float] = Field(None, description="Total net value position")
    volatility: Optional[float] = Field(None, description="Implied volatility")
    atr: Optional[float] = Field(None, description="Average true range")
    model_config = ConfigDict(extra='forbid')

class ChartLayoutConfigV2_5(BaseModel):
    """Pydantic model for unified chart layout configuration - EOTS v2.5 compliant."""
    title_text: str = Field(..., description="Chart title text")
    title_x: float = Field(default=0.5, description="Title x position", ge=0.0, le=1.0)
    title_font_size: int = Field(default=14, description="Title font size", ge=8, le=24)
    title_font_color: str = Field(default="white", description="Title font color")
    paper_bgcolor: str = Field(default="rgba(0, 0, 0, 0)", description="Paper background color")
    plot_bgcolor: str = Field(default="rgba(0, 0, 0, 0)", description="Plot background color")
    font_color: str = Field(default="white", description="Default font color")
    height: int = Field(default=200, description="Chart height in pixels", ge=100, le=1000)
    margin_left: int = Field(default=20, description="Left margin", ge=0, le=100)
    margin_right: int = Field(default=20, description="Right margin", ge=0, le=100)
    margin_top: int = Field(default=40, description="Top margin", ge=0, le=100)
    margin_bottom: int = Field(default=20, description="Bottom margin", ge=0, le=100)
    xaxis_gridcolor: str = Field(default="rgba(255, 255, 255, 0.1)", description="X-axis grid color")
    xaxis_tickfont_color: str = Field(default="white", description="X-axis tick font color")
    yaxis_gridcolor: str = Field(default="rgba(255, 255, 255, 0.1)", description="Y-axis grid color")
    yaxis_tickfont_color: str = Field(default="white", description="Y-axis tick font color")

    def to_plotly_layout(self) -> Dict[str, Any]:
        """Convert to Plotly layout dictionary."""
        return {
            'title': {
                'text': self.title_text,
                'x': self.title_x,
                'font': {'size': self.title_font_size, 'color': self.title_font_color}
            },
            'paper_bgcolor': self.paper_bgcolor,
            'plot_bgcolor': self.plot_bgcolor,
            'font': {'color': self.font_color},
            'height': self.height,
            'margin': dict(
                l=self.margin_left,
                r=self.margin_right,
                t=self.margin_top,
                b=self.margin_bottom
            ),
            'xaxis': {
                'gridcolor': self.xaxis_gridcolor,
                'tickfont': {'color': self.xaxis_tickfont_color}
            },
            'yaxis': {
                'gridcolor': self.yaxis_gridcolor,
                'tickfont': {'color': self.yaxis_tickfont_color}
            }
        }

    model_config = ConfigDict(extra='forbid')

class CrossAssetAnalysis(BaseModel):
    """PYDANTIC-FIRST: Cross-asset regime analysis"""
    asset_correlations: Dict[str, float] = Field(default_factory=dict, description="Asset class correlations")
    regime_consistency: float = Field(default=0.0, description="Cross-asset regime consistency")
    divergence_signals: List[str] = Field(default_factory=list, description="Cross-asset divergence signals")
    confirmation_signals: List[str] = Field(default_factory=list, description="Cross-asset confirmation signals")
    # Asset-specific regimes
    equity_regime: Optional[str] = Field(None, description="Equity market regime")
    bond_regime: Optional[str] = Field(None, description="Bond market regime")
    commodity_regime: Optional[str] = Field(None, description="Commodity market regime")
    currency_regime: Optional[str] = Field(None, description="Currency market regime")
    model_config = ConfigDict(extra='forbid')

# Add after line 2067 (before the end of the file)

class ControlPanelParametersV2_5(BaseModel):
    """PYDANTIC-FIRST: Control panel parameters with strict validation.
    
    *** STRICT VALIDATION: NO DEFAULTS, NO FALLBACKS ***
    *** ALL PARAMETERS MUST BE EXPLICITLY PROVIDED ***
    *** VALIDATION ERRORS WILL FAIL FAST ***
    """
    
    symbol: str = Field(
        ...,  # No default - must be provided
        description="Trading symbol (e.g., 'SPY', 'SPX')",
        min_length=1,
        max_length=10
    )
    
    dte_min: int = Field(
        ...,  # No default - must be provided
        description="Minimum days to expiration",
        ge=0,
        le=365
    )
    
    dte_max: int = Field(
        ...,  # No default - must be provided
        description="Maximum days to expiration",
        ge=0,
        le=365
    )
    
    price_range_percent: int = Field(
        ...,  # No default - must be provided
        description="Price range percentage for filtering",
        ge=1,
        le=100
    )
    
    fetch_interval_seconds: int = Field(
        ...,  # No default - must be provided
        description="Data fetch interval in seconds",
        ge=5,
        le=3600
    )
    
    @field_validator('dte_max')
    @classmethod
    def validate_dte_range(cls, v: int, info: FieldValidationInfo) -> int:
        """Validate that dte_max is greater than or equal to dte_min."""
        if 'dte_min' in info.data and v < info.data['dte_min']:
            raise ValueError("dte_max must be greater than or equal to dte_min")
        return v
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate symbol format."""
        if not v.isalpha():
            raise ValueError("symbol must contain only letters")
        return v.upper()
    
    model_config = ConfigDict(extra='forbid', frozen=True)  # No extra fields allowed, immutable after creation


class AIHubComplianceReportV2_5(BaseModel):
    """PYDANTIC-FIRST: AI Hub compliance reporting and validation."""
    control_panel_params: ControlPanelParametersV2_5 = Field(..., description="Current control panel parameters")
    
    # Data filtering compliance
    options_contracts_filtered: int = Field(default=0, description="Number of options contracts after filtering")
    options_contracts_total: int = Field(default=0, description="Total options contracts before filtering")
    dte_filter_applied: bool = Field(default=False, description="Whether DTE filter was applied")
    price_filter_applied: bool = Field(default=False, description="Whether price range filter was applied")
    
    # Dashboard component compliance
    components_respecting_filters: List[str] = Field(default_factory=list, description="Dashboard components respecting filters")
    components_not_respecting_filters: List[str] = Field(default_factory=list, description="Dashboard components NOT respecting filters")
    
    # Metrics compliance
    metrics_calculated_with_filters: List[str] = Field(default_factory=list, description="Metrics calculated using filtered data")
    metrics_using_raw_data: List[str] = Field(default_factory=list, description="Metrics using unfiltered data")
    
    # Overall compliance score
    compliance_score: float = Field(default=0.0, description="Overall compliance score (0.0-1.0)", ge=0.0, le=1.0)
    compliance_status: str = Field(default="UNKNOWN", description="Compliance status", pattern="^(COMPLIANT|PARTIAL|NON_COMPLIANT|UNKNOWN)$")
    
    # Compliance details
    compliance_issues: List[str] = Field(default_factory=list, description="List of compliance issues found")
    compliance_recommendations: List[str] = Field(default_factory=list, description="Recommendations to improve compliance")
    
    # Timestamps
    report_timestamp: datetime = Field(default_factory=datetime.now, description="When compliance report was generated")
    data_timestamp: Optional[datetime] = Field(None, description="Timestamp of analyzed data")
    
    def calculate_compliance_score(self) -> float:
        """
        ðŸš€ REAL COMPLIANCE CALCULATION: Calculate overall compliance score using REAL tracking data.
        
        This method now integrates with ComponentComplianceTracker for accurate scoring
        instead of relying on fake empty lists.
        """
        # Get real tracking data from compliance tracker
        try:
            from dashboard_application.modes.ai_dashboard.component_compliance_tracker_v2_5 import get_compliance_tracker
            tracker = get_compliance_tracker()
            real_summary = tracker.get_compliance_summary()
            
            # Use real tracking data instead of fake empty lists
            self.components_respecting_filters = real_summary["components_respecting_filters"]
            self.components_not_respecting_filters = real_summary["components_not_respecting_filters"]
            self.metrics_calculated_with_filters = real_summary["metrics_calculated_with_filters"]
            self.metrics_using_raw_data = real_summary["metrics_using_raw_data"]
            
            # Use the real compliance score from tracker
            self.compliance_score = round(real_summary["overall_compliance_score"], 3)
            
            # Update status based on real score
            if self.compliance_score >= 0.9:
                self.compliance_status = "COMPLIANT"
            elif self.compliance_score >= 0.7:
                self.compliance_status = "PARTIAL"
            else:
                self.compliance_status = "NON_COMPLIANT"
            
            return self.compliance_score
            
        except ImportError:
            # Fallback to legacy calculation if tracker not available
            total_components = len(self.components_respecting_filters) + len(self.components_not_respecting_filters)
            if total_components == 0:
                self.compliance_score = 0.0
                self.compliance_status = "UNKNOWN"
                return 0.0
            
            component_score = len(self.components_respecting_filters) / total_components
            
            # Factor in filter application
            filter_score = 0.0
            if self.dte_filter_applied:
                filter_score += 0.5
            if self.price_filter_applied:
                filter_score += 0.5
            
            # Factor in metrics compliance
            total_metrics = len(self.metrics_calculated_with_filters) + len(self.metrics_using_raw_data)
            metrics_score = 0.0
            if total_metrics > 0:
                metrics_score = len(self.metrics_calculated_with_filters) / total_metrics
            
            # Weighted average
            final_score = (component_score * 0.5) + (filter_score * 0.3) + (metrics_score * 0.2)
            self.compliance_score = round(final_score, 3)
            
            # Update status based on score
            if self.compliance_score >= 0.9:
                self.compliance_status = "COMPLIANT"
            elif self.compliance_score >= 0.7:
                self.compliance_status = "PARTIAL"
            else:
                self.compliance_status = "NON_COMPLIANT"
            
            return self.compliance_score
    
    def add_compliance_issue(self, issue: str) -> None:
        """Add a compliance issue to the report."""
        if issue not in self.compliance_issues:
            self.compliance_issues.append(issue)
    
    def add_compliance_recommendation(self, recommendation: str) -> None:
        """Add a compliance recommendation to the report."""
        if recommendation not in self.compliance_recommendations:
            self.compliance_recommendations.append(recommendation)
    
    model_config = ConfigDict(extra='forbid')


class FilteredDataBundleV2_5(BaseModel):
    """PYDANTIC-FIRST: Data bundle with applied control panel filters for AI Hub compliance."""
    
    # Original data reference
    original_bundle: FinalAnalysisBundleV2_5 = Field(..., description="Original unfiltered data bundle")
    
    # Applied filters
    applied_filters: ControlPanelParametersV2_5 = Field(..., description="Filters that were applied")
    
    # Filtered data
    filtered_options_contracts: List[ProcessedContractMetricsV2_5] = Field(default_factory=list, description="Options contracts after filtering")
    filtered_strike_data: List[ProcessedStrikeLevelMetricsV2_5] = Field(default_factory=list, description="Strike-level data after filtering")
    
    # Filter statistics
    original_contracts_count: int = Field(default=0, description="Original number of contracts")
    filtered_contracts_count: int = Field(default=0, description="Number of contracts after filtering")
    contracts_removed_by_dte: int = Field(default=0, description="Contracts removed by DTE filter")
    contracts_removed_by_price: int = Field(default=0, description="Contracts removed by price filter")
    
    # Underlying data (not filtered, but annotated)
    underlying_data: ProcessedUnderlyingAggregatesV2_5 = Field(..., description="Underlying data (not filtered)")
    
    # Compliance report
    compliance_report: AIHubComplianceReportV2_5 = Field(..., description="Compliance report for this filtered bundle")
    
    # Metadata
    filter_timestamp: datetime = Field(default_factory=datetime.now, description="When filtering was applied")
    filter_duration_ms: float = Field(default=0.0, description="Time taken to apply filters in milliseconds")
    
    @classmethod
    def create_from_bundle(cls, original_bundle: FinalAnalysisBundleV2_5, control_params: ControlPanelParametersV2_5) -> 'FilteredDataBundleV2_5':
        """Create a filtered data bundle from original bundle and control panel parameters."""
        import time
        start_time = time.time()
        
        # Initialize counters
        original_count = 0
        filtered_contracts = []
        filtered_strikes = []
        removed_by_dte = 0
        removed_by_price = 0
        
        # Get current underlying price for price range filtering
        underlying_price = None
        if (original_bundle.processed_data_bundle and 
            original_bundle.processed_data_bundle.underlying_data_enriched and
            original_bundle.processed_data_bundle.underlying_data_enriched.price):
            underlying_price = original_bundle.processed_data_bundle.underlying_data_enriched.price
        
        # Calculate price range bounds
        price_range_lower = None
        price_range_upper = None
        if underlying_price:
            price_range_pct = control_params.price_range_percent / 100.0
            price_range_lower = underlying_price * (1 - price_range_pct)
            price_range_upper = underlying_price * (1 + price_range_pct)
        
        # Filter options contracts
        if (original_bundle.processed_data_bundle and 
            original_bundle.processed_data_bundle.options_data_with_metrics):
            
            for contract in original_bundle.processed_data_bundle.options_data_with_metrics:
                original_count += 1
                
                # Apply DTE filter
                if contract.dte_calc is not None:
                    if not (control_params.dte_min <= contract.dte_calc <= control_params.dte_max):
                        removed_by_dte += 1
                        continue
                
                # Apply price range filter
                if (contract.strike and price_range_lower and price_range_upper):
                    if not (price_range_lower <= contract.strike <= price_range_upper):
                        removed_by_price += 1
                        continue
                
                # Contract passed all filters
                filtered_contracts.append(contract)
        
        # Filter strike-level data
        if (original_bundle.processed_data_bundle and 
            original_bundle.processed_data_bundle.strike_level_data_with_metrics):
            
            for strike_data in original_bundle.processed_data_bundle.strike_level_data_with_metrics:
                # Apply price range filter to strikes
                if (strike_data.strike and price_range_lower and price_range_upper):
                    if price_range_lower <= strike_data.strike <= price_range_upper:
                        filtered_strikes.append(strike_data)
        
        # Create compliance report
        compliance_report = AIHubComplianceReportV2_5(
            control_panel_params=control_params,
            options_contracts_filtered=len(filtered_contracts),
            options_contracts_total=original_count,
            dte_filter_applied=True,
            price_filter_applied=price_range_lower is not None,
            data_timestamp=original_bundle.bundle_timestamp
        )
        
        # Calculate filter duration
        filter_duration = (time.time() - start_time) * 1000
        
        # Create filtered bundle
        filtered_bundle = cls(
            original_bundle=original_bundle,
            applied_filters=control_params,
            filtered_options_contracts=filtered_contracts,
            filtered_strike_data=filtered_strikes,
            original_contracts_count=original_count,
            filtered_contracts_count=len(filtered_contracts),
            contracts_removed_by_dte=removed_by_dte,
            contracts_removed_by_price=removed_by_price,
            underlying_data=ProcessedUnderlyingAggregatesV2_5(
                symbol=control_params.symbol, 
                timestamp=datetime.now(),
                gib_oi_based=0.0,
                gib_raw_gamma_units=0.0,
                # ... rest of fields as before ...
            ),
            compliance_report=compliance_report,
            filter_duration_ms=filter_duration
        )
        
        # Calculate compliance score
        compliance_report.calculate_compliance_score()
        
        return filtered_bundle
    
    def get_filter_summary(self) -> str:
        """Get a summary of applied filters and their impact."""
        filter_efficiency = (self.filtered_contracts_count / self.original_contracts_count * 100) if self.original_contracts_count > 0 else 0
        
        return (f"Filtered {self.original_contracts_count} contracts to {self.filtered_contracts_count} "
                f"({filter_efficiency:.1f}% retained) | "
                f"DTE: {self.applied_filters.dte_min}-{self.applied_filters.dte_max} | "
                f"Price: Â±{self.applied_filters.price_range_percent}% | "
                f"Processing: {self.filter_duration_ms:.1f}ms")
    
    model_config = ConfigDict(extra='forbid')

# ===== HUIHUI EXPERT-SPECIFIC SCHEMAS =====

class HuiHuiMarketRegimeSchema(BaseModel):
    """PYDANTIC-FIRST: Market Regime Expert specific data schema for database storage."""
    
    # Analysis metadata
    analysis_id: str = Field(..., description="Unique analysis identifier")
    expert_type: Literal["market_regime"] = Field(default="market_regime", description="Expert type")
    ticker: str = Field(..., description="Analyzed ticker")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    
    # Core regime analysis
    current_regime_id: int = Field(..., description="Current regime ID (1-20)")
    current_regime_name: str = Field(..., description="Current regime name")
    regime_confidence: float = Field(..., description="Regime confidence score", ge=0.0, le=1.0)
    regime_probability: float = Field(..., description="Regime probability", ge=0.0, le=1.0)
    
    # VRI 3.0 Components
    vri_3_composite: float = Field(..., description="VRI 3.0 composite score")
    volatility_regime_score: float = Field(..., description="Volatility regime component")
    flow_intensity_score: float = Field(..., description="Flow intensity component")
    regime_stability_score: float = Field(..., description="Regime stability component")
    transition_momentum_score: float = Field(..., description="Transition momentum component")
    
    # Regime characteristics
    volatility_level: str = Field(..., description="Volatility level (low/medium/high/extreme)")
    trend_direction: str = Field(..., description="Trend direction (bullish/bearish/sideways)")
    flow_pattern: str = Field(..., description="Flow pattern (accumulation/distribution/neutral)")
    risk_appetite: str = Field(..., description="Risk appetite (risk_on/risk_off/neutral)")
    
    # Transition prediction
    predicted_regime_id: Optional[int] = Field(None, description="Predicted next regime ID")
    transition_probability: float = Field(default=0.0, description="Transition probability", ge=0.0, le=1.0)
    expected_transition_timeframe: Optional[str] = Field(None, description="Expected transition timeframe")
    
    # Performance metrics
    processing_time_ms: float = Field(..., description="Processing time in milliseconds", ge=0.0)
    data_quality_score: float = Field(default=1.0, description="Data quality score", ge=0.0, le=1.0)
    confidence_level: float = Field(default=0.0, description="Overall confidence level", ge=0.0, le=1.0)
    
    # Supporting data
    supporting_indicators: List[str] = Field(default_factory=list, description="Supporting indicators")
    conflicting_indicators: List[str] = Field(default_factory=list, description="Conflicting indicators")
    early_warning_signals: List[str] = Field(default_factory=list, description="Early warning signals")
    
    # Errors and warnings
    errors: List[str] = Field(default_factory=list, description="Analysis errors")
    warnings: List[str] = Field(default_factory=list, description="Analysis warnings")

    model_config = ConfigDict(extra='forbid')

class HuiHuiOptionsFlowSchema(BaseModel):
    """PYDANTIC-FIRST: Options Flow Expert specific data schema for database storage."""
    
    # Analysis metadata
    analysis_id: str = Field(..., description="Unique analysis identifier")
    expert_type: Literal["options_flow"] = Field(default="options_flow", description="Expert type")
    ticker: str = Field(..., description="Analyzed ticker")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    
    # Core flow metrics
    vapi_fa_z_score: float = Field(..., description="VAPI-FA Z-score")
    dwfd_z_score: float = Field(..., description="DWFD Z-score")
    tw_laf_score: float = Field(..., description="TW-LAF score")
    gib_oi_based: float = Field(..., description="GIB OI-based value")
    
    # SDAG Analysis
    sdag_multiplicative: float = Field(default=0.0, description="SDAG multiplicative methodology")
    sdag_directional: float = Field(default=0.0, description="SDAG directional methodology")
    sdag_weighted: float = Field(default=0.0, description="SDAG weighted methodology")
    sdag_volatility_focused: float = Field(default=0.0, description="SDAG volatility-focused methodology")
    
    # DAG Analysis
    dag_multiplicative: float = Field(default=0.0, description="DAG multiplicative methodology")
    dag_additive: float = Field(default=0.0, description="DAG additive methodology")
    dag_weighted: float = Field(default=0.0, description="DAG weighted methodology")
    dag_consensus: float = Field(default=0.0, description="DAG consensus methodology")
    
    # Flow classification
    flow_type: str = Field(..., description="Primary flow type")
    flow_subtype: str = Field(..., description="Flow subtype")
    flow_intensity: str = Field(..., description="Flow intensity level")
    directional_bias: str = Field(..., description="Directional bias (bullish/bearish/neutral)")
    
    # Participant analysis
    institutional_probability: float = Field(..., description="Institutional participant probability", ge=0.0, le=1.0)
    retail_probability: float = Field(..., description="Retail participant probability", ge=0.0, le=1.0)
    dealer_probability: float = Field(..., description="Dealer participant probability", ge=0.0, le=1.0)
    
    # Intelligence metrics
    sophistication_score: float = Field(..., description="Flow sophistication score", ge=0.0, le=1.0)
    information_content: float = Field(..., description="Information content score", ge=0.0, le=1.0)
    market_impact_potential: float = Field(..., description="Potential market impact", ge=0.0, le=1.0)
    
    # Gamma dynamics
    gamma_exposure_net: float = Field(default=0.0, description="Net gamma exposure")
    gamma_concentration_risk: float = Field(default=0.0, description="Gamma concentration risk")
    dealer_hedging_pressure: float = Field(default=0.0, description="Dealer hedging pressure")
    
    # Performance metrics
    processing_time_ms: float = Field(..., description="Processing time in milliseconds", ge=0.0)
    contracts_processed: int = Field(..., description="Number of contracts processed", ge=0)
    processing_speed_cps: float = Field(..., description="Processing speed (contracts per second)", ge=0.0)
    accuracy_score: float = Field(default=0.0, description="Accuracy score", ge=0.0, le=1.0)
    confidence_level: float = Field(default=0.0, description="Overall confidence level", ge=0.0, le=1.0)
    
    # Risk assessment
    flow_risk_score: float = Field(default=0.0, description="Flow-based risk score", ge=0.0, le=1.0)
    liquidity_risk: float = Field(default=0.0, description="Liquidity risk assessment", ge=0.0, le=1.0)
    execution_risk: float = Field(default=0.0, description="Execution risk assessment", ge=0.0, le=1.0)
    
    # Supporting evidence
    supporting_indicators: List[str] = Field(default_factory=list, description="Supporting indicators")
    confidence_factors: List[str] = Field(default_factory=list, description="Confidence factors")
    
    # Errors and warnings
    errors: List[str] = Field(default_factory=list, description="Analysis errors")
    warnings: List[str] = Field(default_factory=list, description="Analysis warnings")

    model_config = ConfigDict(extra='forbid')

class HuiHuiSentimentSchema(BaseModel):
    """PYDANTIC-FIRST: Sentiment Expert specific data schema for database storage."""
    
    # Analysis metadata
    analysis_id: str = Field(..., description="Unique analysis identifier")
    expert_type: Literal["sentiment"] = Field(default="sentiment", description="Expert type")
    ticker: str = Field(..., description="Analyzed ticker")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    
    # Core sentiment metrics
    overall_sentiment_score: float = Field(..., description="Overall sentiment score", ge=-1.0, le=1.0)
    sentiment_direction: str = Field(..., description="Sentiment direction (bullish/bearish/neutral)")
    sentiment_strength: float = Field(..., description="Sentiment strength", ge=0.0, le=1.0)
    sentiment_confidence: float = Field(..., description="Sentiment confidence", ge=0.0, le=1.0)
    
    # Sentiment components
    price_action_sentiment: float = Field(default=0.0, description="Price action sentiment", ge=-1.0, le=1.0)
    volume_sentiment: float = Field(default=0.0, description="Volume sentiment", ge=-1.0, le=1.0)
    options_sentiment: float = Field(default=0.0, description="Options sentiment", ge=-1.0, le=1.0)
    news_sentiment: Optional[float] = Field(None, description="News sentiment (if available)", ge=-1.0, le=1.0)
    social_sentiment: Optional[float] = Field(None, description="Social media sentiment (if available)", ge=-1.0, le=1.0)
    
    # Behavioral analysis
    fear_greed_index: float = Field(default=0.0, description="Fear/Greed index", ge=0.0, le=100.0)
    contrarian_signals: List[str] = Field(default_factory=list, description="Contrarian signals detected")
    crowd_psychology_state: str = Field(default="neutral", description="Crowd psychology state")
    behavioral_biases: List[str] = Field(default_factory=list, description="Behavioral biases detected")
    
    # Market microstructure
    liquidity_sentiment: float = Field(default=0.0, description="Liquidity-based sentiment", ge=-1.0, le=1.0)
    volatility_sentiment: float = Field(default=0.0, description="Volatility-based sentiment", ge=-1.0, le=1.0)
    momentum_sentiment: float = Field(default=0.0, description="Momentum-based sentiment", ge=-1.0, le=1.0)
    
    # Risk regime analysis
    current_risk_regime: str = Field(..., description="Current risk regime")
    risk_regime_confidence: float = Field(..., description="Risk regime confidence", ge=0.0, le=1.0)
    risk_level: int = Field(..., description="Risk level (1-6: low to black swan)", ge=1, le=6)
    tail_risk_probability: float = Field(..., description="Tail risk probability", ge=0.0, le=1.0)
    
    # Sentiment dynamics
    sentiment_momentum: float = Field(default=0.0, description="Sentiment momentum", ge=0.0, le=1.0)
    sentiment_volatility: float = Field(default=0.0, description="Sentiment volatility", ge=0.0, le=1.0)
    sentiment_persistence: float = Field(default=0.0, description="Sentiment persistence", ge=0.0, le=1.0)
    sentiment_reversal_probability: float = Field(default=0.0, description="Sentiment reversal probability", ge=0.0, le=1.0)
    
    # Performance metrics
    processing_time_ms: float = Field(..., description="Processing time in milliseconds", ge=0.0)
    data_sources_used: List[str] = Field(default_factory=list, description="Data sources used in analysis")
    analysis_depth_score: float = Field(default=0.0, description="Analysis depth score", ge=0.0, le=1.0)
    confidence_level: float = Field(default=0.0, description="Overall confidence level", ge=0.0, le=1.0)
    
    # Supporting data
    key_sentiment_drivers: List[str] = Field(default_factory=list, description="Key sentiment drivers")
    sentiment_warnings: List[str] = Field(default_factory=list, description="Sentiment warnings")
    sentiment_opportunities: List[str] = Field(default_factory=list, description="Sentiment opportunities")
    
    # Errors and warnings
    errors: List[str] = Field(default_factory=list, description="Analysis errors")
    warnings: List[str] = Field(default_factory=list, description="Analysis warnings")

    model_config = ConfigDict(extra='forbid')

class HuiHuiOrchestratorSchema(BaseModel):
    """PYDANTIC-FIRST: Orchestrator Expert specific data schema for database storage."""
    
    # Analysis metadata
    analysis_id: str = Field(..., description="Unique analysis identifier")
    expert_type: Literal["orchestrator"] = Field(default="orchestrator", description="Expert type")
    ticker: str = Field(..., description="Analyzed ticker")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    
    # Orchestration decision
    decision_type: str = Field(..., description="Type of orchestration decision")
    priority_level: int = Field(..., description="Priority level (1=highest, 5=lowest)", ge=1, le=5)
    confidence_score: float = Field(..., description="Confidence in decision", ge=0.0, le=1.0)
    
    # Expert coordination
    experts_consulted: List[str] = Field(default_factory=list, description="Experts consulted for this decision")
    expert_consensus_score: float = Field(default=0.0, description="Expert consensus score", ge=0.0, le=1.0)
    conflicting_signals: List[str] = Field(default_factory=list, description="Conflicting signals between experts")
    
    # Strategic synthesis
    market_regime_assessment: str = Field(..., description="Market regime assessment")
    flow_assessment: str = Field(..., description="Options flow assessment")
    sentiment_assessment: str = Field(..., description="Sentiment assessment")
    overall_market_view: str = Field(..., description="Overall market view")
    
    # Recommendations
    primary_recommendation: str = Field(..., description="Primary trading recommendation")
    secondary_recommendations: List[str] = Field(default_factory=list, description="Secondary recommendations")
    risk_warnings: List[str] = Field(default_factory=list, description="Risk warnings")
    opportunity_highlights: List[str] = Field(default_factory=list, description="Opportunity highlights")
    
    # Resource allocation
    component_routing: Dict[str, bool] = Field(default_factory=dict, description="Which components to activate")
    parameter_adjustments: Dict[str, Any] = Field(default_factory=dict, description="Dynamic parameter adjustments")
    resource_allocation: Dict[str, float] = Field(default_factory=dict, description="Resource allocation percentages")
    
    # Strategic context
    market_context: Dict[str, Any] = Field(default_factory=dict, description="Market context factors")
    time_horizon: str = Field(..., description="Recommended time horizon")
    risk_tolerance: str = Field(..., description="Recommended risk tolerance")
    position_sizing_guidance: Optional[str] = Field(None, description="Position sizing guidance")
    
    # Performance tracking
    processing_time_ms: float = Field(..., description="Processing time in milliseconds", ge=0.0)
    decision_complexity_score: float = Field(default=0.0, description="Decision complexity score", ge=0.0, le=1.0)
    synthesis_quality_score: float = Field(default=0.0, description="Synthesis quality score", ge=0.0, le=1.0)
    
    # Outcome tracking
    expected_outcome: str = Field(..., description="Expected outcome from this decision")
    success_probability: float = Field(default=0.0, description="Success probability estimate", ge=0.0, le=1.0)
    fallback_strategy: Optional[str] = Field(None, description="Fallback if primary decision fails")
    
    # Learning data
    decision_reasoning: str = Field(..., description="AI reasoning for the decision")
    learning_insights: List[str] = Field(default_factory=list, description="Learning insights generated")
    adaptation_suggestions: List[str] = Field(default_factory=list, description="System adaptation suggestions")
    
    # Errors and warnings
    errors: List[str] = Field(default_factory=list, description="Analysis errors")
    warnings: List[str] = Field(default_factory=list, description="Analysis warnings")

    model_config = ConfigDict(extra='forbid')

# ===== HUIHUI UNIFIED EXPERT RESPONSE SCHEMA =====

# ===== UNIFIED AI INTELLIGENCE SCHEMAS =====

class UnifiedLearningResult(BaseModel):
    """Unified learning result combining memory, performance, and optimization."""
    symbol: str = Field(description="Trading symbol")
    timestamp: datetime = Field(default_factory=datetime.now, description="Learning timestamp")
    learning_insights: Union[List[str], Dict[str, Any]] = Field(default_factory=list, description="Learning insights discovered")
    performance_improvements: Dict[str, Any] = Field(default_factory=dict, description="Performance improvement results")
    expert_adaptations: Dict[str, Any] = Field(default_factory=dict, description="Expert adaptation results")
    confidence_updates: Dict[str, Any] = Field(default_factory=dict, description="Confidence scoring updates")
    next_learning_cycle: datetime = Field(default_factory=datetime.now, description="Next learning cycle time")
    
    # Additional fields that HuiHui learning system uses
    learning_cycle_type: Optional[str] = Field(None, description="Type of learning cycle")
    analysis_timestamp: Optional[datetime] = Field(None, description="Analysis timestamp")
    lookback_period_days: Optional[int] = Field(None, description="Lookback period in days")
    optimization_recommendations: Optional[List[str]] = Field(None, description="Optimization recommendations")
    performance_improvement_score: Optional[float] = Field(None, description="Performance improvement score")
    confidence_score: Optional[float] = Field(None, description="Confidence score")
    eots_schema_compliance: Optional[bool] = Field(None, description="EOTS schema compliance")
    learning_metadata: Optional[Dict[str, Any]] = Field(None, description="Learning metadata")
    
    model_config = ConfigDict(extra='allow')  # Allow extra fields for flexibility

class UnifiedIntelligenceAnalysis(BaseModel):
    """Unified intelligence analysis combining all AI systems."""
    symbol: str = Field(description="Trading symbol")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Overall confidence score")
    market_regime_analysis: str = Field(description="Market regime analysis")
    options_flow_analysis: str = Field(description="Options flow analysis")
    sentiment_analysis: str = Field(description="Sentiment analysis")
    strategic_recommendations: List[str] = Field(default_factory=list, description="Strategic recommendations")
    risk_assessment: str = Field(description="Risk assessment")
    learning_insights: List[str] = Field(default_factory=list, description="Learning insights")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    
    model_config = ConfigDict(extra='forbid')

class HuiHuiUnifiedExpertResponse(BaseModel):
    """PYDANTIC-FIRST: Unified response schema for all HuiHui experts."""
    
    # Common metadata
    analysis_id: str = Field(..., description="Unique analysis identifier")
    expert_type: HuiHuiExpertType = Field(..., description="Expert type used")
    ticker: str = Field(..., description="Analyzed ticker")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    # Usage tracking
    tokens_used: Optional[int] = Field(None, description="Tokens used in analysis")
    cache_hit: bool = Field(default=False, description="Whether result was from cache")
    @model_validator(mode='after')
    @classmethod
    def validate_expert_data(cls, values):
        """Ensure exactly one expert data field is populated."""
        expert_data_fields = [
            values.market_regime_data,
            values.options_flow_data, 
            values.sentiment_data,
            values.orchestrator_data
        ]
        
        populated_count = sum(1 for field in expert_data_fields if field is not None)
        
        if populated_count == 0:
            raise ValueError("At least one expert data field must be populated")
        elif populated_count > 1:
            raise ValueError("Only one expert data field should be populated")
            
        return values

    model_config = ConfigDict(extra='forbid')

class PatternThresholds(BaseModel):
    """Pydantic model for pattern detection thresholds with enhanced validation."""
    volatility_expansion: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Threshold for volatility expansion patterns"
    )
    trend_continuation: float = Field(
        default=0.65, ge=0.0, le=1.0,
        description="Threshold for trend continuation patterns"
    )
    accumulation: float = Field(
        default=0.75, ge=0.0, le=1.0,
        description="Threshold for accumulation patterns"
    )
    distribution: float = Field(
        default=0.75, ge=0.0, le=1.0,
        description="Threshold for distribution patterns"
    )
    consolidation: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="Threshold for consolidation patterns"
    )
    significant_pos_thresh: float = Field(
        default=1000.0, ge=0.0,
        description="Threshold for significant positive flow"
    )
    dwfd_strong_thresh: float = Field(
        default=1.5, ge=0.0,
        description="Threshold for strong DWFD signals"
    )
    moderate_confidence_thresh: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Threshold for moderate confidence"
    )
    vapi_fa_bullish_thresh: float = Field(
        default=1.5, ge=0.0,
        description="Z-score threshold for bullish VAPI-FA"
    )
    vapi_fa_bearish_thresh: float = Field(
        default=-1.5, le=0.0,
        description="Z-score threshold for bearish VAPI-FA"
    )
    vri_bullish_thresh: float = Field(
        default=0.6, ge=0.0,
        description="VRI threshold for bullish regime"
    )
    vri_bearish_thresh: float = Field(
        default=-0.6, le=0.0,
        description="VRI threshold for bearish regime"
    )
    hedging_pressure_thresh: float = Field(
        default=500.0, ge=0.0,
        description="Threshold for significant hedging pressure"
    )

    model_config = ConfigDict(
        extra='forbid',
        json_encoders={
            float: lambda v: round(v, 6)  # Round float values to 6 decimal places
        }
    )

    @model_validator(mode='before')
    def validate_threshold_values(cls, values):
        """Validate that thresholds are reasonable."""
        if not isinstance(values, dict):
            return values
        for field_name, value in values.items():
            if isinstance(value, (int, float)) and 'thresh' in field_name and abs(value) > 10000:
                raise ValueError(f"Threshold {field_name} is unreasonably large: {value}")
        return values

class MarketPattern(BaseModel):
    """Pydantic model for market patterns with full validation and enhanced features."""
    pattern_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique pattern identifier"
    )
    pattern_type: str = Field(
        ...,
        description="Type of pattern (VOLATILITY_EXPANSION, TREND_CONTINUATION, etc.)"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Pattern confidence score"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Pattern detection timestamp"
    )
    market_conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Market conditions when pattern detected"
    )
    supporting_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Supporting metrics for pattern"
    )
    regime_state: MarketRegimeState = Field(
        default=MarketRegimeState.UNDEFINED,
        description="Market regime when pattern detected"
    )
    huihui_confluence: Optional[Dict[str, Any]] = Field(
        default=None,
        description="HuiHui expert confluence data"
    )
    pattern_duration: Optional[timedelta] = Field(
        default=None,
        description="Expected or historical pattern duration"
    )
    pattern_strength: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Pattern strength indicator"
    )
    key_levels: Dict[str, float] = Field(
        default_factory=dict,
        description="Key price levels associated with the pattern"
    )
    volume_profile: Dict[str, float] = Field(
        default_factory=dict,
        description="Volume profile data supporting the pattern"
    )
    options_flow_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Options flow data supporting the pattern"
    )

    model_config = ConfigDict(extra='forbid')  # Strict validation
    
    # JSON serialization configuration
    json_encoders: ClassVar[Dict[Any, Any]] = {
        datetime: lambda v: v.isoformat(),
        MarketRegimeState: lambda v: v.value,
        timedelta: lambda v: v.total_seconds(),
        float: lambda v: round(v, 6),
    }

    @field_validator('pattern_type')
    @classmethod
    def validate_pattern_type(cls, v: str) -> str:
        """Validate pattern type is one of the allowed values."""
        allowed_patterns = {
            'VOLATILITY_EXPANSION',
            'TREND_CONTINUATION',
            'ACCUMULATION',
            'DISTRIBUTION',
            'CONSOLIDATION',
            'BREAKOUT',
            'BREAKDOWN',
            'REVERSAL',
            'DIVERGENCE',
            'MOMENTUM'
        }
        if v not in allowed_patterns:
            raise ValueError(f"Invalid pattern type. Must be one of: {allowed_patterns}")
        return v

    @field_validator('market_conditions')
    @classmethod
    def validate_market_conditions(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate market conditions have required fields."""
        required_fields = {
            'volatility',
            'trend_strength',
            'volume_trend',
            'price_momentum',
            'market_regime',
            'liquidity_state'
        }
        missing_fields = required_fields - set(v.keys())
        if missing_fields:
            raise ValueError(f"Market conditions missing required fields: {missing_fields}")
        return v

    @field_validator('supporting_metrics')
    @classmethod
    def validate_supporting_metrics(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate supporting metrics have required fields."""
        required_fields = {
            'vapi_fa_z',
            'dwfd_z',
            'tw_laf_z',
            'vri_score',
            'flow_score',
            'momentum_score'
        }
        missing_fields = required_fields - set(v.keys())
        if missing_fields:
            raise ValueError(f"Supporting metrics missing required fields: {missing_fields}")
        return v

    @field_validator('key_levels')
    @classmethod
    def validate_key_levels(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate key levels are within reasonable range of each other."""
        if len(v) >= 2:
            levels = sorted(v.values())
            max_gap = max(b - a for a, b in zip(levels, levels[1:]))
            if max_gap > 100:  # Arbitrary threshold, adjust based on your needs
                raise ValueError(f"Unreasonable gap between key levels: {max_gap}")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary for storage."""
        return self.model_dump()

    def merge_confluence_data(self, new_data: Dict[str, Any]) -> None:
        """Merge new confluence data with existing data."""
        if self.huihui_confluence is None:
            self.huihui_confluence = {}
        self.huihui_confluence.update(new_data)
        self._update_confidence_score()

    def _update_confidence_score(self) -> None:
        """Update confidence score based on supporting data."""
        factors = [
            self.pattern_strength,
            len(self.supporting_metrics) / 6,  # Normalize by required metrics
            len(self.market_conditions) / 6,   # Normalize by required conditions
            float(self.regime_state != MarketRegimeState.UNDEFINED),
            float(bool(self.huihui_confluence))
        ]
        self.confidence_score = sum(factors) / len(factors)

class StrategicRecommendationV2_5(BaseModel):
    """Pydantic model for strategic recommendations with full validation."""
    recommendation_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique recommendation identifier")
    ticker: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(default_factory=datetime.now, description="Recommendation timestamp")
    
    # Core recommendation data
    recommendation_type: str = Field(..., description="Type of recommendation (entry, exit, adjustment)")
    direction: str = Field(..., description="Trade direction (long, short, neutral)")
    time_horizon: str = Field(..., description="Time horizon for the recommendation")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in recommendation")
    
    # Strategy details
    strategy_name: str = Field(..., description="Name of the recommended strategy")
    entry_criteria: List[str] = Field(default_factory=list, description="Entry criteria")
    exit_criteria: List[str] = Field(default_factory=list, description="Exit criteria")
    risk_parameters: Dict[str, Any] = Field(default_factory=dict, description="Risk management parameters")
    
    # Supporting data
    market_context: Dict[str, Any] = Field(default_factory=dict, description="Market context at recommendation time")
    supporting_signals: List[str] = Field(default_factory=list, description="Supporting signals")
    risk_factors: List[str] = Field(default_factory=list, description="Risk factors to consider")
    
    # HuiHui expert insights
    expert_confluence: Dict[str, float] = Field(default_factory=dict, description="Expert confluence scores")
    regime_context: str = Field(..., description="Market regime context")
    flow_context: str = Field(..., description="Options flow context")
    sentiment_context: str = Field(..., description="Market sentiment context")
    
    # Performance tracking
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time in milliseconds")
    data_quality_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Data quality score")
    
    # Errors and warnings
    errors: List[str] = Field(default_factory=list, description="Analysis errors")
    warnings: List[str] = Field(default_factory=list, description="Analysis warnings")
    
    model_config = ConfigDict(extra='forbid')

class LearningInsightV2_5(BaseModel):
    """Pydantic model for learning insights with full validation."""
    insight_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique insight identifier")
    ticker: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(default_factory=datetime.now, description="Insight timestamp")
    
    # Core insight data
    insight_type: str = Field(..., description="Type of insight (pattern, anomaly, adaptation)")
    insight_category: str = Field(..., description="Category (market_regime, flow, sentiment, etc.)")
    insight_description: str = Field(..., description="Detailed description of the insight")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in insight")
    
    # Learning context
    learning_source: str = Field(..., description="Source of the learning (backtest, live, simulation)")
    market_context: Dict[str, Any] = Field(default_factory=dict, description="Market context when insight was gained")
    supporting_data: Dict[str, Any] = Field(default_factory=dict, description="Supporting data points")
    
    # Adaptation suggestions
    adaptation_type: Optional[str] = Field(None, description="Type of adaptation suggested")
    adaptation_params: Dict[str, Any] = Field(default_factory=dict, description="Suggested parameter adjustments")
    expected_improvement: str = Field(..., description="Expected improvement from adaptation")
    
    # Verification metrics
    verification_status: str = Field(default="pending", description="Insight verification status")
    verification_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Verification score if validated")
    verification_timestamp: Optional[datetime] = Field(None, description="When insight was verified")
    
    # Integration tracking
    integration_status: str = Field(default="pending", description="Integration status of insight")
    integration_priority: int = Field(default=3, ge=1, le=5, description="Priority for integration (1=highest)")
    integration_complexity: int = Field(default=3, ge=1, le=5, description="Complexity of integration (1=lowest)")
    
    # Performance impact
    performance_impact_area: List[str] = Field(default_factory=list, description="Areas impacted by insight")
    performance_metrics_pre: Dict[str, float] = Field(default_factory=dict, description="Performance metrics before")
    performance_metrics_post: Dict[str, float] = Field(default_factory=dict, description="Performance metrics after")
    
    # Error tracking
    errors: List[str] = Field(default_factory=list, description="Errors in insight generation")
    warnings: List[str] = Field(default_factory=list, description="Warnings about insight")
    
    model_config = ConfigDict(extra='forbid')

class MarketPredictionV2_5(BaseModel):
    """Pydantic model for market predictions."""
    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = Field(..., description="Trading symbol")
    prediction_type: str = Field(..., description="Type of prediction (price, trend, volatility)")
    prediction_value: float = Field(..., description="Predicted value")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in prediction")
    time_horizon: str = Field(..., description="Time horizon for prediction")
    prediction_timestamp: datetime = Field(default_factory=datetime.now)
    market_context: Dict[str, Any] = Field(default_factory=dict)
    model_metadata: Dict[str, Any] = Field(default_factory=dict)
    validation_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(extra='forbid')

class PredictionPerformanceV2_5(BaseModel):
    """Pydantic model for tracking prediction performance."""
    prediction_id: str = Field(..., description="Reference to original prediction")
    symbol: str = Field(..., description="Trading symbol")
    actual_value: float = Field(..., description="Actual observed value")
    prediction_error: float = Field(..., description="Absolute prediction error")
    relative_error: float = Field(..., description="Relative prediction error")
    direction_correct: bool = Field(..., description="Whether direction was predicted correctly")
    validation_timestamp: datetime = Field(default_factory=datetime.now)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    
    model_config = ConfigDict(extra='forbid')

class PredictionConfigV2_5(BaseModel):
    """Pydantic model for prediction configuration."""
    prediction_window: int = Field(default=30, ge=1, description="Days to look back for predictions")
    min_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence for predictions")
    max_predictions_per_symbol: int = Field(default=5, ge=1, description="Maximum predictions per symbol")
    prediction_frequency: str = Field(default="1H", description="How often to generate predictions")
    validation_frequency: str = Field(default="1D", description="How often to validate predictions")
    
    model_config = ConfigDict(extra='forbid')

class UnifiedPredictionResult(BaseModel):
    """Unified prediction result with proper validation."""
    symbol: str = Field(..., description="Trading symbol")
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    predictions: List[MarketPredictionV2_5] = Field(default_factory=list)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    model_performance: Dict[str, Any] = Field(default_factory=dict)
    confidence_metrics: Dict[str, float] = Field(default_factory=dict)
    next_prediction_cycle: datetime = Field(..., description="Next prediction cycle timestamp")
    prediction_cycle_type: str = Field(..., description="Type of prediction cycle")
    lookback_period_days: int = Field(..., ge=1, description="Lookback period in days")
    prediction_quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall prediction quality")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    optimization_recommendations: List[str] = Field(default_factory=list)
    eots_schema_compliance: bool = Field(default=True)
    prediction_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(extra='forbid')

class OptimizationConfigV2_5(BaseModel):
    """Pydantic model for optimization configuration."""
    optimization_window: int = Field(default=30, ge=1, description="Days to look back for optimization")
    min_trades_required: int = Field(default=10, ge=1, description="Minimum trades required for optimization")
    max_parameter_change: float = Field(default=0.2, ge=0.0, le=1.0, description="Maximum allowed parameter change")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence for changes")
    optimization_frequency: str = Field(default="1D", description="How often to run optimization")
    
    model_config = ConfigDict(extra='forbid')

class OptimizationMetricsV2_5(BaseModel):
    """Pydantic model for tracking optimization metrics."""
    total_optimizations: int = Field(default=0, ge=0)
    successful_optimizations: int = Field(default=0, ge=0)
    failed_optimizations: int = Field(default=0, ge=0)
    average_improvement: float = Field(default=0.0, ge=0.0)
    optimization_cycles_completed: int = Field(default=0, ge=0)
    total_processing_time_ms: float = Field(default=0.0, ge=0.0)
    
    model_config = ConfigDict(extra='forbid')

class ParameterOptimizationResultV2_5(BaseModel):
    """Pydantic model for optimization results."""
    optimization_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = Field(..., description="Trading symbol")
    parameter_name: str = Field(..., description="Name of optimized parameter")
    old_value: float = Field(..., description="Previous parameter value")
    new_value: float = Field(..., description="Optimized parameter value")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in optimization")
    performance_impact: Dict[str, float] = Field(default_factory=dict, description="Expected performance impact")
    optimization_timestamp: datetime = Field(default_factory=datetime.now)
    validation_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(extra='forbid')

class SystemStateV2_5(BaseModel):
    """Pydantic model for tracking the complete system state."""
    system_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique system instance ID")
    startup_time: datetime = Field(default_factory=datetime.now, description="System startup timestamp")
    last_update: datetime = Field(default_factory=datetime.now, description="Last state update timestamp")
    
    # Core System Status
    is_running: bool = Field(default=False, description="Whether the system is currently running")
    current_mode: str = Field(default="initialization", description="Current system operation mode")
    active_processes: List[str] = Field(default_factory=list, description="List of active system processes")
    
    # Component States
    ai_system_health: AISystemHealthV2_5 = Field(default_factory=lambda: AISystemHealthV2_5(), description="AI system health status")
    learning_status: LearningStatusV2_5 = Field(default_factory=lambda: LearningStatusV2_5(), description="Learning system status")
    learning_metrics: LearningMetricsV2_5 = Field(default_factory=lambda: LearningMetricsV2_5(), description="Learning system metrics")
    
    # Performance Tracking
    system_metrics: Dict[str, Any] = Field(default_factory=dict, description="System performance metrics")
    error_count: int = Field(default=0, description="Total error count since startup")
    warning_count: int = Field(default=0, description="Total warning count since startup")
    
    # Resource Usage
    memory_usage_mb: float = Field(default=0.0, description="Current memory usage in MB")
    cpu_usage_percent: float = Field(default=0.0, description="Current CPU usage percentage")
    disk_usage_percent: float = Field(default=0.0, description="Current disk usage percentage")
    
    # System Messages
    status_message: str = Field(default="System initializing...", description="Current status message")
    last_error: Optional[str] = Field(default=None, description="Last error message")
    last_warning: Optional[str] = Field(default=None, description="Last warning message")
    
    model_config = ConfigDict(extra='forbid')

class ProcessedUnderlyingAggregatesV2_5(ConsolidatedUnderlyingDataV2_5):
    """PYDANTIC-FIRST: Processed underlying aggregates with enriched metrics."""
    gib_oi_based: Optional[float] = None
    gib_raw_gamma_units: Optional[float] = None
    # ... (other additional fields as needed) ...
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow'
    )

ProcessedUnderlyingAggregatesV2_5 = ConsolidatedUnderlyingDataV2_5  # Backward compatibility alias

if __name__ == '__main__':
    print("Pydantic models defined.")












