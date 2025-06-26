"""
Pydantic models for the HuiHui AI Expert System integration in EOTS v2.5.

This module defines the data structures used for the HuiHui AI expert system,
including expert configurations, analysis requests/responses, and performance tracking.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, ConfigDict
from .base_types import PandasDataFrame
from .processed_data import ProcessedDataBundleV2_5
from .recommendation_schemas import ActiveRecommendationPayloadV2_5

class HuiHuiExpertType(str, Enum):
    """Defines the different types of experts available in the HuiHui AI system."""
    MARKET_REGIME = "market_regime"; OPTIONS_FLOW = "options_flow"; SENTIMENT = "sentiment"; ORCHESTRATOR = "orchestrator"
    VOLATILITY = "volatility"; LIQUIDITY = "liquidity"; RISK = "risk"; EXECUTION = "execution"

class HuiHuiModelConfigV2_5(BaseModel):
    """Configuration settings for individual HuiHui AI models controlling generation behavior and integration."""
    expert_type: HuiHuiExpertType = Field(default=HuiHuiExpertType.ORCHESTRATOR, description="Type of expert this configuration applies to.")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Controls randomness in model outputs (0=deterministic, 1.0=creative).")
    max_tokens: int = Field(default=4000, ge=100, le=8000, description="Maximum number of tokens to generate in the response.")
    enable_eots_integration: bool = Field(default=True, description="Whether to integrate with EOTS for additional context and data enrichment.")
    context_budget: int = Field(default=4000, ge=1000, le=8000, description="Number of tokens to reserve for context in the prompt.")
    timeout_seconds: int = Field(default=90, ge=30, le=300, description="Maximum time in seconds to wait for a response from the model.")
    model_config = ConfigDict(extra='forbid')


class HuiHuiExpertConfigV2_5(BaseModel):
    """Configuration for individual HuiHui experts, defining their behavior and capabilities."""
    expert_name: str = Field(..., min_length=3, max_length=100, description="Unique identifier for this expert configuration.")
    specialist_id: str = Field(..., description="ID of the specialist model to use for this expert (e.g., 'expert-market-regime-v2').")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Controls randomness in this expert's outputs (0=deterministic, 1.0=creative).")
    keywords: List[str] = Field(default_factory=list, description="List of keywords that trigger this expert's activation in the ensemble.")
    eots_metrics: List[str] = Field(default_factory=list, description="List of EOTS metrics this expert should have access to for analysis.")
    performance_weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Weight of this expert's opinion in ensemble decisions (0.0 to 1.0).")
    is_active: bool = Field(default=True, description="Whether this expert is currently active and should be included in the ensemble.")
    model_config = ConfigDict(extra='forbid')


class HuiHuiAnalysisRequestV2_5(BaseModel):
    """Request model for submitting analysis tasks to the HuiHui AI system."""
    symbol: str = Field(..., description="Trading symbol to analyze (e.g., 'SPY', 'QQQ').")
    analysis_type: str = Field(..., description="Type of analysis to perform (e.g., 'market_regime', 'flow_analysis').")
    bundle_data: Optional[ProcessedDataBundleV2_5] = Field(None, description="Processed market data bundle for analysis.")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for the analysis (e.g., market conditions, recent news).")
    expert_preference: Optional[HuiHuiExpertType] = Field(None, description="Preferred expert type for this analysis (optional).")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When this request was created (UTC).")
    request_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about this request (e.g., request ID, user ID).")
    model_config = ConfigDict(extra='forbid')


class HuiHuiAnalysisResponseV2_5(BaseModel):
    """Response model containing the results of a HuiHui AI analysis."""
    expert_used: HuiHuiExpertType = Field(..., description="Type of expert that generated this response.")
    analysis_content: str = Field(..., description="Detailed analysis content in markdown format with embedded visualizations.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Expert's confidence in the analysis (0.0 to 1.0).")
    processing_time: float = Field(..., ge=0.0, description="Time taken to generate the analysis in seconds.")
    insights: List[str] = Field(default_factory=list, description="Key insights extracted from the analysis.")
    eots_predictions: Optional[List[Dict[str, Any]]] = Field(None, description="Relevant EOTS predictions that informed this analysis.")
    recommendations: Optional[List[Dict[str, Any]]] = Field(None, description="Trading recommendations derived from the analysis.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When this response was generated (UTC).")
    model_config = ConfigDict(extra='forbid')


class HuiHuiUsageRecordV2_5(BaseModel):
    """Tracks usage and performance metrics for individual HuiHui expert invocations."""
    expert_used: HuiHuiExpertType = Field(..., description="Type of expert that was used for this analysis.")
    symbol: str = Field(..., description="Trading symbol that was analyzed (e.g., 'SPY', 'QQQ').")
    processing_time: float = Field(..., ge=0.0, description="Time taken to process the request in seconds.")
    success: bool = Field(..., description="Whether the request was processed successfully.")
    error_message: Optional[str] = Field(None, description="Error message if the request failed, None if successful.")
    market_condition: str = Field(default="normal", description="Prevailing market condition during analysis (e.g., 'bull', 'bear', 'range').")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When this usage record was created (UTC).")
    model_config = ConfigDict(extra='forbid')


class HuiHuiPerformanceMetricsV2_5(BaseModel):
    """Aggregated performance metrics for a specific HuiHui expert type over time."""
    expert_type: HuiHuiExpertType = Field(..., description="Type of expert these metrics apply to.")
    total_requests: int = Field(default=0, ge=0, description="Total number of requests processed by this expert type.")
    successful_requests: int = Field(default=0, ge=0, description="Number of successfully processed requests by this expert type.")
    average_processing_time: float = Field(default=0.0, ge=0.0, description="Average processing time in seconds across all requests.")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Ratio of successful requests to total requests (0.0 to 1.0).")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="When these metrics were last calculated (UTC).")
    performance_by_market_condition: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, 
        description="Performance metrics (success_rate, avg_processing_time) broken down by market condition (e.g., 'bull', 'bear')."
    )
    model_config = ConfigDict(extra='forbid')


class HuiHuiEnsembleConfigV2_5(BaseModel):
    """Configuration for creating and managing ensembles of multiple HuiHui experts."""
    ensemble_name: str = Field(..., description="Unique name identifying this expert ensemble configuration.")
    member_experts: List[HuiHuiExpertConfigV2_5] = Field(default_factory=list, description="List of expert configurations included in this ensemble.")
    voting_strategy: str = Field(default="weighted_average", description="Strategy for combining expert opinions (e.g., 'weighted_average', 'majority_vote').")
    min_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum confidence score (0.0-1.0) required for the ensemble to provide a final answer.")
    fallback_expert: Optional[str] = Field(None, description="Name of the expert to use when ensemble confidence is below the threshold.")
    model_config = ConfigDict(extra='forbid')


class HuiHuiUserFeedbackV2_5(BaseModel):
    """Stores user feedback and ratings for HuiHui AI analysis results."""
    request_id: str = Field(..., description="Unique identifier of the original analysis request this feedback refers to.")
    expert_used: HuiHuiExpertType = Field(..., description="Type of expert that generated the analysis being rated.")
    rating: int = Field(..., ge=1, le=5, description="User's rating of the analysis quality (1=Poor to 5=Excellent).")
    helpful: bool = Field(..., description="Whether the analysis was helpful for the user's specific needs.")
    comments: Optional[str] = Field(None, description="Optional detailed comments from the user about the analysis.")
    suggested_improvements: List[str] = Field(default_factory=list, description="Specific suggestions from the user for improving the analysis.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the feedback was submitted (UTC).")
    model_config = ConfigDict(extra='forbid')
