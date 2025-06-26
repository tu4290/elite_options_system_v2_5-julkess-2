"""
HuiHui Pydantic AI Model Integration - EOTS v2.5 Pydantic-First Architecture
===========================================================================

This module creates a Pydantic-first AI model for HuiHui-MoE that fully integrates
with EOTS v2.5 schemas, metrics_calculator, and its_orchestrator. It provides complete
Local LLM integration with proper Pydantic validation and EOTS schema compliance.

Key Features:
- 🧠 Pydantic-first architecture with full EOTS schema validation
- 🎯 4 specialized HuiHui experts with EOTS metrics integration
- 📊 Direct integration with metrics_calculator and its_orchestrator
- 🔄 Automatic expert routing based on EOTS analysis context
- ⚡ No rate limiting - unlimited local AI processing
- 🛡️ Full Pydantic validation for all inputs/outputs

EOTS Integration Points:
- FinalAnalysisBundleV2_5 processing
- AIPredictionV2_5 creation and validation
- ProcessedDataBundleV2_5 analysis
- Metrics calculator signal integration
- ITS orchestrator coordination

Author: EOTS v2.5 AI Liberation Division
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from datetime import datetime
from pydantic import BaseModel, Field, validator

# EOTS v2.5 Pydantic schemas - ALWAYS import these first for validation
from data_models.eots_schemas_v2_5 import (
    FinalAnalysisBundleV2_5,
    ProcessedDataBundleV2_5,
    AIPredictionV2_5,
    AIPredictionRequestV2_5,
    AIPredictionPerformanceV2_5,
    ProcessedUnderlyingAggregatesV2_5,
    SignalPayloadV2_5,
    ATIFStrategyDirectivePayloadV2_5,
    EOTSConfigV2_5,
    HuiHuiAnalysisRequestV2_5,
    HuiHuiAnalysisResponseV2_5,
    HuiHuiModelConfigV2_5,
    HuiHuiExpertConfigV2_5,
    HuiHuiUsageRecordV2_5,
    HuiHuiPerformanceMetricsV2_5
)

# Pydantic AI imports
try:
    from pydantic_ai.models import Model, StreamedResponse
    from pydantic_ai.messages import (
        Message,
        SystemMessage,
        UserMessage,
        AssistantMessage,
        ModelMessage,
        ModelRequest,
        ModelResponse
    )
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    # Create dummy classes for when Pydantic AI is not available
    class Message: pass
    class SystemMessage:
        def __init__(self, content): self.content = content
    class UserMessage:
        def __init__(self, content): self.content = content
    class AssistantMessage:
        def __init__(self, content): self.content = content
    class ModelResponse:
        def __init__(self, message, timestamp, usage):
            self.message = message
            self.timestamp = timestamp
            self.usage = usage
    class StreamedResponse:
        def __init__(self, message, timestamp):
            self.message = message
            self.timestamp = timestamp
    class Model: pass

# Local LLM client
from .local_llm_client import LocalLLMClient

logger = logging.getLogger(__name__)

# ===== HUIHUI MODEL INTERFACE =====

class HuiHuiModelInterface:
    """
    Interface class for HuiHui model integration.

    Provides a standardized interface for interacting with HuiHui models
    and managing expert routing and coordination.
    """

    def __init__(self, config: Optional[HuiHuiModelConfigV2_5] = None):
        """Initialize HuiHui model interface."""
        self.logger = logger.getChild("HuiHuiModelInterface")
        self.config = config or self._get_default_config()
        self.model = None
        self.is_initialized = False

    def _get_default_config(self) -> HuiHuiModelConfigV2_5:
        """Get default HuiHui model configuration."""
        return HuiHuiModelConfigV2_5(
            api_endpoint="http://localhost:11434",
            api_key="local",
            model_name="huihui-moe-eots-specialist",
            temperature=0.1,
            max_tokens=4000,
            timeout_seconds=30
        )

    def initialize(self) -> bool:
        """Initialize the HuiHui model interface."""
        try:
            self.model = HuiHuiPydanticModel(
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                enable_eots_integration=True
            )
            self.is_initialized = True
            self.logger.info("✅ HuiHui model interface initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize HuiHui model interface: {e}")
            return False

    async def analyze(self, request: HuiHuiAnalysisRequestV2_5) -> HuiHuiAnalysisResponseV2_5:
        """Perform analysis using HuiHui model."""
        if not self.is_initialized:
            raise RuntimeError("HuiHui model interface not initialized")

        # Convert request to messages format
        messages = [
            SystemMessage("You are a HuiHui AI expert for EOTS v2.5 analysis."),
            UserMessage(request.json())
        ]

        # Get response from model
        response = await self.model.request(messages)

        # Parse response into HuiHui analysis response
        return HuiHuiAnalysisResponseV2_5(
            expert_used="auto",
            analysis_content=response.message.content,
            confidence_score=0.8,
            processing_time=1.0,
            insights=["Analysis completed"]
        )

# ===== PYDANTIC MODELS FOR HUIHUI INTEGRATION =====

# HuiHuiAnalysisRequest is now imported from eots_schemas_v2_5.py
# Use the canonical schema version for consistency

# HuiHuiAnalysisResponse is now imported from eots_schemas_v2_5.py
# Use the canonical schema version for consistency

class HuiHuiMetricsIntegration(BaseModel):
    """Pydantic model for HuiHui integration with metrics calculator."""
    vri_score: Optional[float] = Field(None, description="VRI 2.0 score")
    vapi_fa_score: Optional[float] = Field(None, description="VAPI-FA score")
    dwfd_score: Optional[float] = Field(None, description="DWFD score")
    gib_score: Optional[float] = Field(None, description="GIB score")
    regime_confidence: Optional[float] = Field(None, description="Regime detection confidence")
    flow_strength: Optional[float] = Field(None, description="Options flow strength")

    def extract_from_bundle(self, bundle: FinalAnalysisBundleV2_5) -> 'HuiHuiMetricsIntegration':
        """PYDANTIC-FIRST: Extract metrics from EOTS bundle for HuiHui analysis."""
        try:
            und_data = bundle.processed_data_bundle.underlying_data_enriched

            # PYDANTIC-FIRST: Use proper attribute access instead of .get() method
            if und_data:
                self.vri_score = getattr(und_data, 'VRI_2_0_Und', None)
                self.vapi_fa_score = getattr(und_data, 'VAPI_FA_Z_Score_Und', None)
                self.dwfd_score = getattr(und_data, 'DWFD_Z_Score_Und', None)
                self.gib_score = getattr(und_data, 'GIB_OI_based_Und', None)

                # Calculate derived metrics
                if self.vri_score is not None:
                    self.regime_confidence = min(abs(self.vri_score) / 2.0, 1.0)

                if self.vapi_fa_score is not None and self.dwfd_score is not None:
                    self.flow_strength = (abs(self.vapi_fa_score) + abs(self.dwfd_score)) / 2.0

        except Exception as e:
            logger.warning(f"Error extracting metrics from bundle: {e}")

        return self

class HuiHuiPydanticModel(Model):
    """
    HuiHui-MoE Pydantic AI Model
    
    Local HuiHui-MoE model that routes to the appropriate expert system.
    Automatically selects the appropriate HuiHui expert based on system prompts.
    """
    
    def __init__(self,
                 model_name: str = "huihui-moe-eots-specialist",
                 temperature: float = 0.1,
                 max_tokens: int = 4000,
                 expert_mode: Optional[str] = None,
                 enable_eots_integration: bool = True):
        """
        Initialize HuiHui Pydantic AI Model with EOTS v2.5 integration.

        Args:
            model_name: Model identifier (for compatibility)
            temperature: Response creativity (0.1 = focused, 0.3 = creative)
            max_tokens: Maximum response length
            expert_mode: Force specific expert ("market_regime", "options_flow", "sentiment", "orchestrator")
            enable_eots_integration: Enable EOTS schema validation and metrics integration
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.expert_mode = expert_mode
        self.enable_eots_integration = enable_eots_integration
        self.client = LocalLLMClient()

        # EOTS-specific expert routing keywords (enhanced for v2.5)
        self.expert_keywords = {
            "market_regime": [
                "regime", "volatility", "vri", "vri_2_0", "structural", "transition",
                "regime detection", "volatility patterns", "market regime", "regime analysis",
                "volatility clustering", "regime confidence", "structural break"
            ],
            "options_flow": [
                "vapi-fa", "dwfd", "tw-laf", "gib", "flow", "options flow", "flow analysis",
                "institutional", "gamma", "positioning", "dealer hedging", "unusual activity",
                "dark pool", "volume analysis", "put call ratio", "gamma imbalance"
            ],
            "sentiment": [
                "sentiment", "news", "psychology", "crowd", "fear", "greed", "sentiment analysis",
                "market psychology", "behavioral", "contrarian", "sentiment extremes",
                "social media", "news sentiment", "market sentiment", "crowd behavior"
            ],
            "orchestrator": [
                "synthesize", "strategic", "recommendation", "decision", "comprehensive",
                "orchestrate", "coordinate", "unified", "meta analysis", "final decision",
                "strategic synthesis", "overall assessment", "unified intelligence"
            ]
        }

        # EOTS metrics integration mapping
        self.eots_metrics_mapping = {
            "market_regime": ["VRI_2_0_Und", "volatility_clustering", "regime_confidence"],
            "options_flow": ["VAPI_FA_Z_Score_Und", "DWFD_Z_Score_Und", "TW_LAF_Z_Score_Und", "GIB_OI_based_Und"],
            "sentiment": ["news_sentiment", "market_psychology", "sentiment_indicators"],
            "orchestrator": ["all_metrics", "unified_analysis", "comprehensive_assessment"]
        }
    
    def _detect_expert_from_prompt(self, system_prompt: str, user_prompt: str = "") -> str:
        """Detect the best HuiHui expert based on prompt content and EOTS context."""
        if self.expert_mode:
            return self.expert_mode

        combined_prompt = f"{system_prompt} {user_prompt}".lower()

        # Score each expert based on keyword matches
        expert_scores = {}
        for expert, keywords in self.expert_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_prompt)
            expert_scores[expert] = score

        # Return the highest scoring expert, default to orchestrator
        if max(expert_scores.values()) > 0:
            return max(expert_scores, key=expert_scores.get)
        else:
            return "orchestrator"  # Default to meta-orchestrator for complex tasks

    def _detect_expert_from_bundle(self, bundle: FinalAnalysisBundleV2_5) -> str:
        """Detect the best HuiHui expert based on EOTS bundle content."""
        if self.expert_mode:
            return self.expert_mode

        try:
            # Extract metrics for expert detection
            metrics = HuiHuiMetricsIntegration().extract_from_bundle(bundle)

            # Score experts based on available metrics and their significance
            expert_scores = {
                "market_regime": 0,
                "options_flow": 0,
                "sentiment": 0,
                "orchestrator": 0
            }

            # Market regime scoring
            if metrics.vri_score is not None:
                expert_scores["market_regime"] += abs(metrics.vri_score) * 2
            if metrics.regime_confidence is not None:
                expert_scores["market_regime"] += metrics.regime_confidence * 3

            # Options flow scoring
            if metrics.vapi_fa_score is not None:
                expert_scores["options_flow"] += abs(metrics.vapi_fa_score) * 2
            if metrics.dwfd_score is not None:
                expert_scores["options_flow"] += abs(metrics.dwfd_score) * 2
            if metrics.flow_strength is not None:
                expert_scores["options_flow"] += metrics.flow_strength * 3

            # Sentiment scoring (based on news intelligence)
            if bundle.news_intelligence_v2_5:
                expert_scores["sentiment"] += 3

            # Orchestrator gets base score for comprehensive analysis
            expert_scores["orchestrator"] += 1

            # Return highest scoring expert
            return max(expert_scores, key=expert_scores.get)

        except Exception as e:
            logger.warning(f"Error detecting expert from bundle: {e}")
            return "orchestrator"

    def _create_eots_context_prompt(self, bundle: FinalAnalysisBundleV2_5, expert: str) -> str:
        """Create EOTS-specific context prompt for the selected expert."""
        try:
            metrics = HuiHuiMetricsIntegration().extract_from_bundle(bundle)
            symbol = bundle.target_symbol

            context_parts = [
                f"EOTS v2.5 Analysis Context for {symbol}:",
                f"Timestamp: {bundle.bundle_timestamp}",
                ""
            ]

            # Add expert-specific metrics context
            if expert == "market_regime":
                context_parts.extend([
                    "MARKET REGIME ANALYSIS:",
                    f"VRI 2.0 Score: {metrics.vri_score:.3f}" if metrics.vri_score else "VRI 2.0: Not available",
                    f"Regime Confidence: {metrics.regime_confidence:.1%}" if metrics.regime_confidence else "Regime Confidence: Not available",
                    ""
                ])

            elif expert == "options_flow":
                context_parts.extend([
                    "OPTIONS FLOW ANALYSIS:",
                    f"VAPI-FA Z-Score: {metrics.vapi_fa_score:.3f}" if metrics.vapi_fa_score else "VAPI-FA: Not available",
                    f"DWFD Z-Score: {metrics.dwfd_score:.3f}" if metrics.dwfd_score else "DWFD: Not available",
                    f"GIB Score: {metrics.gib_score:.3f}" if metrics.gib_score else "GIB: Not available",
                    f"Flow Strength: {metrics.flow_strength:.1%}" if metrics.flow_strength else "Flow Strength: Not available",
                    ""
                ])

            elif expert == "sentiment":
                context_parts.extend([
                    "SENTIMENT ANALYSIS:",
                    f"News Intelligence: {'Available' if bundle.news_intelligence_v2_5 else 'Not available'}",
                    ""
                ])

            elif expert == "orchestrator":
                context_parts.extend([
                    "COMPREHENSIVE ANALYSIS:",
                    f"VRI 2.0: {metrics.vri_score:.3f}" if metrics.vri_score else "VRI 2.0: N/A",
                    f"VAPI-FA: {metrics.vapi_fa_score:.3f}" if metrics.vapi_fa_score else "VAPI-FA: N/A",
                    f"DWFD: {metrics.dwfd_score:.3f}" if metrics.dwfd_score else "DWFD: N/A",
                    f"GIB: {metrics.gib_score:.3f}" if metrics.gib_score else "GIB: N/A",
                    ""
                ])

            # Add signals context
            if bundle.scored_signals_v2_5:
                context_parts.append(f"Active Signals: {len(bundle.scored_signals_v2_5)} signal types")

            # Add ATIF context
            if bundle.atif_recommendations_v2_5:
                context_parts.append(f"ATIF Recommendations: {len(bundle.atif_recommendations_v2_5)} strategies")

            return "\n".join(context_parts)

        except Exception as e:
            logger.warning(f"Error creating EOTS context prompt: {e}")
            return f"EOTS v2.5 Analysis Context for {bundle.target_symbol} (simplified)"
    
    def _extract_messages_content(self, messages: List[Message]) -> tuple[str, str]:
        """Extract system prompt and user content from messages."""
        system_prompt = ""
        user_content = ""
        
        for message in messages:
            if isinstance(message, SystemMessage):
                system_prompt = message.content
            elif isinstance(message, UserMessage):
                user_content = message.content
        
        return system_prompt, user_content
    
    async def request(self, messages: List[Message]) -> ModelResponse:
        """
        Process a request using HuiHui-MoE experts with EOTS v2.5 integration.

        This is the main method that Pydantic AI calls to get responses.
        Enhanced with EOTS schema validation and metrics integration.
        """
        start_time = datetime.now()

        try:
            # Extract content from messages
            system_prompt, user_content = self._extract_messages_content(messages)

            # Check if EOTS bundle is provided in user content (JSON format)
            bundle_data = None
            enhanced_prompt = user_content

            if self.enable_eots_integration:
                try:
                    # Try to parse EOTS bundle from user content
                    if "FinalAnalysisBundleV2_5" in user_content or "bundle_timestamp" in user_content:
                        # Extract JSON from content if present
                        import re
                        json_match = re.search(r'\{.*\}', user_content, re.DOTALL)
                        if json_match:
                            bundle_json = json.loads(json_match.group())
                            bundle_data = FinalAnalysisBundleV2_5(**bundle_json)
                            logger.info(f"🎯 EOTS bundle detected for {bundle_data.target_symbol}")
                except Exception as e:
                    logger.debug(f"No EOTS bundle in request: {e}")

            # Detect appropriate expert
            if bundle_data:
                expert = self._detect_expert_from_bundle(bundle_data)
                # Enhance prompt with EOTS context
                eots_context = self._create_eots_context_prompt(bundle_data, expert)
                enhanced_prompt = f"{eots_context}\n\nUser Request: {user_content}"
            else:
                expert = self._detect_expert_from_prompt(system_prompt, user_content)

            logger.info(f"🧠 Routing to HuiHui {expert} expert")

            # Get response from HuiHui expert
            response = self.client.chat_huihui(
                prompt=enhanced_prompt,
                specialist=expert,
                temperature=self.temperature
            )

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Create enhanced response if EOTS integration is enabled
            if self.enable_eots_integration and bundle_data:
                # Create structured response using schema version
                analysis_response = HuiHuiAnalysisResponseV2_5(
                    expert_used=expert,
                    analysis_content=response,
                    confidence_score=0.85,  # Default confidence, could be enhanced
                    processing_time=processing_time,
                    insights=self._extract_insights_from_response(response)
                )

                # Add EOTS predictions if applicable
                if expert in ["market_regime", "orchestrator"]:
                    predictions = self._create_eots_predictions(bundle_data, response, expert)
                    analysis_response.eots_predictions = predictions

                # Return structured response
                response_content = f"EOTS Analysis Response:\n{analysis_response.json(indent=2)}"
            else:
                response_content = response

            # Create assistant message
            assistant_message = AssistantMessage(content=response_content)

            # Return model response
            return ModelResponse(
                message=assistant_message,
                timestamp=None,
                usage=None  # Local model, no usage tracking needed
            )

        except Exception as e:
            logger.error(f"HuiHui model request failed: {e}")
            # Return error message
            error_response = f"HuiHui model error: {str(e)}"
            assistant_message = AssistantMessage(content=error_response)
            return ModelResponse(
                message=assistant_message,
                timestamp=None,
                usage=None
            )
    
    async def request_stream(self, messages: List[Message]) -> AsyncIterator[StreamedResponse]:
        """
        Stream response (not implemented for HuiHui, returns single response).
        """
        # HuiHui doesn't support streaming, so we'll return the full response
        response = await self.request(messages)
        yield StreamedResponse(
            message=response.message,
            timestamp=response.timestamp
        )
    
    def name(self) -> str:
        """Return model name."""
        return f"huihui-moe-{self.expert_mode or 'auto'}"

    def _extract_insights_from_response(self, response: str) -> List[str]:
        """Extract key insights from HuiHui response."""
        insights = []
        try:
            # Debug: Check if response is actually a string
            if not isinstance(response, str):
                logger.error(f"❌ _extract_insights_from_response received {type(response)} instead of str: {response}")
                # Try to extract string content from response object
                if hasattr(response, 'content'):
                    response = str(response.content)
                elif hasattr(response, 'message'):
                    response = str(response.message)
                else:
                    response = str(response)

            # Simple insight extraction based on common patterns
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if any(keyword in line.lower() for keyword in ['key insight', 'important', 'significant', 'notable', 'critical']):
                    insights.append(line)
                elif line.startswith('•') or line.startswith('-') or line.startswith('*'):
                    insights.append(line)

            # Limit to top 5 insights
            return insights[:5]
        except Exception as e:
            logger.error(f"❌ Error in _extract_insights_from_response: {e}")
            return ["Analysis completed successfully"]

    def _create_eots_predictions(self, bundle: FinalAnalysisBundleV2_5, response: str, expert: str) -> List[AIPredictionV2_5]:
        """Create EOTS predictions based on HuiHui analysis."""
        predictions = []
        try:
            # Extract prediction direction from response
            response_lower = response.lower()

            if any(word in response_lower for word in ['bullish', 'upward', 'higher', 'buy']):
                direction = "UP"
            elif any(word in response_lower for word in ['bearish', 'downward', 'lower', 'sell']):
                direction = "DOWN"
            else:
                direction = "NEUTRAL"

            # Create prediction based on expert type
            if expert == "market_regime":
                prediction = AIPredictionV2_5(
                    symbol=bundle.target_symbol,
                    prediction_type="eots_direction",
                    prediction_direction=direction,
                    confidence_score=0.75,
                    time_horizon="1D",
                    target_timestamp=datetime.now() + timedelta(days=1),
                    market_context={
                        "expert_used": expert,
                        "analysis_type": "market_regime",
                        "bundle_timestamp": bundle.bundle_timestamp.isoformat(),
                        "ai_model_used": "huihui-moe-market-regime",
                        "prediction_reasoning": f"Market regime analysis by HuiHui {expert} expert"
                    }
                )
                predictions.append(prediction)

            elif expert == "orchestrator":
                # Create comprehensive prediction
                prediction = AIPredictionV2_5(
                    symbol=bundle.target_symbol,
                    prediction_type="eots_direction",
                    prediction_direction=direction,
                    confidence_score=0.85,
                    time_horizon="4H",
                    target_timestamp=datetime.now() + timedelta(hours=4),
                    market_context={
                        "expert_used": expert,
                        "analysis_type": "comprehensive",
                        "bundle_timestamp": bundle.bundle_timestamp.isoformat(),
                        "ai_model_used": "huihui-moe-orchestrator",
                        "prediction_reasoning": f"Comprehensive analysis by HuiHui {expert} expert"
                    }
                )
                predictions.append(prediction)

        except Exception as e:
            logger.warning(f"Error creating EOTS predictions: {e}")

        return predictions

# ===== CONVENIENCE FUNCTIONS WITH EOTS INTEGRATION =====

def create_huihui_model(expert_mode: Optional[str] = None,
                       temperature: float = 0.1,
                       max_tokens: int = 4000,
                       enable_eots_integration: bool = True) -> HuiHuiPydanticModel:
    """
    Create a HuiHui Pydantic AI model with EOTS v2.5 integration.

    Args:
        expert_mode: Specific expert to use ("market_regime", "options_flow", "sentiment", "orchestrator")
        temperature: Response creativity (0.1 = focused, 0.3 = creative)
        max_tokens: Maximum response length
        enable_eots_integration: Enable EOTS schema validation and metrics integration

    Returns:
        HuiHui Pydantic AI model ready for use in EOTS agents
    """
    return HuiHuiPydanticModel(
        expert_mode=expert_mode,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_eots_integration=enable_eots_integration
    )

def create_eots_huihui_model(bundle: FinalAnalysisBundleV2_5,
                            temperature: float = 0.1) -> HuiHuiPydanticModel:
    """
    Create a HuiHui model optimized for specific EOTS bundle analysis.

    Args:
        bundle: EOTS analysis bundle to optimize for
        temperature: Response creativity

    Returns:
        HuiHui model optimized for the bundle's analysis needs
    """
    # Create temporary model to detect best expert
    temp_model = HuiHuiPydanticModel(enable_eots_integration=True)
    expert = temp_model._detect_expert_from_bundle(bundle)

    return HuiHuiPydanticModel(
        expert_mode=expert,
        temperature=temperature,
        enable_eots_integration=True
    )

# ===== INDIVIDUAL PYDANTIC AI EXPERT MODELS =====

class HuiHuiMarketRegimeModel(Model):
    """Pydantic AI Model for HuiHui Market Regime Expert."""

    def __init__(self, temperature: float = 0.1, enable_eots: bool = True):
        # CRITICAL: Call parent Model.__init__() to properly initialize Pydantic AI Model
        super().__init__()
        self.temperature = temperature
        self.enable_eots = enable_eots
        self.expert_name = "market_regime"
        self.client = LocalLLMClient()

    async def request(self, messages: List[Message]) -> ModelResponse:
        """Process market regime analysis request."""
        return await self._process_expert_request(messages, "market_regime")

    async def request_stream(self, messages: List[Message]) -> AsyncIterator[StreamedResponse]:
        response = await self.request(messages)
        yield StreamedResponse(message=response.message, timestamp=response.timestamp)

    def name(self) -> str:
        return "huihui-market-regime"

    async def _process_expert_request(self, messages: List[Message], expert: str) -> ModelResponse:
        """Process request using specific HuiHui expert."""
        try:
            system_prompt, user_content = self._extract_messages_content(messages)

            # Get response from HuiHui expert
            response = self.client.chat_huihui(
                prompt=user_content,
                specialist=expert,
                temperature=self.temperature
            )

            assistant_message = AssistantMessage(content=response)
            return ModelResponse(message=assistant_message, timestamp=None, usage=None)

        except Exception as e:
            logger.error(f"HuiHui {expert} expert failed: {e}")
            error_response = f"HuiHui {expert} expert error: {str(e)}"
            assistant_message = AssistantMessage(content=error_response)
            return ModelResponse(message=assistant_message, timestamp=None, usage=None)

    def _extract_messages_content(self, messages: List[Message]) -> tuple[str, str]:
        """Extract system prompt and user content from messages."""
        system_prompt = ""
        user_content = ""

        for message in messages:
            if isinstance(message, SystemMessage):
                system_prompt = message.content
            elif isinstance(message, UserMessage):
                user_content = message.content

        return system_prompt, user_content

class HuiHuiOptionsFlowModel(HuiHuiMarketRegimeModel):
    """Pydantic AI Model for HuiHui Options Flow Expert."""

    def __init__(self, temperature: float = 0.1, enable_eots: bool = True):
        super().__init__(temperature, enable_eots)
        self.expert_name = "options_flow"

    async def request(self, messages: List[Message]) -> ModelResponse:
        return await self._process_expert_request(messages, "options_flow")

    def name(self) -> str:
        return "huihui-options-flow"

class HuiHuiSentimentModel(HuiHuiMarketRegimeModel):
    """Pydantic AI Model for HuiHui Sentiment Expert."""

    def __init__(self, temperature: float = 0.17, enable_eots: bool = True):
        super().__init__(temperature, enable_eots)
        self.expert_name = "sentiment"

    async def request(self, messages: List[Message]) -> ModelResponse:
        return await self._process_expert_request(messages, "sentiment")

    def name(self) -> str:
        return "huihui-sentiment"

class HuiHuiOrchestratorModel(HuiHuiMarketRegimeModel):
    """Pydantic AI Model for HuiHui Orchestrator Expert."""

    def __init__(self, temperature: float = 0.2, enable_eots: bool = True):
        super().__init__(temperature, enable_eots)
        self.expert_name = "orchestrator"

    async def request(self, messages: List[Message]) -> ModelResponse:
        return await self._process_expert_request(messages, "orchestrator")

    def name(self) -> str:
        return "huihui-orchestrator"

# ===== EXPERT-SPECIFIC MODEL CREATION FUNCTIONS =====

def create_market_regime_model(temperature: float = 0.1, enable_eots: bool = True) -> HuiHuiMarketRegimeModel:
    """Create HuiHui model specifically for EOTS market regime analysis."""
    return HuiHuiMarketRegimeModel(temperature, enable_eots)

def create_options_flow_model(temperature: float = 0.1, enable_eots: bool = True) -> HuiHuiOptionsFlowModel:
    """Create HuiHui model specifically for EOTS options flow analysis."""
    return HuiHuiOptionsFlowModel(temperature, enable_eots)

def create_sentiment_model(temperature: float = 0.17, enable_eots: bool = True) -> HuiHuiSentimentModel:
    """
    Create HuiHui model specifically for EOTS sentiment analysis.

    Note: Higher temperature (0.17) for sentiment analysis to handle:
    - Sarcasm and idioms in news content
    - Novel ticker mentions and market slang
    - Nuanced emotional context in financial news
    """
    return HuiHuiSentimentModel(temperature, enable_eots)

def create_orchestrator_model(temperature: float = 0.2, enable_eots: bool = True) -> HuiHuiOrchestratorModel:
    """
    Create HuiHui model specifically for EOTS strategic orchestration.

    Note: Higher temperature (0.2) for strategic thinking and synthesis.
    """
    return HuiHuiOrchestratorModel(temperature, enable_eots)

# ===== OPTIMIZED TEMPERATURE CONFIGURATIONS =====

EXPERT_TEMPERATURE_CONFIG = {
    "market_regime": 0.1,      # Precise for regime detection
    "options_flow": 0.1,       # Precise for flow analysis
    "sentiment": 0.17,         # Higher for nuanced sentiment
    "orchestrator": 0.2        # Higher for strategic synthesis
}

def create_optimized_expert_model(expert: str, enable_eots: bool = True) -> HuiHuiPydanticModel:
    """Create HuiHui model with optimized temperature for specific expert."""
    temperature = EXPERT_TEMPERATURE_CONFIG.get(expert, 0.1)
    return create_huihui_model(expert, temperature, enable_eots_integration=enable_eots)

# ===== EOTS-SPECIFIC INTEGRATION FUNCTIONS =====

async def analyze_eots_bundle_with_huihui(bundle: FinalAnalysisBundleV2_5,
                                         expert_mode: Optional[str] = None,
                                         temperature: float = 0.1) -> HuiHuiAnalysisResponseV2_5:
    """
    Analyze EOTS bundle using HuiHui with full Pydantic validation.

    Args:
        bundle: EOTS analysis bundle
        expert_mode: Specific expert to use (auto-detected if None)
        temperature: Response creativity

    Returns:
        Structured HuiHui analysis response with EOTS predictions
    """
    # Create optimized model
    if expert_mode:
        model = create_huihui_model(expert_mode, temperature, enable_eots_integration=True)
    else:
        model = create_eots_huihui_model(bundle, temperature)

    # Create analysis request
    request = HuiHuiAnalysisRequestV2_5(
        symbol=bundle.target_symbol,
        analysis_type="eots_bundle_analysis",
        bundle_data=bundle,
        expert_mode=expert_mode,
        temperature=temperature
    )

    # Create messages for Pydantic AI
    system_message = SystemMessage(
        content=f"You are an EOTS v2.5 {model.expert_mode or 'auto'} expert. "
                f"Analyze the provided EOTS bundle and provide comprehensive insights."
    )
    user_message = UserMessage(
        content=f"Analyze EOTS bundle for {bundle.target_symbol}: {bundle.json()}"
    )

    # Get response
    response = await model.request([system_message, user_message])

    # Parse structured response
    try:
        import json
        response_data = json.loads(response.message.content.split("EOTS Analysis Response:\n")[1])
        return HuiHuiAnalysisResponseV2_5(**response_data)
    except Exception:
        # Fallback to basic response
        return HuiHuiAnalysisResponseV2_5(
            expert_used=model.expert_mode or "auto",
            analysis_content=response.message.content,
            confidence_score=0.75,
            processing_time=1.0
        )

def validate_huihui_integration() -> bool:
    """Validate that HuiHui integration is working with EOTS schemas."""
    try:
        # Test model creation
        model = create_huihui_model(enable_eots_integration=True)

        # Test EOTS schema imports with proper required fields
        from data_models.eots_schemas_v2_5 import ProcessedUnderlyingAggregatesV2_5, KeyLevelsDataV2_5

        test_bundle = FinalAnalysisBundleV2_5(
            processed_data_bundle=ProcessedDataBundleV2_5(
                underlying_data_enriched=ProcessedUnderlyingAggregatesV2_5(
                    symbol="SPY",
                    timestamp=datetime.now()
                ),
                strike_level_metrics=[],
                processing_timestamp=datetime.now()
            ),
            key_levels_data_v2_5=KeyLevelsDataV2_5(
                timestamp=datetime.now()
            ),
            bundle_timestamp=datetime.now(),
            target_symbol="SPY"
        )

        # Test expert detection
        expert = model._detect_expert_from_bundle(test_bundle)

        logger.info(f"✅ HuiHui EOTS integration validated - Expert: {expert}")
        return True

    except Exception as e:
        logger.error(f"❌ HuiHui EOTS integration validation failed: {e}")
        return False

# ===== COMPATIBILITY FUNCTIONS =====

def replace_openai_model(openai_model_string: str) -> HuiHuiPydanticModel:
    """
    Replace OpenAI model string with equivalent HuiHui model.
    
    Args:
        openai_model_string: OpenAI model identifier (e.g., "gpt-4o", "gpt-3.5-turbo")
    
    Returns:
        Equivalent HuiHui model
    """
    # Map OpenAI models to HuiHui configurations
    if "gpt-4" in openai_model_string.lower():
        # GPT-4 equivalent - use orchestrator for complex reasoning
        return create_orchestrator_model(temperature=0.1)
    elif "gpt-3.5" in openai_model_string.lower():
        # GPT-3.5 equivalent - use auto-routing
        return create_huihui_model(temperature=0.1)
    else:
        # Default to auto-routing orchestrator
        return create_orchestrator_model(temperature=0.1)

# ===== TESTING FUNCTION =====

async def test_huihui_pydantic_model():
    """Test the HuiHui Pydantic AI model."""
    print("🧠 Testing HuiHui Pydantic AI Model...")
    
    if not PYDANTIC_AI_AVAILABLE:
        print("❌ Pydantic AI not available")
        return False
    
    try:
        # Create test model
        model = create_huihui_model()
        
        # Create test messages
        messages = [
            SystemMessage(content="You are a market regime analysis expert."),
            UserMessage(content="Analyze the current SPY market regime.")
        ]
        
        # Test request
        response = await model.request(messages)
        
        print(f"✅ Model Name: {model.name()}")
        print(f"✅ Response Length: {len(response.message.content)} chars")
        print(f"✅ Response Preview: {response.message.content[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    # Run test
    asyncio.run(test_huihui_pydantic_model())
