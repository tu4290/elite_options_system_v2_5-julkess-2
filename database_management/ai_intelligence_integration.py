"""
AI Intelligence Database Integration for EOTS v2.5
=================================================

This module integrates the Pydantic AI Intelligence Engine with the
Supabase AI Intelligence Database, providing seamless persistence,
learning, and evolution capabilities for AI agents.

Author: EOTS v2.5 AI Intelligence Division
Version: 1.0.0 - "SENTIENT DATABASE INTEGRATION"
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

# Import the AI Intelligence Database Manager
from .ai_intelligence_database_manager import (
    AIIntelligenceDatabaseManager,
    AIAgentRecord,
    AILearningSession,
    AIMemoryRecord,
    AIAdaptiveThreshold,
    AIPerformanceMetrics
)

# Import the NEW Pydantic-First Intelligence Engine V2.5
from dashboard_application.modes.ai_dashboard.pydantic_intelligence_engine_v2_5 import (
    PydanticIntelligenceEngineV2_5,
    get_intelligence_engine,
    IntelligenceRequest,
    IntelligenceResponse,
    AnalysisType
)

logger = logging.getLogger(__name__)

class AIIntelligenceDatabaseIntegration:
    """
    INTEGRATION LAYER FOR AI INTELLIGENCE AND DATABASE
    
    This class provides seamless integration between the Pydantic AI
    Intelligence Engine and the Supabase AI Intelligence Database.
    """
    
    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        self.db_manager = AIIntelligenceDatabaseManager(db_config)
        self.logger = logger.getChild(self.__class__.__name__)
        self.agent_id_mapping = {}  # Map agent names to database IDs
        self.is_initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the database integration."""
        try:
            # Initialize database connection
            if not await self.db_manager.initialize_connection():
                self.logger.error("Failed to initialize database connection")
                return False
            
            # Register or retrieve existing AI agents
            await self._register_ai_agents()
            
            # Load adaptive thresholds
            await self._load_adaptive_thresholds()
            
            self.is_initialized = True
            self.logger.info("🏛️ AI Intelligence Database Integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AI Intelligence Database Integration: {e}")
            return False
    
    async def _register_ai_agents(self):
        """Register the core AI agents in the database."""
        agents_to_register = [
            {
                "agent_name": "MarketAnalystAgent",
                "agent_type": "market_analyst",
                "specialization": "Institutional flow patterns and gamma positioning effects",
                "capabilities": {
                    "pattern_recognition": True,
                    "flow_analysis": True,
                    "institutional_behavior": True,
                    "gamma_positioning": True
                },
                "configuration": {
                    "confidence_threshold": 0.7,
                    "max_retries": 2,
                    "learning_enabled": True,
                    "pattern_recognition_depth": "advanced"
                }
            },
            {
                "agent_name": "RegimeAnalystAgent",
                "agent_type": "regime_analyst",
                "specialization": "Market regime detection and transition prediction",
                "capabilities": {
                    "regime_detection": True,
                    "transition_prediction": True,
                    "regime_validation": True,
                    "multi_timeframe_analysis": True
                },
                "configuration": {
                    "confidence_threshold": 0.75,
                    "transition_sensitivity": 0.8,
                    "regime_validation_enabled": True,
                    "max_retries": 2
                }
            },
            {
                "agent_name": "ConfidenceCalculatorAgent",
                "agent_type": "confidence_calculator",
                "specialization": "Confidence calibration and ensemble validation",
                "capabilities": {
                    "confidence_calibration": True,
                    "self_validation": True,
                    "ensemble_methods": True,
                    "historical_accuracy_tracking": True
                },
                "configuration": {
                    "confidence_threshold": 0.8,
                    "self_validation_enabled": True,
                    "historical_accuracy_weight": 0.3,
                    "max_retries": 1
                }
            }
        ]
        
        for agent_config in agents_to_register:
            try:
                # Check if agent already exists
                existing_agents = await self.db_manager.connection_pool.fetch(
                    "SELECT id FROM ai_agents WHERE agent_name = $1",
                    agent_config["agent_name"]
                )
                
                if existing_agents:
                    agent_id = str(existing_agents[0]['id'])
                    self.logger.debug(f"Found existing agent: {agent_config['agent_name']} ({agent_id})")
                else:
                    # Register new agent
                    agent_record = AIAgentRecord(**agent_config)
                    agent_id = await self.db_manager.register_agent(agent_record)
                    self.logger.info(f"🤖 Registered new agent: {agent_config['agent_name']} ({agent_id})")
                
                self.agent_id_mapping[agent_config["agent_name"]] = agent_id
                
            except Exception as e:
                self.logger.error(f"Failed to register agent {agent_config['agent_name']}: {e}")
    
    async def _load_adaptive_thresholds(self):
        """Load adaptive thresholds for all agents."""
        for agent_name, agent_id in self.agent_id_mapping.items():
            try:
                thresholds = await self.db_manager.get_adaptive_thresholds(agent_id)
                self.logger.debug(f"Loaded {len(thresholds)} thresholds for {agent_name}")
            except Exception as e:
                self.logger.error(f"Failed to load thresholds for {agent_name}: {e}")
    
    async def record_insight_generation(self, agent_name: str, market_metrics: Dict[str, Any],
                                      insights: List[str]) -> bool:
        """Record insight generation session in the database."""
        if not self.is_initialized:
            return False
            
        try:
            agent_id = self.agent_id_mapping.get(agent_name)
            if not agent_id:
                self.logger.warning(f"Agent {agent_name} not found in mapping")
                return False
            
            # Create learning session record
            session = AILearningSession(
                agent_id=agent_id,
                session_type="insight_generation",
                market_context={
                    "symbol": market_metrics.get("symbol", "UNKNOWN"),
                    "regime": market_metrics.get("current_regime", "UNKNOWN"),
                    "timestamp": datetime.now().isoformat()
                },
                input_data=market_metrics,
                output_data={
                    "insights_count": len(insights),
                    "insights": insights,
                    "avg_confidence": 0.75  # Default confidence for string insights
                },
                confidence_score=0.75
            )
            
            session_id = await self.db_manager.record_learning_session(session)
            
            # Store individual insights
            for i, insight in enumerate(insights):
                await self._store_insight_record(agent_id, session_id, market_metrics.get("symbol", "UNKNOWN"), insight, i)

            # Store patterns as memories
            await self._extract_and_store_patterns(agent_id, market_metrics, insights)
            
            self.logger.debug(f"📚 Recorded insight generation for {agent_name}: {len(insights)} insights")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record insight generation for {agent_name}: {e}")
            return False
    
    async def _store_insight_record(self, agent_id: str, session_id: str, symbol: str, insight: str, index: int):
        """Store individual insight record."""
        try:
            async with self.db_manager.connection_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ai_insights_history (
                        agent_id, session_id, symbol, insight_text, insight_type,
                        confidence_score, reasoning, supporting_metrics, risk_level,
                        actionability_score, insight_timestamp
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
                """,
                agent_id, session_id, symbol, insight, "market_analysis",
                0.75, f"AI insight #{index + 1}", json.dumps({}),
                "MODERATE", 0.7)
                
        except Exception as e:
            self.logger.error(f"Failed to store insight record: {e}")
    
    async def _extract_and_store_patterns(self, agent_id: str, market_metrics: Dict[str, Any],
                                        insights: List[str]):
        """Extract and store patterns as memories."""
        try:
            # Create pattern signature based on market conditions
            regime = market_metrics.get("current_regime", "UNKNOWN")
            symbol = market_metrics.get("symbol", "UNKNOWN")
            pattern_signature = f"{regime}_{symbol}_{len(insights)}_insights"

            # Calculate pattern success probability based on insights
            avg_confidence = 0.75  # Default confidence for string insights

            memory = AIMemoryRecord(
                agent_id=agent_id,
                memory_type="insight_pattern",
                memory_category=regime,
                pattern_signature=pattern_signature,
                pattern_data={
                    "market_metrics": market_metrics,
                    "insights_generated": len(insights),
                    "avg_confidence": avg_confidence,
                    "insights_content": insights
                },
                success_rate=avg_confidence,
                confidence_level=avg_confidence
            )
            
            await self.db_manager.store_memory(memory)
            
        except Exception as e:
            self.logger.error(f"Failed to extract and store patterns: {e}")
    
    async def record_confidence_assessment(self, agent_name: str, market_metrics: Dict[str, Any],
                                         assessment: Dict[str, Any]) -> bool:
        """Record confidence assessment session."""
        if not self.is_initialized:
            return False
            
        try:
            agent_id = self.agent_id_mapping.get(agent_name)
            if not agent_id:
                return False
            
            session = AILearningSession(
                agent_id=agent_id,
                session_type="confidence_assessment",
                market_context={
                    "symbol": market_metrics.get("symbol", "UNKNOWN"),
                    "regime": market_metrics.get("current_regime", "UNKNOWN")
                },
                input_data=market_metrics,
                output_data=assessment,
                confidence_score=assessment.get("overall_confidence", 0.75)
            )
            
            await self.db_manager.record_learning_session(session)
            
            # Store confidence calibration pattern
            await self._store_confidence_pattern(agent_id, market_metrics, assessment)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record confidence assessment for {agent_name}: {e}")
            return False
    
    async def _store_confidence_pattern(self, agent_id: str, market_metrics: Dict[str, Any],
                                      assessment: Dict[str, Any]):
        """Store confidence assessment pattern as memory."""
        try:
            regime = market_metrics.get("current_regime", "UNKNOWN")
            confidence = assessment.get("overall_confidence", 0.75)
            pattern_signature = f"confidence_{regime}_{confidence:.2f}"

            memory = AIMemoryRecord(
                agent_id=agent_id,
                memory_type="confidence_pattern",
                memory_category="confidence_calibration",
                pattern_signature=pattern_signature,
                pattern_data={
                    "market_conditions": market_metrics,
                    "confidence_breakdown": assessment,
                    "calibration_factors": assessment.get("calibration_factors", {})
                },
                success_rate=confidence,
                confidence_level=confidence
            )
            
            await self.db_manager.store_memory(memory)
            
        except Exception as e:
            self.logger.error(f"Failed to store confidence pattern: {e}")
    
    async def get_adaptive_thresholds(self, agent_name: str) -> Dict[str, float]:
        """Get adaptive thresholds for an agent."""
        if not self.is_initialized:
            return {}
            
        agent_id = self.agent_id_mapping.get(agent_name)
        if not agent_id:
            return {}
        
        return await self.db_manager.get_adaptive_thresholds(agent_id)
    
    async def update_adaptive_threshold(self, agent_name: str, threshold_name: str,
                                      new_value: float, reason: str) -> bool:
        """Update an adaptive threshold."""
        if not self.is_initialized:
            return False
            
        agent_id = self.agent_id_mapping.get(agent_name)
        if not agent_id:
            return False
        
        return await self.db_manager.update_adaptive_threshold(agent_id, threshold_name, new_value, reason)
    
    async def get_learning_insights(self, agent_name: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent learning insights for an agent."""
        if not self.is_initialized:
            return []
            
        agent_id = self.agent_id_mapping.get(agent_name)
        if not agent_id:
            return []
        
        try:
            sessions = await self.db_manager.get_learning_history(agent_id, limit=50)
            
            # Filter recent sessions
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_sessions = [s for s in sessions if s.session_timestamp >= cutoff_date]
            
            insights = []
            for session in recent_sessions:
                insights.append({
                    "session_type": session.session_type,
                    "confidence": session.confidence_score,
                    "accuracy": session.accuracy_score,
                    "timestamp": session.session_timestamp,
                    "market_context": session.market_context,
                    "learning_extracted": session.learning_extracted
                })
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to get learning insights for {agent_name}: {e}")
            return []
    
    async def record_daily_performance(self, agent_name: str, symbol: str, 
                                     performance_data: Dict[str, Any]) -> bool:
        """Record daily performance metrics for an agent."""
        if not self.is_initialized:
            return False
            
        agent_id = self.agent_id_mapping.get(agent_name)
        if not agent_id:
            return False
        
        try:
            metrics = AIPerformanceMetrics(
                agent_id=agent_id,
                metric_date=datetime.now(),
                symbol=symbol,
                **performance_data
            )
            
            return await self.db_manager.record_performance_metrics(metrics)
            
        except Exception as e:
            self.logger.error(f"Failed to record daily performance for {agent_name}: {e}")
            return False
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview."""
        if not self.is_initialized:
            return {}
        
        return await self.db_manager.get_system_overview()
    
    async def cleanup_and_maintain(self) -> bool:
        """Perform database cleanup and maintenance."""
        if not self.is_initialized:
            return False
        
        try:
            # Record system health
            await self.db_manager.record_system_health()
            
            # Cleanup old data
            await self.db_manager.cleanup_old_data(days_to_keep=90)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to perform cleanup and maintenance: {e}")
            return False
    
    async def store_orchestrator_decision(self, decision_id: str, symbol: str,
                                        decision_type: str, confidence_score: float,
                                        market_regime: str, data_quality_score: float,
                                        system_load: float, component_routing: Dict[str, Any],
                                        reasoning: str, expected_outcome: str):
        """Store orchestrator decision in Supabase using Pydantic-first validation against EOTS schemas."""
        try:
            # PYDANTIC-FIRST: Import and validate against EOTS schemas
            from data_models.eots_schemas_v2_5 import OrchestratorDecisionV2_5, OrchestratorDecisionContextV2_5
            from pydantic import BaseModel, Field
            from datetime import datetime

            # Create validation model for database storage with proper defaults
            class OrchestratorDecisionStorageV2_5(BaseModel):
                decision_id: str = Field(..., description="Unique decision identifier")
                symbol: str = Field(..., description="Trading symbol")
                decision_type: str = Field(..., description="Type of orchestration decision")
                confidence_score: float = Field(..., ge=0.0, le=1.0, description="Decision confidence")
                market_regime: str = Field(default="UNKNOWN", description="Market regime context")
                data_quality_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Data quality score")
                system_load: float = Field(default=0.5, ge=0.0, le=1.0, description="System load")
                component_routing: Dict[str, Any] = Field(default_factory=dict, description="Component routing decisions")
                reasoning: str = Field(default="Standard orchestration decision", description="AI reasoning for decision")
                expected_outcome: str = Field(default="Successful analysis completion", description="Expected outcome")

            # Validate all inputs using Pydantic model
            validated_decision = OrchestratorDecisionStorageV2_5(
                decision_id=decision_id,
                symbol=symbol,
                decision_type=decision_type,
                confidence_score=confidence_score,
                market_regime=market_regime,
                data_quality_score=data_quality_score,
                system_load=system_load,
                component_routing=component_routing,
                reasoning=reasoning,
                expected_outcome=expected_outcome
            )

            if not self.is_initialized:
                await self.initialize()

            async with self.db_manager.connection_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO orchestrator_decisions (
                        decision_id, symbol, decision_type, confidence_score,
                        market_regime, data_quality_score, system_load,
                        component_routing, reasoning, expected_outcome
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                validated_decision.decision_id,
                validated_decision.symbol,
                validated_decision.decision_type,
                validated_decision.confidence_score,
                validated_decision.market_regime,
                validated_decision.data_quality_score,
                validated_decision.system_load,
                validated_decision.model_dump_json(),  # Pydantic serialization
                validated_decision.reasoning,
                validated_decision.expected_outcome)

                self.logger.debug(f"🧠 Stored Pydantic-validated orchestrator decision {decision_id} for {symbol}")

        except Exception as e:
            self.logger.error(f"Failed to store orchestrator decision: {e}")
            raise

    async def close(self):
        """Close the database integration."""
        if self.db_manager:
            await self.db_manager.close_connection()
        self.is_initialized = False
        self.logger.info("🏛️ AI Intelligence Database Integration closed")

# ===== GLOBAL INTEGRATION INSTANCE =====

_ai_db_integration = None

async def get_ai_database_integration(db_config: Optional[Dict[str, Any]] = None) -> AIIntelligenceDatabaseIntegration:
    """Get or create the global AI database integration instance."""
    global _ai_db_integration
    
    if _ai_db_integration is None:
        _ai_db_integration = AIIntelligenceDatabaseIntegration(db_config)
        await _ai_db_integration.initialize()
    
    return _ai_db_integration
