"""
Orchestrator Bridge - Connection to ITS Orchestrator
====================================================

Bridge system that connects the its_orchestrator_v2_5.py (HuiHui Expert #4 - Meta-Orchestrator)
with the 3 specialized HuiHui experts:

1. Market Regime Expert
2. Options Flow Expert  
3. Sentiment Expert

This bridge enables the Meta-Orchestrator to coordinate and synthesize insights
from all 3 specialists while maintaining clean separation of concerns.

Architecture:
- its_orchestrator_v2_5.py = HuiHui Expert #4 (Meta-Orchestrator)
- Bridge coordinates the 3 specialists
- Clean communication protocols
- Unified response synthesis

Author: EOTS v2.5 AI Architecture Division
"""

from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Bridge configuration
BRIDGE_CONFIG = {
    "name": "HuiHui Orchestrator Bridge",
    "version": "2.5.0",
    "experts_managed": ["market_regime", "options_flow", "sentiment"],
    "orchestrator_location": "core_analytics_engine.its_orchestrator_v2_5",
    "communication_protocol": "async_pydantic",
    "response_synthesis": "enabled"
}

# Bridge status
_bridge_initialized = False
_experts_connected = {
    "market_regime": False,
    "options_flow": False,
    "sentiment": False
}
_orchestrator_connected = False

def get_bridge_status() -> Dict[str, Any]:
    """Get orchestrator bridge status."""
    return {
        "config": BRIDGE_CONFIG,
        "status": {
            "bridge_initialized": _bridge_initialized,
            "experts_connected": _experts_connected.copy(),
            "orchestrator_connected": _orchestrator_connected,
            "all_systems_ready": _bridge_initialized and _orchestrator_connected and all(_experts_connected.values())
        }
    }

# Import guards
try:
    from .expert_coordinator import ExpertCoordinator
    COORDINATOR_AVAILABLE = True
except ImportError:
    COORDINATOR_AVAILABLE = False
    logger.debug("Expert Coordinator not available")

try:
    from .its_integration import ITSOrchestratorIntegration
    ITS_INTEGRATION_AVAILABLE = True
except ImportError:
    ITS_INTEGRATION_AVAILABLE = False
    logger.debug("ITS Orchestrator Integration not available")

def get_expert_coordinator():
    """Get Expert Coordinator instance."""
    if not COORDINATOR_AVAILABLE:
        raise ImportError("Expert Coordinator not available")
    from .expert_coordinator import get_coordinator
    return get_coordinator()

def get_its_integration():
    """Get ITS Orchestrator Integration instance."""
    if not ITS_INTEGRATION_AVAILABLE:
        raise ImportError("ITS Orchestrator Integration not available")
    from .its_integration import get_its_integration
    return get_its_integration()

async def initialize_bridge() -> bool:
    """Initialize the orchestrator bridge system."""
    global _bridge_initialized, _experts_connected, _orchestrator_connected
    
    try:
        logger.info("🌉 Initializing HuiHui Orchestrator Bridge...")
        
        # Initialize expert coordinator
        if COORDINATOR_AVAILABLE:
            coordinator = get_expert_coordinator()
            await coordinator.initialize()
            
            # Check expert connections
            _experts_connected["market_regime"] = await coordinator.is_expert_available("market_regime")
            _experts_connected["options_flow"] = await coordinator.is_expert_available("options_flow")
            _experts_connected["sentiment"] = await coordinator.is_expert_available("sentiment")
            
            logger.info(f"✅ Expert connections: {_experts_connected}")
        
        # Initialize ITS integration
        if ITS_INTEGRATION_AVAILABLE:
            its_integration = get_its_integration()
            _orchestrator_connected = await its_integration.test_connection()
            logger.info(f"✅ ITS Orchestrator connection: {_orchestrator_connected}")
        
        _bridge_initialized = True
        logger.info("🚀 HuiHui Orchestrator Bridge initialized successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize orchestrator bridge: {e}")
        return False

# Convenience functions for its_orchestrator_v2_5.py
async def coordinate_expert_analysis(symbol: str, data: Dict[str, Any], expert_preferences: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Coordinate analysis across the 3 HuiHui specialists.
    
    This function is called by its_orchestrator_v2_5.py to get insights
    from the 3 specialists before making final orchestration decisions.
    """
    try:
        if not COORDINATOR_AVAILABLE:
            logger.warning("Expert coordinator not available")
            return {}
        
        coordinator = get_expert_coordinator()
        return await coordinator.coordinate_analysis(symbol, data, expert_preferences)
        
    except Exception as e:
        logger.error(f"Failed to coordinate expert analysis: {e}")
        return {}

async def synthesize_expert_insights(expert_responses: Dict[str, Any]) -> str:
    """
    Synthesize insights from multiple experts into unified analysis.
    
    This function helps its_orchestrator_v2_5.py combine insights from
    the 3 specialists into a coherent strategic recommendation.
    """
    try:
        if not COORDINATOR_AVAILABLE:
            logger.warning("Expert coordinator not available")
            return "Expert synthesis not available"
        
        coordinator = get_expert_coordinator()
        return await coordinator.synthesize_insights(expert_responses)
        
    except Exception as e:
        logger.error(f"Failed to synthesize expert insights: {e}")
        return f"Synthesis error: {str(e)}"

__all__ = [
    "BRIDGE_CONFIG",
    "get_bridge_status",
    "get_expert_coordinator",
    "get_its_integration",
    "initialize_bridge",
    "coordinate_expert_analysis",
    "synthesize_expert_insights",
    "COORDINATOR_AVAILABLE",
    "ITS_INTEGRATION_AVAILABLE"
]
