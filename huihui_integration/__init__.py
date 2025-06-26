"""
HuiHui Integration - Advanced AI Expert System for EOTS v2.5
============================================================

This module provides a comprehensive AI expert system with 3 specialized HuiHui experts:
- Market Regime Expert: VRI analysis, volatility patterns, regime detection
- Options Flow Expert: VAPI-FA, DWFD, institutional flow analysis  
- Sentiment Expert: News analysis, market psychology, sentiment indicators

The 4th expert (Meta-Orchestrator) is integrated as its_orchestrator_v2_5.py in core_analytics_engine.

Features:
- Individual expert development and learning
- Dedicated databases per expert
- Advanced learning algorithms
- Cross-expert knowledge sharing
- Performance tracking and optimization
- Supabase-only data storage (no SQLite)

Author: EOTS v2.5 AI Architecture Division
"""

from typing import Optional, Dict, Any, List
import logging

# Version information
__version__ = "2.5.0"
__author__ = "EOTS v2.5 AI Architecture Division"
__description__ = "Advanced HuiHui AI Expert System for Elite Options Trading"

# Configure logging for HuiHui integration
logger = logging.getLogger(__name__)

# Expert types (3 specialists + orchestrator in core engine)
HUIHUI_EXPERTS = {
    "market_regime": "Market Regime & Volatility Analysis Expert",
    "options_flow": "Options Flow & Institutional Behavior Expert", 
    "sentiment": "Sentiment & News Intelligence Expert"
    # Note: "orchestrator" is its_orchestrator_v2_5.py in core_analytics_engine
}

# System status
_system_initialized = False
_expert_status = {
    "market_regime": False,
    "options_flow": False,
    "sentiment": False
}

def get_system_info() -> Dict[str, Any]:
    """Get HuiHui integration system information."""
    return {
        "version": __version__,
        "experts_available": list(HUIHUI_EXPERTS.keys()),
        "system_initialized": _system_initialized,
        "expert_status": _expert_status.copy(),
        "description": __description__
    }

def is_system_ready() -> bool:
    """Check if HuiHui integration system is ready."""
    return _system_initialized and all(_expert_status.values())

# Import guards for optional dependencies
try:
    from .core.model_interface import HuiHuiModelInterface
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    logger.warning("HuiHui core interface not available")

try:
    from .orchestrator_bridge.expert_coordinator import ExpertCoordinator
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False
    logger.warning("HuiHui orchestrator bridge not available")

try:
    from .monitoring.usage_monitor import HuiHuiUsageMonitor
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    logger.warning("HuiHui monitoring not available")

# Lazy loading functions
def get_expert_coordinator():
    """Get the expert coordinator for managing the 3 HuiHui specialists."""
    if not BRIDGE_AVAILABLE:
        raise ImportError("Expert coordinator not available")
    from .orchestrator_bridge.expert_coordinator import get_coordinator
    return get_coordinator()

def get_usage_monitor():
    """Get the HuiHui usage monitor for performance tracking."""
    if not MONITORING_AVAILABLE:
        raise ImportError("Usage monitor not available")
    from .monitoring.usage_monitor import get_usage_monitor
    return get_usage_monitor()

def get_market_regime_expert():
    """Get the Market Regime Expert."""
    try:
        from .experts.market_regime.expert import MarketRegimeExpert
        return MarketRegimeExpert()
    except ImportError:
        logger.error("Market Regime Expert not available")
        return None

def get_options_flow_expert():
    """Get the Options Flow Expert."""
    try:
        from .experts.options_flow.expert import OptionsFlowExpert
        return OptionsFlowExpert()
    except ImportError:
        logger.error("Options Flow Expert not available")
        return None

def get_sentiment_expert():
    """Get the Sentiment Expert."""
    try:
        from .experts.sentiment.expert import SentimentExpert
        return SentimentExpert()
    except ImportError:
        logger.error("Sentiment Expert not available")
        return None

# System initialization
async def initialize_huihui_system(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Initialize the complete HuiHui integration system.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        bool: True if initialization successful
    """
    global _system_initialized, _expert_status
    
    try:
        logger.info("🧠 Initializing HuiHui Integration System v2.5...")
        
        # Initialize core components
        if CORE_AVAILABLE:
            logger.info("✅ Core interface available")
        
        # Initialize monitoring
        if MONITORING_AVAILABLE:
            monitor = get_usage_monitor()
            logger.info("✅ Usage monitoring initialized")
        
        # Initialize expert coordinator
        if BRIDGE_AVAILABLE:
            coordinator = get_expert_coordinator()
            logger.info("✅ Expert coordinator initialized")
        
        # Initialize individual experts
        market_expert = get_market_regime_expert()
        if market_expert:
            _expert_status["market_regime"] = True
            logger.info("✅ Market Regime Expert initialized")
        
        options_expert = get_options_flow_expert()
        if options_expert:
            _expert_status["options_flow"] = True
            logger.info("✅ Options Flow Expert initialized")
        
        sentiment_expert = get_sentiment_expert()
        if sentiment_expert:
            _expert_status["sentiment"] = True
            logger.info("✅ Sentiment Expert initialized")
        
        _system_initialized = True
        logger.info("🚀 HuiHui Integration System v2.5 initialized successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize HuiHui system: {e}")
        return False

# Convenience functions for EOTS integration
def quick_market_analysis(symbol: str, data: Dict[str, Any]) -> Optional[str]:
    """Quick market regime analysis using HuiHui Market Regime Expert."""
    try:
        expert = get_market_regime_expert()
        if expert:
            return expert.analyze_regime(symbol, data)
        return None
    except Exception as e:
        logger.error(f"Market analysis failed: {e}")
        return None

def quick_flow_analysis(symbol: str, data: Dict[str, Any]) -> Optional[str]:
    """Quick options flow analysis using HuiHui Options Flow Expert."""
    try:
        expert = get_options_flow_expert()
        if expert:
            return expert.analyze_flow(symbol, data)
        return None
    except Exception as e:
        logger.error(f"Flow analysis failed: {e}")
        return None

def quick_sentiment_analysis(symbol: str, data: Dict[str, Any]) -> Optional[str]:
    """Quick sentiment analysis using HuiHui Sentiment Expert."""
    try:
        expert = get_sentiment_expert()
        if expert:
            return expert.analyze_sentiment(symbol, data)
        return None
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return None

# Export main components
__all__ = [
    "HUIHUI_EXPERTS",
    "get_system_info",
    "is_system_ready", 
    "initialize_huihui_system",
    "get_expert_coordinator",
    "get_usage_monitor",
    "get_market_regime_expert",
    "get_options_flow_expert", 
    "get_sentiment_expert",
    "quick_market_analysis",
    "quick_flow_analysis",
    "quick_sentiment_analysis"
]
