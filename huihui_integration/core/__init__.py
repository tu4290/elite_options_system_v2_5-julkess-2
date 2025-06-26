"""
HuiHui Core - Base Components and Interfaces
============================================

Core components for the HuiHui AI expert system including:
- Base expert classes and interfaces
- Model communication protocols
- Inter-expert communication systems
- Core Pydantic models and schemas

Author: EOTS v2.5 AI Architecture Division
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Core component availability
CORE_COMPONENTS = {
    "base_expert": False,
    "model_interface": False,
    "expert_communication": False
}

def get_core_status() -> Dict[str, Any]:
    """Get status of core HuiHui components."""
    return {
        "components": CORE_COMPONENTS.copy(),
        "all_available": all(CORE_COMPONENTS.values())
    }

# Import guards for core components
try:
    from .base_expert import BaseHuiHuiExpert
    CORE_COMPONENTS["base_expert"] = True
except ImportError:
    logger.debug("Base expert class not available")

try:
    from .model_interface import HuiHuiModelInterface
    CORE_COMPONENTS["model_interface"] = True
except ImportError:
    logger.debug("Model interface not available")

try:
    from .expert_communication import ExpertCommunicationProtocol
    CORE_COMPONENTS["expert_communication"] = True
except ImportError:
    logger.debug("Expert communication not available")

__all__ = [
    "get_core_status",
    "CORE_COMPONENTS"
]
