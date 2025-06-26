"""Dashboard Modes Package for EOTS v2.5"""

# Import all dashboard mode modules
from .main_dashboard_display_v2_5 import create_layout as main_dashboard_create_layout
from .flow_mode_display_v2_5 import *
from .structure_mode_display_v2_5 import *
from .time_decay_mode_display_v2_5 import *
from .volatility_mode_display_v2_5 import *
from .advanced_flow_mode_v2_5 import *
from .ai_dashboard.ai_dashboard_display_v2_5 import create_layout as ai_dashboard_create_layout

# Create a class wrapper for compatibility
class MainDashboardDisplayV2_5:
    """Wrapper class for main dashboard display functionality"""

    @staticmethod
    def create_layout(bundle, config):
        return main_dashboard_create_layout(bundle, config)

class AIDashboardDisplayV2_5:
    """Wrapper class for AI dashboard display functionality"""

    @staticmethod
    def create_layout(bundle, config, db_manager=None):
        return ai_dashboard_create_layout(bundle, config, db_manager)

__all__ = [
    'MainDashboardDisplayV2_5',
    'main_dashboard_create_layout',
    'AIDashboardDisplayV2_5',
    'ai_dashboard_create_layout'
]