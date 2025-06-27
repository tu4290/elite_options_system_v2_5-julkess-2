"""
AI Hub Metrics Module - Row 2 Metric Containers v2.5
====================================================

This module contains the 3 metric containers for Row 2:
- Flow Intelligence Container (VAPI-FA, DWFD, TW-LAF, Transition Gauge)
- Volatility & Gamma Container (VRI 2.0, A-DAG, GIB, SVR)
- Custom Formulas Container (LWPAI, VABAI, AOFM, LIDB, TPDLF)

Author: EOTS v2.5 Development Team
Version: 2.5.1 (Modular)
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from dash import dcc, html
import plotly.graph_objects as go

# EOTS Schema imports - Pydantic-first validation
from data_models.eots_schemas_v2_5 import (
    FinalAnalysisBundleV2_5,
    ProcessedDataBundleV2_5,
    ProcessedUnderlyingAggregatesV2_5,
    AdvancedOptionsMetricsV2_5
)

# Import existing components - preserve dependencies
from .components import (
    AI_COLORS, AI_TYPOGRAPHY, AI_SPACING, AI_EFFECTS,
    create_placeholder_card, get_card_style, create_clickable_title_with_info
)

from .visualizations import (
    create_confidence_meter, create_regime_transition_gauge
)

from .compliance_decorators_v2_5 import track_compliance

logger = logging.getLogger(__name__)

def create_metric_gauge(value: float, title: str, range_min: float = -3.0, range_max: float = 3.0) -> go.Figure:
    """Create a standardized metric gauge for consistent display."""
    try:
        # Normalize value to 0-1 range for gauge
        normalized_value = (value - range_min) / (range_max - range_min)
        normalized_value = max(0, min(1, normalized_value))  # Clamp to 0-1
        
        # Determine color based on value
        if abs(value) >= 2.0:
            color = AI_COLORS['danger']
        elif abs(value) >= 1.5:
            color = AI_COLORS['warning']
        elif abs(value) >= 1.0:
            color = AI_COLORS['primary']
        else:
            color = AI_COLORS['success']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 12, 'color': 'white'}},
            number={'font': {'size': 16, 'color': color}},
            gauge={
                'axis': {'range': [range_min, range_max], 'tickcolor': 'white', 'tickfont': {'size': 8}},
                'bar': {'color': color, 'thickness': 0.7},
                'bgcolor': 'rgba(0, 0, 0, 0.1)',
                'borderwidth': 1,
                'bordercolor': 'rgba(255, 255, 255, 0.3)',
                'steps': [
                    {'range': [range_min, -1], 'color': 'rgba(255, 71, 87, 0.2)'},
                    {'range': [-1, 1], 'color': 'rgba(107, 207, 127, 0.2)'},
                    {'range': [1, range_max], 'color': 'rgba(255, 71, 87, 0.2)'}
                ]
            }
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            font={'color': 'white'},
            height=120,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating metric gauge for {title}: {str(e)}")
        return go.Figure()

@track_compliance("flow_intelligence_container", "Flow Intelligence Container")
def create_flow_intelligence_container(bundle_data: FinalAnalysisBundleV2_5, symbol: str) -> html.Div:
    """
    Create Flow Intelligence container with VAPI-FA, DWFD, TW-LAF, and Transition Gauge.
    
    Args:
        bundle_data: Validated FinalAnalysisBundleV2_5
        symbol: Trading symbol
        
    Returns:
        html.Div: Flow intelligence container
    """
    try:
        # Extract data using Pydantic model access
        processed_data = bundle_data.processed_data_bundle
        enriched_data = processed_data.underlying_data_enriched if processed_data else None
        
        if not enriched_data:
            return create_placeholder_card("📊 Flow Intelligence", "No flow data available")
        
        # Extract flow metrics with safe access
        vapi_fa = getattr(enriched_data, 'vapi_fa_z_score_und', 0.0) or 0.0
        dwfd = getattr(enriched_data, 'dwfd_z_score_und', 0.0) or 0.0
        tw_laf = getattr(enriched_data, 'tw_laf_z_score_und', 0.0) or 0.0
        
        # Calculate transition probability (simplified logic)
        transition_prob = min(abs(vapi_fa) + abs(dwfd) + abs(tw_laf), 3.0) / 3.0 * 0.8
        
        return html.Div([
            html.Div([
                create_clickable_title_with_info(
                    "📊 Flow Intelligence",
                    "flow_intelligence",
                    "Advanced options flow metrics: VAPI-FA, DWFD, TW-LAF, and regime transition probability"
                )
            ], className="container-header"),
            
            html.Div([
                # VAPI-FA Gauge
                html.Div([
                    dcc.Graph(
                        figure=create_metric_gauge(vapi_fa, "VAPI-FA"),
                        config={'displayModeBar': False}
                    )
                ], className="metric-gauge"),
                
                # DWFD Gauge
                html.Div([
                    dcc.Graph(
                        figure=create_metric_gauge(dwfd, "DWFD"),
                        config={'displayModeBar': False}
                    )
                ], className="metric-gauge"),
                
                # TW-LAF Gauge
                html.Div([
                    dcc.Graph(
                        figure=create_metric_gauge(tw_laf, "TW-LAF"),
                        config={'displayModeBar': False}
                    )
                ], className="metric-gauge"),
                
                # Transition Gauge
                html.Div([
                    dcc.Graph(
                        figure=create_regime_transition_gauge(transition_prob, 0.7, 0.6),
                        config={'displayModeBar': False}
                    )
                ], className="metric-gauge")
                
            ], className="metrics-grid")
            
        ], style=get_card_style('primary'))
        
    except Exception as e:
        logger.error(f"Error creating flow intelligence container: {str(e)}")
        return create_placeholder_card("📊 Flow Intelligence", f"Error: {str(e)}")

@track_compliance("volatility_gamma_container", "Volatility & Gamma Container")
def create_volatility_gamma_container(bundle_data: FinalAnalysisBundleV2_5, symbol: str) -> html.Div:
    """
    Create Volatility & Gamma container with VRI 2.0, A-DAG, GIB, SVR.
    
    Args:
        bundle_data: Validated FinalAnalysisBundleV2_5
        symbol: Trading symbol
        
    Returns:
        html.Div: Volatility & gamma container
    """
    try:
        # Extract data using Pydantic model access
        processed_data = bundle_data.processed_data_bundle
        enriched_data = processed_data.underlying_data_enriched if processed_data else None
        
        if not enriched_data:
            return create_placeholder_card("📈 Volatility & Gamma", "No volatility data available")
        
        # Extract volatility metrics with safe access and normalization
        vri_2_0 = (getattr(enriched_data, 'vri_2_0_und', 0.0) or 0.0) / 5000  # Normalize for gauge
        a_dag = (getattr(enriched_data, 'a_dag_total_und', 0.0) or 0.0) / 50000  # Normalize for gauge
        gib = (getattr(enriched_data, 'gib_oi_based_und', 0.0) or 0.0) / 100000  # Normalize for gauge
        
        # Calculate SVR (simplified - would need more data for real calculation)
        svr = min(abs(vri_2_0) * 0.5, 2.0)
        
        return html.Div([
            html.Div([
                create_clickable_title_with_info(
                    "📈 Volatility & Gamma",
                    "volatility_gamma",
                    "Volatility and gamma exposure metrics: VRI 2.0, A-DAG, GIB, SVR"
                )
            ], className="container-header"),
            
            html.Div([
                # VRI 2.0 Gauge
                html.Div([
                    dcc.Graph(
                        figure=create_metric_gauge(vri_2_0, "VRI 2.0", -2.0, 2.0),
                        config={'displayModeBar': False}
                    )
                ], className="metric-gauge"),
                
                # A-DAG Gauge
                html.Div([
                    dcc.Graph(
                        figure=create_metric_gauge(a_dag, "A-DAG", -2.0, 2.0),
                        config={'displayModeBar': False}
                    )
                ], className="metric-gauge"),
                
                # GIB Gauge
                html.Div([
                    dcc.Graph(
                        figure=create_metric_gauge(gib, "GIB", -2.0, 2.0),
                        config={'displayModeBar': False}
                    )
                ], className="metric-gauge"),
                
                # SVR Gauge
                html.Div([
                    dcc.Graph(
                        figure=create_metric_gauge(svr, "SVR", 0.0, 3.0),
                        config={'displayModeBar': False}
                    )
                ], className="metric-gauge")
                
            ], className="metrics-grid")
            
        ], style=get_card_style('analysis'))
        
    except Exception as e:
        logger.error(f"Error creating volatility gamma container: {str(e)}")
        return create_placeholder_card("📈 Volatility & Gamma", f"Error: {str(e)}")

@track_compliance("custom_formulas_container", "Custom Formulas Container")
def create_custom_formulas_container(bundle_data: FinalAnalysisBundleV2_5, symbol: str) -> html.Div:
    """
    Create Custom Formulas container with LWPAI, VABAI, AOFM, LIDB, TPDLF.
    
    Args:
        bundle_data: Validated FinalAnalysisBundleV2_5
        symbol: Trading symbol
        
    Returns:
        html.Div: Custom formulas container
    """
    try:
        # Extract advanced options metrics using Pydantic model access
        processed_data = bundle_data.processed_data_bundle
        advanced_metrics = None
        
        # Try to get advanced metrics from processed data
        if processed_data and hasattr(processed_data, 'advanced_options_metrics_v2_5'):
            advanced_metrics = processed_data.advanced_options_metrics_v2_5
        
        # Default values if no advanced metrics available
        lwpai = 0.0
        vabai = 0.0
        aofm = 0.0
        lidb = 0.0
        tpdlf = 0.0
        
        if advanced_metrics and isinstance(advanced_metrics, AdvancedOptionsMetricsV2_5):
            lwpai = getattr(advanced_metrics, 'lwpai', 0.0) or 0.0
            vabai = getattr(advanced_metrics, 'vabai', 0.0) or 0.0
            aofm = getattr(advanced_metrics, 'aofm', 0.0) or 0.0
            lidb = getattr(advanced_metrics, 'lidb', 0.0) or 0.0
            # TPDLF might not be directly available, calculate from other metrics
            tpdlf = (lwpai + vabai) * 0.5  # Simplified calculation
        
        return html.Div([
            html.Div([
                create_clickable_title_with_info(
                    "🎯 Custom Formulas",
                    "custom_formulas",
                    "Your proprietary trading formulas: LWPAI, VABAI, AOFM, LIDB, TPDLF"
                )
            ], className="container-header"),
            
            html.Div([
                # LWPAI Gauge
                html.Div([
                    dcc.Graph(
                        figure=create_metric_gauge(lwpai, "LWPAI", -1.0, 1.0),
                        config={'displayModeBar': False}
                    )
                ], className="metric-gauge"),
                
                # VABAI Gauge
                html.Div([
                    dcc.Graph(
                        figure=create_metric_gauge(vabai, "VABAI", -1.0, 1.0),
                        config={'displayModeBar': False}
                    )
                ], className="metric-gauge"),
                
                # AOFM Gauge
                html.Div([
                    dcc.Graph(
                        figure=create_metric_gauge(aofm, "AOFM", -1.0, 1.0),
                        config={'displayModeBar': False}
                    )
                ], className="metric-gauge"),
                
                # LIDB Gauge
                html.Div([
                    dcc.Graph(
                        figure=create_metric_gauge(lidb, "LIDB", -1.0, 1.0),
                        config={'displayModeBar': False}
                    )
                ], className="metric-gauge")
                
            ], className="metrics-grid")
            
        ], style=get_card_style('success'))
        
    except Exception as e:
        logger.error(f"Error creating custom formulas container: {str(e)}")
        return create_placeholder_card("🎯 Custom Formulas", f"Error: {str(e)}")

