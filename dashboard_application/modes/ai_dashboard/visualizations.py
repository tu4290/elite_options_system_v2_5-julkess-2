"""
AI Dashboard Visualizations Module for EOTS v2.5
================================================

This module contains all chart and graph creation functions for the AI dashboard including:
- Market state visualizations
- Performance charts
- Confidence meters
- Radar/polygon charts
- Time series plots
- Gauge charts

Author: EOTS v2.5 Development Team
Version: 2.5.0
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from dashboard_application.utils_dashboard_v2_5 import (
    PLOTLY_TEMPLATE,
    create_empty_figure,
    add_timestamp_annotation,
    apply_dark_theme_template
)

from data_models.eots_schemas_v2_5 import (
    FinalAnalysisBundleV2_5,
    ProcessedDataBundleV2_5,
    ProcessedUnderlyingAggregatesV2_5
)

from .components import AI_COLORS, AI_TYPOGRAPHY, AI_SPACING, AI_EFFECTS

logger = logging.getLogger(__name__)

# ===== MARKET STATE VISUALIZATIONS =====

def create_market_state_visualization(bundle_data: FinalAnalysisBundleV2_5) -> go.Figure:
    """Create a visualization of current market state dynamics using REAL EOTS v2.5 metrics."""
    try:
        # Extract real metrics from processed data using Pydantic models
        processed_data = bundle_data.processed_data_bundle
        if not processed_data:
            # Return empty chart if no data
            return create_empty_figure("Market State Visualization", reason="No processed data available")

        metrics = processed_data.underlying_data_enriched.model_dump()

        # Extract strike data for structural metrics
        strike_data = processed_data.strike_level_data_with_metrics
        if strike_data:
            # Convert Pydantic models to DataFrame for calculations
            strike_df = pd.DataFrame([item.model_dump() for item in strike_data])
            if not strike_df.empty:
                a_dag_total = abs(strike_df.get('a_dag_strike', pd.Series([0])).sum())
                vri_2_0_avg = abs(strike_df.get('vri_2_0_strike', pd.Series([0])).mean())
            else:
                a_dag_total = 0
                vri_2_0_avg = 0
        else:
            a_dag_total = 0
            vri_2_0_avg = 0

        # Extract REAL Enhanced Flow Metrics and normalize for radar chart
        metrics_data = {
            'VAPI-FA Intensity': min(abs(metrics.get('vapi_fa_z_score_und', 0)) / 3.0, 1.0),
            'DWFD Smart Money': min(abs(metrics.get('dwfd_z_score_und', 0)) / 2.5, 1.0),
            'TW-LAF Conviction': min(abs(metrics.get('tw_laf_z_score_und', 0)) / 2.0, 1.0),
            'A-DAG Pressure': min(a_dag_total / 100000, 1.0),
            'VRI 2.0 Risk': min(vri_2_0_avg / 15000, 1.0),
            'GIB Imbalance': min(abs(metrics.get('gib_oi_based_und', 0)) / 150000, 1.0)
        }

        # Create radar chart
        categories = list(metrics_data.keys())
        values = list(metrics_data.values())

        fig = go.Figure()

        # Add radar trace
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Market State',
            line=dict(color=AI_COLORS['primary'], width=2),
            fillcolor=f"rgba(255, 217, 61, 0.3)"
        ))

        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    tickfont=dict(size=10, color='white')
                ),
                angularaxis=dict(
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    tickfont=dict(size=11, color='white')
                ),
                bgcolor='rgba(0, 0, 0, 0)'
            ),
            showlegend=False,
            title=dict(
                text="Market State Dynamics",
                x=0.5,
                font=dict(size=14, color='white')
            ),
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating market state visualization: {str(e)}")
        return create_empty_figure("Market State Visualization", reason=f"Error: {str(e)}")


def create_enhanced_market_state_visualization(bundle_data: FinalAnalysisBundleV2_5) -> go.Figure:
    """Create enhanced market state visualization with additional metrics."""
    try:
        # Extract metrics using Pydantic models
        processed_data = bundle_data.processed_data_bundle
        if not processed_data:
            return create_empty_figure("Enhanced Market State", reason="No processed data available")

        metrics = processed_data.underlying_data_enriched.model_dump()

        # Enhanced metrics for visualization
        enhanced_metrics = {
            'Flow Intensity': min(abs(metrics.get('vapi_fa_z_score_und', 0)) / 3.0, 1.0),
            'Smart Money': min(abs(metrics.get('dwfd_z_score_und', 0)) / 2.5, 1.0),
            'Conviction': min(abs(metrics.get('tw_laf_z_score_und', 0)) / 2.0, 1.0),
            'Volatility': min(abs(metrics.get('gib_oi_based_und', 0)) / 150000, 1.0),
            'Momentum': min(abs(metrics.get('vapi_fa_z_score_und', 0) + metrics.get('dwfd_z_score_und', 0)) / 5.0, 1.0),
            'Risk Level': min(abs(metrics.get('tw_laf_z_score_und', 0)) / 2.5, 1.0)
        }

        # Create enhanced radar chart
        categories = list(enhanced_metrics.keys())
        values = list(enhanced_metrics.values())

        fig = go.Figure()

        # Add main trace
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Current State',
            line=dict(color=AI_COLORS['primary'], width=3),
            fillcolor=f"rgba(255, 217, 61, 0.4)"
        ))

        # Add reference circle at 0.5
        reference_values = [0.5] * len(categories)
        fig.add_trace(go.Scatterpolar(
            r=reference_values,
            theta=categories,
            mode='lines',
            name='Reference',
            line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dash')
        ))

        # Update layout with enhanced styling
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    tickfont=dict(size=10, color='white'),
                    tickvals=[0.2, 0.4, 0.6, 0.8, 1.0]
                ),
                angularaxis=dict(
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    tickfont=dict(size=12, color='white', family='Arial Black')
                ),
                bgcolor='rgba(0, 0, 0, 0)'
            ),
            showlegend=False,
            title=dict(
                text="Enhanced Market Forces",
                x=0.5,
                font=dict(size=16, color='white', family='Arial Black')
            ),
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            height=350,
            margin=dict(l=30, r=30, t=50, b=30)
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating enhanced market state visualization: {str(e)}")
        return create_empty_figure("Enhanced Market State", reason=f"Error: {str(e)}")


def create_confidence_meter(confidence: float, title: str = "AI Confidence") -> go.Figure:
    """Create a confidence meter gauge chart."""
    try:
        # Determine color based on confidence level
        if confidence >= 0.8:
            color = AI_COLORS['success']
        elif confidence >= 0.6:
            color = AI_COLORS['warning']
        else:
            color = AI_COLORS['danger']

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 16, 'color': 'white'}},
            delta={'reference': 70, 'increasing': {'color': AI_COLORS['success']}},
            gauge={
                'axis': {'range': [None, 100], 'tickcolor': 'white'},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(255, 71, 87, 0.3)'},
                    {'range': [50, 80], 'color': 'rgba(255, 167, 38, 0.3)'},
                    {'range': [80, 100], 'color': 'rgba(107, 207, 127, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            font={'color': 'white'},
            height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating confidence meter: {str(e)}")
        return create_empty_figure("AI Confidence Meter", reason=f"Error: {str(e)}")


def create_confluence_gauge(confluence_score: float) -> go.Figure:
    """Create a confluence gauge visualization."""
    try:
        # Determine color and level
        if confluence_score >= 0.8:
            color = AI_COLORS['success']
            level = "Extreme"
        elif confluence_score >= 0.6:
            color = AI_COLORS['primary']
            level = "High"
        elif confluence_score >= 0.4:
            color = AI_COLORS['warning']
            level = "Moderate"
        else:
            color = AI_COLORS['danger']
            level = "Low"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confluence_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Confluence: {level}", 'font': {'size': 14, 'color': 'white'}},
            number={'font': {'size': 20, 'color': color}},
            gauge={
                'axis': {'range': [None, 100], 'tickcolor': 'white', 'tickfont': {'size': 10}},
                'bar': {'color': color, 'thickness': 0.8},
                'bgcolor': 'rgba(0, 0, 0, 0.1)',
                'borderwidth': 2,
                'bordercolor': 'rgba(255, 255, 255, 0.3)',
                'steps': [
                    {'range': [0, 25], 'color': 'rgba(255, 71, 87, 0.2)'},
                    {'range': [25, 50], 'color': 'rgba(255, 167, 38, 0.2)'},
                    {'range': [50, 75], 'color': 'rgba(255, 217, 61, 0.2)'},
                    {'range': [75, 100], 'color': 'rgba(107, 207, 127, 0.2)'}
                ]
            }
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            font={'color': 'white'},
            height=200,
            margin=dict(l=10, r=10, t=30, b=10)
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating confluence gauge: {str(e)}")
        return create_empty_figure("Confluence Gauge", reason=f"Error: {str(e)}")


def create_regime_transition_gauge(transition_prob: float, regime_confidence: float, confluence_score: float) -> go.Figure:
    """Create a regime transition gauge visualization for the 4th quadrant."""
    try:
        # Determine gauge styling based on transition probability
        if transition_prob >= 0.7:
            gauge_color = AI_COLORS['danger']
            gauge_level = "High Risk"
        elif transition_prob >= 0.4:
            gauge_color = AI_COLORS['warning']
            gauge_level = "Moderate Risk"
        else:
            gauge_color = AI_COLORS['success']
            gauge_level = "Low Risk"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=transition_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Transition Risk", 'font': {'size': 14, 'color': 'white'}},
            number={'font': {'size': 18, 'color': gauge_color}},
            gauge={
                'axis': {'range': [None, 100], 'tickcolor': 'white', 'tickfont': {'size': 9}},
                'bar': {'color': gauge_color, 'thickness': 0.7},
                'bgcolor': 'rgba(0, 0, 0, 0.1)',
                'borderwidth': 2,
                'bordercolor': 'rgba(255, 255, 255, 0.3)',
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(107, 207, 127, 0.2)'},  # Low risk - green
                    {'range': [30, 60], 'color': 'rgba(255, 167, 38, 0.2)'},   # Moderate risk - orange
                    {'range': [60, 100], 'color': 'rgba(255, 71, 87, 0.2)'}    # High risk - red
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 3},
                    'thickness': 0.75,
                    'value': 70  # Warning threshold
                }
            }
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            font={'color': 'white'},
            height=180,
            margin=dict(l=10, r=10, t=30, b=10)
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating regime transition gauge: {str(e)}")
        return create_empty_figure("Transition Gauge", reason=f"Error: {str(e)}")


def create_ai_performance_chart(performance_data: Dict[str, Any]) -> go.Figure:
    """Create AI performance tracking chart."""
    try:
        # Extract performance data
        dates = performance_data.get('dates', [])
        accuracy = performance_data.get('accuracy', [])
        confidence = performance_data.get('confidence', [])
        learning_curve = performance_data.get('learning_curve', [])

        if not dates or not accuracy:
            return create_empty_figure("AI Performance Chart", reason="No performance data available")

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('AI Performance Metrics', 'Learning Progression'),
            vertical_spacing=0.15,
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )

        # Add accuracy trace
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=accuracy,
                mode='lines+markers',
                name='Accuracy',
                line=dict(color=AI_COLORS['success'], width=3),
                marker=dict(size=6)
            ),
            row=1, col=1
        )

        # Add confidence trace
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=confidence,
                mode='lines+markers',
                name='Confidence',
                line=dict(color=AI_COLORS['primary'], width=2),
                marker=dict(size=4),
                yaxis='y2'
            ),
            row=1, col=1
        )

        # Add learning curve
        if learning_curve:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=learning_curve,
                    mode='lines',
                    name='Learning Curve',
                    line=dict(color=AI_COLORS['info'], width=2, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(66, 165, 245, 0.1)'
                ),
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color='white')
            ),
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='white'),
            margin=dict(l=40, r=40, t=60, b=40)
        )

        # Update axes
        fig.update_xaxes(gridcolor='rgba(255, 255, 255, 0.1)', tickfont=dict(color='white'))
        fig.update_yaxes(gridcolor='rgba(255, 255, 255, 0.1)', tickfont=dict(color='white'))

        return fig

    except Exception as e:
        logger.error(f"Error creating AI performance chart: {str(e)}")
        return create_empty_figure("AI Performance Chart", reason=f"Error: {str(e)}")


def create_pure_metrics_visualization(metrics: Dict[str, Any], symbol: str) -> go.Figure:
    """Create pure EOTS metrics visualization."""
    try:
        # Extract key metrics
        metric_names = []
        metric_values = []

        key_metrics = [
            ('VAPI-FA Z-Score', 'vapi_fa_z_score_und'),
            ('DWFD Z-Score', 'dwfd_z_score_und'),
            ('TW-LAF Z-Score', 'tw_laf_z_score_und'),
            ('GIB OI-Based', 'gib_oi_based_und'),
            ('A-DAG Total', 'a_dag_total_und'),
            ('VRI 2.0', 'vri_2_0_und')
        ]

        for name, key in key_metrics:
            value = metrics.get(key, 0)
            if value is not None:
                metric_names.append(name)
                metric_values.append(float(value))

        if not metric_names:
            return create_empty_figure("EOTS Metrics", reason="No metrics data available")

        # Create bar chart
        colors = [AI_COLORS['success'] if v > 0 else AI_COLORS['danger'] for v in metric_values]

        fig = go.Figure(data=[
            go.Bar(
                x=metric_names,
                y=metric_values,
                marker_color=colors,
                text=[f"{v:.2f}" for v in metric_values],
                textposition='auto',
                textfont=dict(color='white', size=10)
            )
        ])

        fig.update_layout(
            title=dict(
                text=f"Raw EOTS Metrics - {symbol}",
                x=0.5,
                font=dict(size=16, color='white')
            ),
            xaxis=dict(
                tickangle=45,
                tickfont=dict(color='white', size=10),
                gridcolor='rgba(255, 255, 255, 0.1)'
            ),
            yaxis=dict(
                tickfont=dict(color='white'),
                gridcolor='rgba(255, 255, 255, 0.1)',
                zeroline=True,
                zerolinecolor='rgba(255, 255, 255, 0.3)'
            ),
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            height=300,
            margin=dict(l=40, r=40, t=50, b=80)
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating pure metrics visualization: {str(e)}")
        return create_empty_figure("EOTS Metrics", reason=f"Error: {str(e)}")


def create_comprehensive_metrics_chart(metrics: Dict[str, Any], symbol: str) -> go.Figure:
    """Create comprehensive EOTS metrics chart with all tiers properly displayed."""
    try:
        # Define all EOTS metrics by tier
        tier_3_metrics = {
            'vapi_fa_z_score_und': 'VAPI-FA Z',
            'vapi_fa_raw_und': 'VAPI-FA Raw',
            'dwfd_z_score_und': 'DWFD Z',
            'dwfd_raw_und': 'DWFD Raw',
            'tw_laf_z_score_und': 'TW-LAF Z',
            'tw_laf_raw_und': 'TW-LAF Raw'
        }

        tier_2_metrics = {
            'gib_oi_based_und': 'GIB OI',
            'vri_2_0_und': 'VRI 2.0',
            'a_dag_total_und': 'A-DAG',
            'hp_eod_und': 'HP-EOD',
            'td_gib_und': 'TD-GIB'
        }

        tier_1_metrics = {
            'a_mspi_und': 'A-MSPI',
            'e_sdag_mult_und': 'E-SDAG',
            'a_sai_und': 'A-SAI',
            'a_ssi_und': 'A-SSI',
            'atr_und': 'ATR'
        }

        # Create single comprehensive bar chart
        fig = go.Figure()

        # Collect all metrics for display
        all_names = []
        all_values = []
        all_colors = []

        # Add Tier 3 metrics
        for metric_key, display_name in tier_3_metrics.items():
            value = metrics.get(metric_key, 0.0)
            if value is not None:
                all_names.append(display_name)
                all_values.append(float(value))
                all_colors.append('#ffc107')  # Warning color for Tier 3

        # Add Tier 2 metrics
        for metric_key, display_name in tier_2_metrics.items():
            value = metrics.get(metric_key, 0.0)
            if value is not None:
                all_names.append(display_name)
                all_values.append(float(value))
                all_colors.append('#007bff')  # Primary color for Tier 2

        # Add Tier 1 metrics
        for metric_key, display_name in tier_1_metrics.items():
            value = metrics.get(metric_key, 0.0)
            if value is not None:
                all_names.append(display_name)
                all_values.append(float(value))
                all_colors.append('#28a745')  # Success color for Tier 1

        if not all_names:
            return create_empty_figure("Comprehensive EOTS Metrics", reason="No metrics data available")

        # Create the comprehensive bar chart
        fig.add_trace(go.Bar(
            x=all_names,
            y=all_values,
            marker_color=all_colors,
            text=[f"{v:.3f}" for v in all_values],
            textposition='auto',
            textfont=dict(color='white', size=9),
            name='EOTS Metrics',
            hovertemplate='<b>%{x}</b><br>Value: %{y:.3f}<extra></extra>'
        ))

        # Add tier separators as vertical lines
        tier_3_count = len([v for v in tier_3_metrics.values() if metrics.get(list(tier_3_metrics.keys())[list(tier_3_metrics.values()).index(v)], 0) is not None])
        tier_2_count = len([v for v in tier_2_metrics.values() if metrics.get(list(tier_2_metrics.keys())[list(tier_2_metrics.values()).index(v)], 0) is not None])

        if tier_3_count > 0:
            fig.add_vline(x=tier_3_count - 0.5, line_dash="dash", line_color="rgba(255, 193, 7, 0.5)", line_width=2)
        if tier_2_count > 0:
            fig.add_vline(x=tier_3_count + tier_2_count - 0.5, line_dash="dash", line_color="rgba(0, 123, 255, 0.5)", line_width=2)

        # Add tier annotations
        if tier_3_count > 0:
            fig.add_annotation(
                x=tier_3_count / 2 - 0.5,
                y=max(all_values) * 1.1 if all_values else 1,
                text="TIER 3: ENHANCED FLOW",
                showarrow=False,
                font=dict(color='#ffc107', size=10, family="Arial Black"),
                bgcolor="rgba(255, 193, 7, 0.1)",
                bordercolor="#ffc107",
                borderwidth=1
            )

        if tier_2_count > 0:
            fig.add_annotation(
                x=tier_3_count + tier_2_count / 2 - 0.5,
                y=max(all_values) * 1.1 if all_values else 1,
                text="TIER 2: ADAPTIVE",
                showarrow=False,
                font=dict(color='#007bff', size=10, family="Arial Black"),
                bgcolor="rgba(0, 123, 255, 0.1)",
                bordercolor="#007bff",
                borderwidth=1
            )

        tier_1_start = tier_3_count + tier_2_count
        tier_1_count = len(all_names) - tier_1_start
        if tier_1_count > 0:
            fig.add_annotation(
                x=tier_1_start + tier_1_count / 2 - 0.5,
                y=max(all_values) * 1.1 if all_values else 1,
                text="TIER 1: CORE",
                showarrow=False,
                font=dict(color='#28a745', size=10, family="Arial Black"),
                bgcolor="rgba(40, 167, 69, 0.1)",
                bordercolor="#28a745",
                borderwidth=1
            )

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Comprehensive EOTS v2.5 Metrics - {symbol}",
                x=0.5,
                font=dict(size=16, color='white', family="Arial Black")
            ),
            xaxis=dict(
                tickangle=45,
                tickfont=dict(color='white', size=9),
                gridcolor='rgba(255, 255, 255, 0.1)',
                title="Metrics (Grouped by Tier)"
            ),
            yaxis=dict(
                tickfont=dict(color='white'),
                gridcolor='rgba(255, 255, 255, 0.1)',
                zeroline=True,
                zerolinecolor='rgba(255, 255, 255, 0.3)',
                title="Values"
            ),
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            height=300,
            margin=dict(l=50, r=50, t=80, b=100),
            showlegend=False
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating comprehensive metrics chart: {str(e)}")
        return create_empty_figure("Comprehensive EOTS Metrics", reason=f"Error: {str(e)}")


# ===== UTILITY FUNCTIONS =====

# PYDANTIC-FIRST: Replace hardcoded dictionary with Pydantic model
from data_models.eots_schemas_v2_5 import ChartLayoutConfigV2_5

def get_unified_chart_layout(title: str, height: int = 200) -> Dict[str, Any]:
    """Get unified chart layout for consistency using Pydantic validation."""
    # Create validated Pydantic model
    layout_config = ChartLayoutConfigV2_5(
        title_text=title,
        height=height
    )

    # Return validated Plotly layout dictionary
    return layout_config.to_plotly_layout()


def get_unified_bar_trace_config(x_data: List, y_data: List, colors: List[str], name: str = 'Values') -> Dict[str, Any]:
    """Get unified bar trace configuration."""
    return {
        'x': x_data,
        'y': y_data,
        'marker_color': colors,
        'name': name,
        'text': [f"{v:.2f}" if isinstance(v, (int, float)) else str(v) for v in y_data],
        'textposition': 'auto',
        'textfont': {'color': 'white', 'size': 10}
    }


def get_unified_line_trace_config(x_data: List, y_data: List, color: str, name: str = 'Line') -> Dict[str, Any]:
    """Get unified line trace configuration."""
    return {
        'x': x_data,
        'y': y_data,
        'mode': 'lines+markers',
        'line': {'color': color, 'width': 2},
        'marker': {'size': 6, 'color': color},
        'name': name
    }
