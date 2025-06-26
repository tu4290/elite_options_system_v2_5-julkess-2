# EOTS v2.5 - S-GRADE, AUTHORITATIVE MAIN DASHBOARD DISPLAY

import logging
from typing import List, Optional, Any, Union, Tuple
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import numpy as np
from pydantic import ValidationError

from dashboard_application import ids
from dashboard_application.utils_dashboard_v2_5 import create_empty_figure, PLOTLY_TEMPLATE, add_timestamp_annotation, apply_dark_theme_template
from data_models import FinalAnalysisBundleV2_5, ProcessedUnderlyingAggregatesV2_5, ActiveRecommendationPayloadV2_5, ProcessedStrikeLevelMetricsV2_5 # Updated import
from utils.config_manager_v2_5 import ConfigManagerV2_5

logger = logging.getLogger(__name__)

# --- Helper Functions for Component Generation ---

def _get_gauge_interpretation(value: float, metric_name: str) -> Tuple[str, str]:
    """Returns (synopsis, interpretation) for gauge values."""
    if metric_name in ["VAPI-FA", "DWFD", "TW-LAF"]:
        # Flow metrics
        if value >= 2:
            return ("Strong bullish flow momentum", "Strong Bullish Signal")
        elif value >= 1:
            return ("Moderate bullish flow", "Moderate Bullish Signal")
        elif value >= -1:
            return ("Neutral/mixed flow", "Neutral/Mixed Signal")
        elif value >= -2:
            return ("Bearish flow momentum", "Moderate Bearish Signal")
        else:
            return ("Strong bearish flow momentum", "Strong Bearish Signal")
    elif metric_name == "GIB OI-Based":
        # Gamma Imbalance
        if value > 5000:
            return ("Dealers long gamma (market stable)", "High Dealer Long Gamma (Price Stability)")
        elif value > 1000:
            return ("Moderate dealer long gamma", "Dealer Long Gamma (Some Stability)")
        elif value > 0:
            return ("Slight dealer long gamma", "Dealer Long Gamma (Some Stability)")
        elif value > -1000:
            return ("Slight dealer short gamma", "Dealer Short Gamma (Some Volatility)")
        elif value > -5000:
            return ("Moderate dealer short gamma", "Dealer Short Gamma (Some Volatility)")
        else:
            return ("Dealers short gamma (market volatile)", "High Dealer Short Gamma (High Volatility)")
    elif metric_name in ["TD-GIB", "HP-EOD"]:
        if value > 0:
            return ("Buying/positive pressure expected", "Positive Pressure Expected")
        else:
            return ("Selling/negative pressure expected", "Negative Pressure Expected")
    else:
        return (f"Value: {value:.2f}", f"Value: {value:.2f}")

def _get_heatmap_interpretation(value: float, metric_name: str) -> str:
    """Returns interpretation text for heatmap values."""
    if metric_name == "SGDHP":
        if value > 50:
            return "Very Strong Support/Resistance Level"
        elif value > 20:
            return "Strong Support/Resistance Level"
        elif value > 5:
            return "Moderate Support/Resistance Level"
        else:
            return "Weak Support/Resistance Level"
    elif metric_name == "UGCH":
        if value > 5:
            return "Very High Greek Confluence"
        elif value > 2:
            return "High Greek Confluence"
        elif value > 0:
            return "Moderate Greek Confluence"
        elif value > -2:
            return "Low Greek Confluence"
        else:
            return "Very Low Greek Confluence"
    else:
        return f"Value: {value:.2f}"

def _get_dashboard_settings(config: ConfigManagerV2_5) -> dict:
    """Get dashboard settings with proper fallbacks"""
    try:
        dashboard_config = config.config.visualization_settings.dashboard
        return dashboard_config.get("main_dashboard_settings", {})
    except Exception as e:
        logger.warning(f"Failed to load dashboard settings: {e}")
        return {}

def _create_flow_gauge(
    metric_name: str,
    value: Optional[float],
    component_id: str,
    config: ConfigManagerV2_5,
    timestamp: Optional[datetime],
    symbol: str
) -> html.Div:
    """Creates a flow gauge for enhanced flow metrics (VAPI-FA, DWFD, TW-LAF)."""
    main_dash_settings = _get_dashboard_settings(config)
    flow_gauge_settings = main_dash_settings.get("flow_gauge", {})
    fig_height = flow_gauge_settings.get("height", 350)
    
    # Chart blurbs for user guidance
    gauge_blurbs = {
        "VAPI-FA": "📈 Volatility-Adjusted Premium Intensity with Flow Acceleration: Measures premium flow momentum adjusted for volatility. +3 = Strong bullish flow, -3 = Strong bearish flow. Use for trend confirmation.",
        "DWFD": "⚖️ Delta-Weighted Flow Divergence: Detects when smart money flows diverge from price action. +3 = Bullish divergence, -3 = Bearish divergence. Use for reversal signals.",
        "TW-LAF": "⏰ Time-Weighted Liquidity-Adjusted Flow: Tracks sustained flow patterns across multiple timeframes. +3 = Sustained bullish pressure, -3 = Sustained bearish pressure. Use for trend strength."
    }
    
    blurb_text = gauge_blurbs.get(metric_name, f"{metric_name} flow analysis")
    gauge_title_text = f"{metric_name}"  # Just the metric name to prevent overlapping
    indicator_title_text = f"{metric_name}"

    if value is None or pd.isna(value):
        fig = create_empty_figure(title=gauge_title_text, height=fig_height, reason="Data N/A")
    else:
        synopsis, interpretation = _get_gauge_interpretation(float(value), metric_name)
        hover_text = f"""
        <b>{symbol} - {metric_name}</b><br>
        Current Value: {float(value):.2f}<br>
        Range: -3 to +3<br>
        <b>Quick Synopsis:</b> {synopsis}<br>
        Interpretation: {interpretation}<br>
        <extra></extra>
        """
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            title={'text': indicator_title_text, 'font': {'size': flow_gauge_settings.get("indicator_font_size", 14)}},
            number={'font': {'size': flow_gauge_settings.get("number_font_size", 20)}},
            gauge={
                'axis': {'range': [-3, 3], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0)"},
                'steps': [
                    {'range': [-3, -2], 'color': '#d62728'},
                    {'range': [-2, -1], 'color': '#ff9896'},
                    {'range': [-1, 1], 'color': '#aec7e8'},
                    {'range': [1, 2], 'color': '#98df8a'},
                    {'range': [2, 3], 'color': '#2ca02c'}
                ],
                'threshold': {
                    'line': {'color': flow_gauge_settings.get("threshold_line_color", "white"), 'width': 3},
                    'thickness': 0.8, 'value': float(value)
                }
            }
        ))
        
        # Add invisible scatter point for custom hover
        fig.add_trace(go.Scatter(
            x=[0.5], y=[0.5],
            mode='markers',
            marker=dict(size=1, opacity=0),
            hovertemplate=hover_text,
            showlegend=False,
            name=""
        ))
        fig.update_layout(
            height=fig_height,  # Normal height without extra space for blurb
            margin=flow_gauge_settings.get("margin", {'t': 30, 'b': 30, 'l': 15, 'r': 15}),  # Reduced top margin since no main title
            template=PLOTLY_TEMPLATE,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x',        # Use x-axis hover to reduce obstruction
            hoverdistance=20      # Reduced hover distance to minimize obstruction
        )
        
        # Apply custom dark theme styling
        fig = apply_dark_theme_template(fig)
        
        # Remove grid lines from gauge charts
        fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
        fig.update_yaxes(showgrid=False, zeroline=False, visible=False)

    if timestamp:
        fig = add_timestamp_annotation(fig, timestamp)

    # Create the graph with collapsible about section
    graph_component = dcc.Graph(
        id=component_id, 
        figure=fig,
        config={
            'displayModeBar': False,
            'displaylogo': False
        },
        style={"height": "350px"}
    )
    
    # Create collapsible about section with elite styling
    about_button = dbc.Button(
        "ℹ️ About", 
        id={"type": "about-toggle", "index": component_id}, 
        color="link", 
        size="sm", 
        className="p-0 text-elite-secondary mb-2 elite-focus-visible",
        style={'font-size': '0.75em'}
    )
    
    about_collapse = dbc.Collapse(
        html.Small(blurb_text, className="text-elite-secondary d-block mb-2", style={'font-size': '0.75em'}),
        id={"type": "about-collapse", "index": component_id},
        is_open=False
    )
    
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                about_button,
                about_collapse,
                graph_component
            ], className="elite-card-body", style={"height": "auto"}),
            className="elite-card fade-in-up",
            style={"height": "auto"}
        )
    ], style={"height": "auto"})

def _create_recommendations_table(
    recommendations: List[ActiveRecommendationPayloadV2_5],
    config: ConfigManagerV2_5,
    timestamp: Optional[datetime],
    symbol: str
) -> html.Div:
    """Creates the display for the ATIF recommendations table using dash_table.DataTable."""
    main_dash_settings = _get_dashboard_settings(config)
    table_settings = main_dash_settings.get("recommendations_table", {})
    max_rationale_len = table_settings.get('max_rationale_length', 50)
    table_title = table_settings.get("title", "ATIF Recommendations")

    card_body_children: List[Union[html.H4, dbc.Alert, dash_table.DataTable, html.Small]] = [
        html.H4(table_title, className="elite-card-title")
    ]

    if not recommendations:
        card_body_children.append(dbc.Alert("No active recommendations.", color="info", className="mt-2 fade-in-up"))
    else:
        data_for_table = []
        for reco in recommendations:
            rationale = reco.target_rationale
            if rationale and len(rationale) > max_rationale_len:
                rationale = rationale[:max_rationale_len] + '...'

            data_for_table.append({
                'Strategy': reco.strategy_type,
                'Bias': reco.trade_bias,
                'Conviction': f"{reco.atif_conviction_score_at_issuance:.2f}",
                'Status': reco.status,
                'Entry': f"{reco.entry_price_initial:.2f}" if reco.entry_price_initial is not None else "N/A",
                'Stop': f"{reco.stop_loss_current:.2f}" if reco.stop_loss_current is not None else "N/A",
                'Target 1': f"{reco.target_1_current:.2f}" if reco.target_1_current is not None else "N/A",
                'Rationale': rationale
            })

        table_component = dash_table.DataTable(
            id=f"{ids.ID_RECOMMENDATIONS_TABLE}-{symbol.lower()}",
            columns=[{"name": i, "id": i} for i in data_for_table[0].keys()] if data_for_table else [],
            data=data_for_table,
            style_cell={
                'textAlign': 'left', 
                'padding': '12px', 
                'minWidth': '80px', 
                'width': 'auto', 
                'maxWidth': '200px',
                'backgroundColor': 'var(--elite-surface)',
                'color': 'var(--elite-text-primary)',
                'border': '1px solid var(--elite-border)',
                'fontFamily': 'var(--elite-font-family)'
            },
            style_header={
                'backgroundColor': 'var(--elite-primary)', 
                'fontWeight': 'bold', 
                'color': 'var(--elite-text-on-primary)',
                'border': '1px solid var(--elite-border)',
                'textAlign': 'center'
            },
            style_data={
                'backgroundColor': 'var(--elite-surface)', 
                'color': 'var(--elite-text-primary)',
                'border': '1px solid var(--elite-border)'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'var(--elite-surface-variant)'
                }
            ],
            style_as_list_view=True,
            page_size=table_settings.get("page_size", 5),
            sort_action="native",
            filter_action="native",
            css=[{
                'selector': '.dash-table-container',
                'rule': 'border-radius: var(--elite-border-radius); overflow: hidden;'
            }]
        )
        card_body_children.append(table_component)

    if timestamp:
        ts_format = config.config.visualization_settings.dashboard.get("timestamp_format", '%Y-%m-%d %H:%M:%S %Z')
        timestamp_text = f"Last updated: {timestamp.strftime(ts_format)}"
        card_body_children.append(html.Small(timestamp_text, className="text-elite-secondary d-block mt-2 text-end"))

    return html.Div([
        dbc.Card(
            dbc.CardBody(card_body_children, className="elite-card-body"),
            className="elite-card fade-in-up"
        )
    ])

def _create_gib_gauge(
    metric_name: str,
    value: Optional[float],
    component_id: str,
    config: ConfigManagerV2_5,
    timestamp: Optional[datetime],
    symbol: str,
    is_dollar_value: bool = False
) -> html.Div:
    """Creates a GIB gauge for gamma imbalance metrics."""
    main_dash_settings = _get_dashboard_settings(config)
    gib_gauge_settings = main_dash_settings.get("gib_gauge", {})
    fig_height = gib_gauge_settings.get("height", 350)
    
    # Chart blurbs for user guidance
    gib_blurbs = {
        "GIB OI-Based": "🎯 Gamma Imbalance Barometer: Measures net gamma exposure from open interest. Positive = Dealers long gamma (price stability), Negative = Dealers short gamma (volatility). Use for volatility forecasting.",
        "TD-GIB": "⏰ Time-Decay Adjusted GIB: GIB adjusted for time decay effects. Shows how gamma imbalance evolves toward expiration. Higher values = Stronger pin risk. Use for EOD positioning.",
        "HP-EOD": "🔚 End-of-Day Hedging Pressure: Predicts hedging flows into market close. Positive = Buying pressure expected, Negative = Selling pressure expected. Use for EOD trade timing."
    }
    
    blurb_text = gib_blurbs.get(metric_name, f"{metric_name} gamma analysis")
    gauge_title_text = f"{metric_name}"  # Just the metric name to prevent overlapping
    indicator_title_text = metric_name

    if value is None or pd.isna(value):
        fig = create_empty_figure(title=gauge_title_text, height=fig_height, reason="Data N/A")
    else:
        # Dynamic scaling based on value type
        if is_dollar_value:
            # For HP_EOD (dollar values)
            abs_val = abs(float(value))
            if abs_val > 50000:
                axis_range = [-100000, 100000]
            elif abs_val > 10000:
                axis_range = [-50000, 50000]
            else:
                axis_range = [-10000, 10000]
            
            steps = gib_gauge_settings.get("steps_dollar", [
                {'range': [axis_range[0], axis_range[0]*0.5], 'color': '#d62728'},
                {'range': [axis_range[0]*0.5, axis_range[0]*0.1], 'color': '#ff9896'},
                {'range': [axis_range[0]*0.1, axis_range[1]*0.1], 'color': '#aec7e8'},
                {'range': [axis_range[1]*0.1, axis_range[1]*0.5], 'color': '#98df8a'},
                {'range': [axis_range[1]*0.5, axis_range[1]], 'color': '#2ca02c'}
            ])
        elif metric_name == "GIB OI-Based":
            # For GIB (large values)
            abs_val = abs(float(value))
            if abs_val > 50000:
                axis_range = [-100000, 100000]
            elif abs_val > 10000:
                axis_range = [-50000, 50000]
            else:
                axis_range = [-10000, 10000]
                
            steps = gib_gauge_settings.get("steps_gib", [
                {'range': [axis_range[0], axis_range[0]*0.5], 'color': '#d62728'},
                {'range': [axis_range[0]*0.5, axis_range[0]*0.1], 'color': '#ff9896'},
                {'range': [axis_range[0]*0.1, axis_range[1]*0.1], 'color': '#aec7e8'},
                {'range': [axis_range[1]*0.1, axis_range[1]*0.5], 'color': '#98df8a'},
                {'range': [axis_range[1]*0.5, axis_range[1]], 'color': '#2ca02c'}
            ])
        else:
            axis_range = gib_gauge_settings.get("axis_range", [-1, 1])
            steps = gib_gauge_settings.get("steps", [
                {'range': [-1, -0.5], 'color': '#d62728'},
                {'range': [-0.5, -0.1], 'color': '#ff9896'},
                {'range': [-0.1, 0.1], 'color': '#aec7e8'},
                {'range': [0.1, 0.5], 'color': '#98df8a'},
                {'range': [0.5, 1], 'color': '#2ca02c'}
            ])

        synopsis, interpretation = _get_gauge_interpretation(float(value), metric_name)
        hover_text = f"""
        <b>{symbol} - {metric_name}</b><br>
        Current Value: {float(value):,.0f}<br>
        Range: {axis_range[0]:,} to {axis_range[1]:,}<br>
        <b>Quick Synopsis:</b> {synopsis}<br>
        Interpretation: {interpretation}<br>
        <extra></extra>
        """
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            title={'text': indicator_title_text, 'font': {'size': gib_gauge_settings.get("indicator_font_size", 14)}},
            number={'font': {'size': gib_gauge_settings.get("number_font_size", 20)}},
            gauge={
                'axis': {'range': axis_range, 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0)"},
                'steps': steps,
                'threshold': {
                    'line': {'color': gib_gauge_settings.get("threshold_line_color", "white"), 'width': 3},
                    'thickness': 0.8, 'value': float(value)
                }
            }
        ))
        
        # Add invisible scatter point for custom hover
        fig.add_trace(go.Scatter(
            x=[0.5], y=[0.5],
            mode='markers',
            marker=dict(size=1, opacity=0),
            hovertemplate=hover_text,
            showlegend=False,
            name=""
        ))
        fig.update_layout(
            height=fig_height,  # Normal height without extra space for blurb
            margin=gib_gauge_settings.get("margin", {'t': 30, 'b': 30, 'l': 15, 'r': 15}),  # Reduced top margin since no main title
            template=PLOTLY_TEMPLATE,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Apply custom dark theme styling
        fig = apply_dark_theme_template(fig)
        
        # Remove grid lines from gauge charts
        fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
        fig.update_yaxes(showgrid=False, zeroline=False, visible=False)

    if timestamp:
        fig = add_timestamp_annotation(fig, timestamp)

    # Create the graph with collapsible about section
    graph_component = dcc.Graph(
        id=component_id, 
        figure=fig,
        config={
            'displayModeBar': False,
            'displaylogo': False
        },
        style={"height": "350px"}
    )
    
    # Create collapsible about section with elite styling
    about_button = dbc.Button(
        "ℹ️ About", 
        id={"type": "about-toggle", "index": component_id}, 
        color="link", 
        size="sm", 
        className="p-0 text-elite-secondary mb-2 elite-focus-visible",
        style={'font-size': '0.75em'}
    )
    
    about_collapse = dbc.Collapse(
        html.Small(blurb_text, className="text-elite-secondary d-block mb-2", style={'font-size': '0.75em'}),
        id={"type": "about-collapse", "index": component_id},
        is_open=False
    )
    
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                about_button,
                about_collapse,
                graph_component
            ], className="elite-card-body", style={"height": "auto"}),
            className="elite-card fade-in-up",
            style={"height": "auto"}
        )
    ], style={"height": "auto"})

def _create_mini_heatmap(
    metric_name: str,
    strike_data: List[ProcessedStrikeLevelMetricsV2_5],
    metric_field: str,
    component_id: str,
    config: ConfigManagerV2_5,
    timestamp: Optional[datetime],
    symbol: str,
    current_price: Optional[float]
) -> html.Div:
    """Creates a mini-heatmap for strike-level metrics like SGDHP and UGCH."""
    main_dash_settings = _get_dashboard_settings(config)
    heatmap_settings = main_dash_settings.get("mini_heatmap", {})
    fig_height = 300
    
    # Chart blurbs for user guidance
    chart_blurbs = {
        "SGDHP": "🎯 Super Gamma-Delta Hedging Pressure: Shows strikes where market makers defend prices most aggressively. Cyan = Strong positive pressure (support), Magenta = Strong negative pressure (resistance). Use for dynamic S/R levels.",
        "UGCH": "⚡ Ultimate Greek Confluence: Highlights strikes where ALL Greeks (Delta, Gamma, Vega, Theta, Charm, Vanna) align. Higher values = stronger structural significance. Use for high-conviction strike selection."
    }
    
    blurb_text = chart_blurbs.get(metric_name, f"{metric_name} strike-level analysis")
    heatmap_title_text = f"{metric_name} Mini-Heatmap"  # Just the metric name to prevent overlapping

    if not strike_data or current_price is None:
        fig = create_empty_figure(title=heatmap_title_text, height=fig_height, reason="Data N/A")
    else:
        price_range = current_price * 0.05
        relevant_strikes = []
        values = []
        
        for strike_info in strike_data:
            if abs(strike_info.strike - current_price) <= price_range:
                metric_value = getattr(strike_info, metric_field, None)
                if metric_value is not None and pd.notna(metric_value):
                    relevant_strikes.append(strike_info.strike)
                    values.append(metric_value)
        
        if not relevant_strikes:
            fig = create_empty_figure(title=heatmap_title_text, height=fig_height, reason="No ATM/NTM Data")
        else:
            sorted_data = sorted(zip(relevant_strikes, values))
            strikes, vals = zip(*sorted_data)
            
            # Custom color scheme for SGDHP: Cyan-Magenta gradient
            if metric_name == "SGDHP":
                colorscale = [
                    [0.0, '#FF00FF'],    # Magenta for negative (strong resistance)
                    [0.2, '#FF66FF'],    # Light magenta
                    [0.4, '#CCCCCC'],    # Neutral gray
                    [0.6, '#66FFFF'],    # Light cyan  
                    [1.0, '#00FFFF']     # Cyan for positive (strong support)
                ]
            else:
                # Keep default for UGCH
                colorscale = heatmap_settings.get("colorscale", "RdYlGn")
            
            # Create custom hover template for heatmap
            hover_template = (
                f"<b>{symbol} - {metric_name}</b><br>"
                "Strike: $%{x:,.0f}<br>"
                f"Current Price: ${current_price:,.2f}<br>"
                "Distance: $%{customdata[2]:,.2f} (%{customdata[3]:.1f}%)<br>"
                f"{metric_name} Value: %{{z:.2f}}<br>"
                "Interpretation: %{customdata[4]}<br>"
                "<extra></extra>"
            )
            
            # Prepare custom data for hover
            custom_data = []
            for strike, val in zip(strikes, vals):
                distance_from_current = abs(strike - current_price)
                pct_from_current = (distance_from_current / current_price) * 100
                interpretation = _get_heatmap_interpretation(val, metric_name)
                custom_data.append([strike, val, distance_from_current, pct_from_current, interpretation])
            
            fig = go.Figure(data=go.Heatmap(
                z=[vals],
                x=strikes,
                y=[metric_name],
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(len=0.5, thickness=10),
                hovertemplate=hover_template,
                customdata=[custom_data]
            ))
            
            fig.update_layout(
                title={'text': heatmap_title_text, 'y':0.85, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 12}},
                height=fig_height,  # Normal height without extra space for blurb
                margin=heatmap_settings.get("margin", {'t': 60, 'b': 30, 'l': 40, 'r': 40}),  # Increased top margin for title
                template=PLOTLY_TEMPLATE,
                xaxis_title="Strike",
                yaxis_title="",
                showlegend=False
            )
            
            # Apply custom dark theme styling
            fig = apply_dark_theme_template(fig)

    if timestamp:
        fig = add_timestamp_annotation(fig, timestamp)

    # Create the graph with collapsible about section
    graph_component = dcc.Graph(
        id=component_id, 
        figure=fig,
        config={
            'displayModeBar': False,
            'displaylogo': False
        },
        style={"height": "300px"}
    )
    
    # Create collapsible about section with elite styling
    about_button = dbc.Button(
        "ℹ️ About", 
        id={"type": "about-toggle", "index": component_id}, 
        color="link", 
        size="sm", 
        className="p-0 text-elite-secondary mb-2 elite-focus-visible",
        style={'font-size': '0.75em'}
    )
    
    about_collapse = dbc.Collapse(
        html.Small(blurb_text, className="text-elite-secondary d-block mb-2", style={'font-size': '0.75em'}),
        id={"type": "about-collapse", "index": component_id},
        is_open=False
    )
    
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                about_button,
                about_collapse,
                graph_component
            ], className="elite-card-body", style={"height": "auto"}),
            className="elite-card fade-in-up",
            style={"height": "auto"}
        ),
    ], style={"height": "auto"})

def _create_ticker_context_summary(
    ticker_context_dict: Optional[Any],
    config: ConfigManagerV2_5,
    timestamp: Optional[datetime],
    symbol: str
) -> html.Div:
    """Creates the ticker context summary display."""
    main_dash_settings = _get_dashboard_settings(config)
    context_settings = main_dash_settings.get("ticker_context", {})
    context_title = context_settings.get("title", "Ticker Context")

    card_body_children = []

    if not ticker_context_dict:
        card_body_children.append(dbc.Alert("No context data available.", color="info", className="mt-2"))
    else:
        context_flags = []
        
        if hasattr(ticker_context_dict, 'is_0dte') and ticker_context_dict.is_0dte:
            context_flags.append(f"{symbol}: 0DTE")
        if hasattr(ticker_context_dict, 'is_1dte') and ticker_context_dict.is_1dte:
            context_flags.append(f"{symbol}: 1DTE")
        if hasattr(ticker_context_dict, 'active_intraday_session') and ticker_context_dict.active_intraday_session:
            context_flags.append(f"Session: {ticker_context_dict.active_intraday_session}")
        if hasattr(ticker_context_dict, 'vix_spy_price_divergence_strong_negative') and ticker_context_dict.vix_spy_price_divergence_strong_negative:
            context_flags.append("Pattern: VIX_DIVERGENCE_ACTIVE")
        if hasattr(ticker_context_dict, 'is_fomc_meeting_day') and ticker_context_dict.is_fomc_meeting_day:
            context_flags.append("Event: FOMC_DAY")
        if hasattr(ticker_context_dict, 'earnings_approaching_flag') and ticker_context_dict.earnings_approaching_flag:
            context_flags.append("Event: EARNINGS_APPROACHING")
        
        if context_flags:
            for flag in context_flags:
                card_body_children.append(
                    dbc.Badge(flag, color="primary", className="me-1 mb-1")
                )
        else:
            card_body_children.append(
                html.Small("No significant context flags active.", className="text-muted")
            )

    if timestamp:
        ts_format = config.config.visualization_settings.dashboard.get("timestamp_format", '%Y-%m-%d %H:%M:%S %Z')
        timestamp_text = f"Last updated: {timestamp.strftime(ts_format)}"
        card_body_children.append(html.Small(timestamp_text, className="text-muted d-block mt-2 text-end"))

    return html.Div([
        dbc.Card(
            dbc.CardBody(card_body_children),
            className="elite-card fade-in-up"
        )
    ])

def _create_atif_recommendations_table(
    atif_recommendations: list,  # List of ATIFStrategyDirectivePayloadV2_5 or similar
    config: ConfigManagerV2_5,
    timestamp: Optional[datetime],
    symbol: str
) -> html.Div:
    """Creates a robust display for ATIF recommendations using dash_table.DataTable."""
    main_dash_settings = _get_dashboard_settings(config)
    table_settings = main_dash_settings.get("atif_recommendations_table", {})
    max_rationale_len = table_settings.get('max_rationale_length', 80)
    table_title = table_settings.get("title", "ATIF Strategy Insights")

    # Collapsible About Section
    about_button = dbc.Button(
        "ℹ️ About ATIF", 
        id={"type": "about-toggle", "index": f"atif-insights-{symbol}"}, 
        color="link", 
        size="sm", 
        className="p-0 text-elite-secondary mb-2 elite-focus-visible",
        style={'font-size': '0.75em'}
    )
    about_collapse = dbc.Collapse(
        html.Small(
            "This table displays the latest trade ideas generated by the Adaptive Trade Idea Framework (ATIF). Each row shows the system's recommended strategy, conviction, rationale, and the context at the time of issuance. High conviction scores indicate strong system confidence based on a holistic analysis of signals, regime, and historical performance. If no recommendation is present, it means the system found insufficient alignment or conviction for a new trade idea.",
            className="text-elite-secondary d-block mb-2", style={'font-size': '0.75em'}
        ),
        id={"type": "about-collapse", "index": f"atif-insights-{symbol}"},
        is_open=False
    )

    card_body_children = [
        html.H4(table_title, className="elite-card-title"),
        about_button,
        about_collapse
    ]

    if not atif_recommendations:
        card_body_children.append(
            dbc.Alert("No actionable ATIF recommendation at this time.", color="warning", className="mt-2 fade-in-up")
        )
    else:
        data_for_table = []
        for reco in atif_recommendations:
            rec = reco.model_dump() if hasattr(reco, 'model_dump') else reco
            rationale = rec.get('supportive_rationale_components', {}).get('rationale', '')
            if rationale and len(rationale) > max_rationale_len:
                rationale_short = rationale[:max_rationale_len] + '...'
            else:
                rationale_short = rationale
            assessment = rec.get('assessment_profile', {})
            context_flags = []
            if assessment:
                if assessment.get('bullish_assessment_score', 0) > assessment.get('bearish_assessment_score', 0):
                    context_flags.append('Bullish')
                elif assessment.get('bearish_assessment_score', 0) > assessment.get('bullish_assessment_score', 0):
                    context_flags.append('Bearish')
                if assessment.get('vol_expansion_score', 0) > 0:
                    context_flags.append('Vol Expansion')
                if assessment.get('vol_contraction_score', 0) > 0:
                    context_flags.append('Vol Contraction')
                if assessment.get('mean_reversion_likelihood', 0) > 0.5:
                    context_flags.append('Mean Reversion')
            bias = 'Bullish' if assessment.get('bullish_assessment_score', 0) > assessment.get('bearish_assessment_score', 0) else 'Bearish'
            color = '#2ca02c' if bias == 'Bullish' else '#d62728'
            data_for_table.append({
                'Strategy': rec.get('selected_strategy_type', 'N/A'),
                'Bias': bias,
                'Conviction': f"{rec.get('final_conviction_score_from_atif', 0):.2f}",
                'Target DTE': f"{rec.get('target_dte_min', 'N/A')} - {rec.get('target_dte_max', 'N/A')}",
                'Delta (Long)': f"{rec.get('target_delta_long_leg_min', 'N/A')} - {rec.get('target_delta_long_leg_max', 'N/A')}",
                'Delta (Short)': f"{rec.get('target_delta_short_leg_min', 'N/A')} - {rec.get('target_delta_short_leg_max', 'N/A')}",
                'Underlying Price': f"{rec.get('underlying_price_at_decision', 'N/A')}",
                'Rationale': rationale_short,
                'Context': ', '.join(context_flags),
                'Full Rationale': rationale,
                'Assessment': assessment,
                'Color': color
            })
        columns = [
            {"name": "Strategy", "id": "Strategy"},
            {"name": "Bias", "id": "Bias"},
            {"name": "Conviction", "id": "Conviction"},
            {"name": "Target DTE", "id": "Target DTE"},
            {"name": "Delta (Long)", "id": "Delta (Long)"},
            {"name": "Delta (Short)", "id": "Delta (Short)"},
            {"name": "Underlying Price", "id": "Underlying Price"},
            {"name": "Rationale", "id": "Rationale"},
            {"name": "Context", "id": "Context"}
        ]
        table_component = dash_table.DataTable(
            id=f"atif-recommendations-table-{symbol.lower()}",
            columns=columns,
            data=data_for_table,
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'minWidth': '80px',
                'width': 'auto',
                'maxWidth': '220px',
                'backgroundColor': 'var(--elite-surface)',
                'color': 'var(--elite-text-primary)',
                'border': '1px solid var(--elite-border)',
                'fontFamily': 'var(--elite-font-family)'
            },
            style_header={
                'backgroundColor': 'var(--elite-primary)',
                'fontWeight': 'bold',
                'color': 'var(--elite-text-on-primary)',
                'border': '1px solid var(--elite-border)',
                'textAlign': 'center'
            },
            style_data={
                'backgroundColor': 'var(--elite-surface)',
                'color': 'var(--elite-text-primary)',
                'border': '1px solid var(--elite-border)'
            },
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Bias} = "Bullish"'},
                    'backgroundColor': '#193c1a',
                },
                {
                    'if': {'filter_query': '{Bias} = "Bearish"'},
                    'backgroundColor': '#3c1a1a',
                },
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'var(--elite-surface-variant)'
                }
            ],
            style_as_list_view=True,
            page_size=table_settings.get("page_size", 5),
            sort_action="native",
            filter_action="native",
            tooltip_data=[
                {
                    'Rationale': {'value': row['Full Rationale'], 'type': 'markdown'}
                } for row in data_for_table
            ],
            tooltip_duration=None,
            css=[{
                'selector': '.dash-table-container',
                'rule': 'border-radius: var(--elite-border-radius); overflow: hidden;'
            }]
        )
        card_body_children.append(table_component)

    if timestamp:
        ts_format = config.config.visualization_settings.dashboard.get("timestamp_format", '%Y-%m-%d %H:%M:%S %Z')
        timestamp_text = f"Last updated: {timestamp.strftime(ts_format)}"
        card_body_children.append(html.Small(timestamp_text, className="text-elite-secondary d-block mt-2 text-end"))

    return html.Div([
        dbc.Card(
            dbc.CardBody(card_body_children, className="elite-card-body"),
            className="elite-card fade-in-up"
        )
    ])

# --- Main Layout Function ---

def create_layout(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> html.Div:
    """Creates the complete layout for the Main Dashboard mode. Strict Pydantic-first: validates all data at the UI boundary."""
    # --- Pydantic-first validation at entry ---
    if not isinstance(bundle, FinalAnalysisBundleV2_5):
        logger.error("Input bundle is not a FinalAnalysisBundleV2_5 instance.")
        return html.Div([dbc.Card(dbc.CardBody([dbc.Alert("Internal error: Invalid analysis bundle type.", color="danger")]))])
    try:
        bundle.model_validate(bundle.model_dump())
    except ValidationError as e:
        logger.error(f"Bundle validation error: {e}")
        return html.Div([dbc.Card(dbc.CardBody([dbc.Alert("Internal error: Analysis bundle failed validation.", color="danger")]))])
    if not bundle or not bundle.processed_data_bundle:
        return html.Div([dbc.Card(dbc.CardBody([dbc.Alert("Analysis data is not available. Cannot render Main Dashboard.", color="danger")]))])

    processed_data = bundle.processed_data_bundle
    und_data = processed_data.underlying_data_enriched
    strike_data = processed_data.strike_level_data_with_metrics
    symbol = bundle.target_symbol or "Unknown"
    bundle_timestamp = bundle.bundle_timestamp
    current_price = und_data.price if und_data else None

    # --- Validate nested Pydantic models/lists ---
    if und_data is not None and not hasattr(und_data, 'model_dump'):
        logger.error("underlying_data_enriched is not a Pydantic model.")
        return html.Div([dbc.Card(dbc.CardBody([dbc.Alert("Internal error: Underlying data is not valid.", color="danger")]))])
    if strike_data is not None:
        if not isinstance(strike_data, list) or not all(hasattr(x, 'model_dump') for x in strike_data):
            logger.error("strike_level_data_with_metrics is not a list of Pydantic models.")
            return html.Div([dbc.Card(dbc.CardBody([dbc.Alert("Internal error: Strike data is not valid.", color="danger")]))])
    if bundle.active_recommendations_v2_5 is not None:
        if not isinstance(bundle.active_recommendations_v2_5, list) or not all(hasattr(x, 'model_dump') for x in bundle.active_recommendations_v2_5):
            logger.error("active_recommendations_v2_5 is not a list of Pydantic models.")
            return html.Div([dbc.Card(dbc.CardBody([dbc.Alert("Internal error: Recommendations data is not valid.", color="danger")]))])

    # --- ATIF Recommendations Table Integration ---
    atif_recommendations = []
    if hasattr(bundle, 'atif_recommendations_v2_5'):
        atif_recommendations = getattr(bundle, 'atif_recommendations_v2_5', [])
    if not atif_recommendations and hasattr(bundle, 'active_recommendations_v2_5'):
        atif_recommendations = bundle.active_recommendations_v2_5

    return html.Div(
        id=ids.ID_MAIN_DASHBOARD_CONTAINER,
        children=[
            dbc.Container(
                fluid=True,
                children=[
                    # Row 2: Flow Metrics (VAPI-FA, DWFD, TW-LAF)
                    dbc.Row([
                        dbc.Col(_create_flow_gauge("VAPI-FA", und_data.vapi_fa_z_score_und, ids.ID_VAPI_GAUGE, config, bundle_timestamp, symbol), md=12, lg=4, className="mb-4"),
                        dbc.Col(_create_flow_gauge("DWFD", und_data.dwfd_z_score_und, ids.ID_DWFD_GAUGE, config, bundle_timestamp, symbol), md=12, lg=4, className="mb-4"),
                        dbc.Col(_create_flow_gauge("TW-LAF", und_data.tw_laf_z_score_und, ids.ID_TW_LAF_GAUGE, config, bundle_timestamp, symbol), md=12, lg=4, className="mb-4"),
                    ], className="mt-3"),
                    # Row 3: GIB Gauges (GIB OI-Based, TD-GIB, HP-EOD)
                    dbc.Row([
                        dbc.Col(_create_gib_gauge("GIB OI-Based", und_data.gib_oi_based_und, f"{ids.ID_GIB_GAUGE}-oi", config, bundle_timestamp, symbol), md=12, lg=4, className="mb-4"),
                        dbc.Col(_create_gib_gauge("TD-GIB", und_data.td_gib_und, f"{ids.ID_GIB_GAUGE}-td", config, bundle_timestamp, symbol, is_dollar_value=False), md=12, lg=4, className="mb-4"),
                        dbc.Col(_create_gib_gauge("HP-EOD", und_data.hp_eod_und, f"{ids.ID_HP_EOD_GAUGE}", config, bundle_timestamp, symbol, is_dollar_value=True), md=12, lg=4, className="mb-4"),
                    ], className="mt-3"),
                    # Row 4: SGDHP Mini-Heatmap (Full Width)
                    dbc.Row([
                        dbc.Col(_create_mini_heatmap("SGDHP", strike_data, "sgdhp_score_strike", "sgdhp-mini-heatmap", config, bundle_timestamp, symbol, current_price), md=12, lg=12, className="mb-4", style={"height": "100%"}),
                    ], className="mt-3"),
                    # Row 5: UGCH Mini-Heatmap (Full Width)
                    dbc.Row([
                        dbc.Col(_create_mini_heatmap("UGCH", strike_data, "ugch_score_strike", "ugch-mini-heatmap", config, bundle_timestamp, symbol, current_price), md=12, lg=12, className="mb-4", style={"height": "100%"}),
                    ], className="mt-3"),
                    # Row 6: Recommendations Table and Ticker Context
                    dbc.Row([
                        dbc.Col(_create_recommendations_table(bundle.active_recommendations_v2_5, config, bundle_timestamp, symbol), md=12, lg=8, className="mb-4"),
                        dbc.Col(_create_ticker_context_summary(und_data.ticker_context_dict_v2_5, config, bundle_timestamp, symbol), md=12, lg=4, className="mb-4"),
                    ], className="mt-3"),
                    # Row 7: ATIF Insights (moved to bottom)
                    dbc.Row([
                        dbc.Col(_create_atif_recommendations_table(atif_recommendations, config, bundle_timestamp, symbol), md=12, lg=12, className="mb-4"),
                    ], className="mt-3"),
                ]
            )
        ]
    )