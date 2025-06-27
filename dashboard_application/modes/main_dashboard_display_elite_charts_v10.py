# dashboard_application/modes/main_dashboard_display_elite_charts_v10.py
"""
EOTS v2.5 - Elite V10 Charts Module
This module contains functions to generate Dash/Plotly components for the new
metrics introduced in the Elite Options Impact Calculator V10.
"""

import logging
from typing import List, Optional, Any, Union, Tuple
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc
import dash_bootstrap_components as dbc
import numpy as np

# Assuming these will be available in the context where this module's functions are called
# from dashboard_application import ids # If specific EOTS IDs are needed
from dashboard_application.utils_dashboard_v2_5 import create_empty_figure, PLOTLY_TEMPLATE, add_timestamp_annotation, apply_dark_theme_template
from data_models import FinalAnalysisBundleV2_5, ProcessedStrikeLevelMetricsV2_5, ProcessedContractMetricsV2_5
from utils.config_manager_v2_5 import ConfigManagerV2_5 # For styling/settings consistency
from core_analytics_engine.elite_metrics_calculator_v10_module import EliteImpactColumns, ConvexValueColumns

logger = logging.getLogger(__name__)

def _get_elite_dashboard_settings(config: ConfigManagerV2_5) -> dict:
    """Helper to get elite chart specific settings if defined, or fallback to general ones."""
    try:
        vis_settings = config.config.visualization_settings.dashboard
        return vis_settings.get("elite_charts_settings", vis_settings.get("main_dashboard_settings", {}))
    except Exception:
        return {}

def create_sdag_vs_strike_scatter(
    strike_data_models: List[ProcessedStrikeLevelMetricsV2_5],
    current_price: Optional[float],
    component_id_prefix: str, # Expecting a prefix to make component IDs unique
    config_manager: ConfigManagerV2_5,
    timestamp: Optional[datetime],
    symbol: str
) -> Optional[html.Div]:
    """
    Creates SDAG Consensus vs Strike scatter plot.
    Data is expected to be a list of Pydantic models (ProcessedStrikeLevelMetricsV2_5).
    """
    if not strike_data_models or current_price is None:
        fig = create_empty_figure(title="SDAG Analysis", reason="Data N/A")
    else:
        try:
            # Convert Pydantic models to DataFrame for easier processing
            df_strike = pd.DataFrame([s.model_dump() for s in strike_data_models])
            if df_strike.empty or EliteImpactColumns.SDAG_CONSENSUS not in df_strike.columns or EliteConvexValueColumns.STRIKE not in df_strike.columns:
                fig = create_empty_figure(title="SDAG Analysis", reason="Required data columns missing")
            else:
                elite_settings = _get_elite_dashboard_settings(config_manager)
                chart_settings = elite_settings.get("sdag_vs_strike_scatter", {})

                strikes = df_strike[EliteConvexValueColumns.STRIKE]
                sdag_consensus = df_strike[EliteImpactColumns.SDAG_CONSENSUS]
                confidence = df_strike.get(EliteImpactColumns.PREDICTION_CONFIDENCE, pd.Series(0.5, index=df_strike.index)) # Default confidence if not present

                fig = go.Figure()
                scatter_trace = go.Scatter(
                    x=strikes,
                    y=sdag_consensus,
                    mode='markers',
                    marker=dict(
                        size=chart_settings.get("marker_size", 10),
                        color=confidence,
                        colorscale=chart_settings.get("colorscale", 'Viridis'),
                        showscale=True,
                        colorbar_title="Prediction Confidence",
                        opacity=0.7
                    ),
                    text=[f"Strike: {s}<br>SDAG: {sd:.2f}<br>Conf: {c:.2f}" for s, sd, c in zip(strikes, sdag_consensus, confidence)],
                    hoverinfo='text'
                )
                fig.add_trace(scatter_trace)

                fig.add_hline(y=chart_settings.get("strong_positive_threshold", 1.5), line_dash="dash", line_color="red", annotation_text="Strong Positive SDAG")
                fig.add_hline(y=chart_settings.get("strong_negative_threshold", -1.5), line_dash="dash", line_color="red", annotation_text="Strong Negative SDAG")
                fig.add_vline(x=current_price, line_dash="solid", line_color="yellow", annotation_text="Current Price")

                fig.update_layout(
                    title_text='SDAG Consensus vs Strike',
                    xaxis_title='Strike Price',
                    yaxis_title='SDAG Consensus Score',
                    template=PLOTLY_TEMPLATE,
                    height=chart_settings.get("height", 400)
                )
                fig = apply_dark_theme_template(fig)
                if timestamp:
                    fig = add_timestamp_annotation(fig, timestamp)
        except Exception as e:
            logger.error(f"Error creating SDAG scatter plot: {e}")
            fig = create_empty_figure(title="SDAG Analysis", reason=f"Plotting error: {e}")

    return html.Div(dcc.Graph(id=f"{component_id_prefix}_sdag_scatter", figure=fig, config={'displayModeBar': False}), className="elite-chart-container")

def _create_elite_score_histogram(
    options_data_models: List[ProcessedContractMetricsV2_5],
    component_id_prefix: str,
    config_manager: ConfigManagerV2_5,
    timestamp: Optional[datetime],
    symbol: str
) -> Optional[html.Div]:
    if not options_data_models:
        fig = create_empty_figure(title="Elite Score Distribution", reason="Data N/A")
    else:
        try:
            df_options = pd.DataFrame([s.model_dump() for s in options_data_models])
            if df_options.empty or EliteImpactColumns.ELITE_IMPACT_SCORE not in df_options.columns:
                fig = create_empty_figure(title="Elite Score Distribution", reason="Elite Impact Score data missing")
            else:
                elite_settings = _get_elite_dashboard_settings(config_manager)
                chart_settings = elite_settings.get("elite_score_histogram", {})
                elite_scores = df_options[EliteImpactColumns.ELITE_IMPACT_SCORE].dropna()

                fig = go.Figure(data=[go.Histogram(
                    x=elite_scores,
                    marker_color=chart_settings.get("color", 'cyan'),
                    xbins=dict(size=chart_settings.get("bin_size", 0.1)) # Example bin size
                )])
                mean_val = elite_scores.mean()
                fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                              annotation_text=f"Mean: {mean_val:.3f}", annotation_position="top right")
                fig.update_layout(
                    title_text='Elite Impact Score Distribution',
                    xaxis_title='Elite Impact Score',
                    yaxis_title='Frequency',
                    template=PLOTLY_TEMPLATE,
                    height=chart_settings.get("height", 350)
                )
                fig = apply_dark_theme_template(fig)
                if timestamp: fig = add_timestamp_annotation(fig, timestamp)
        except Exception as e:
            logger.error(f"Error creating Elite Score Histogram: {e}")
            fig = create_empty_figure(title="Elite Score Distribution", reason=f"Plotting error: {e}")
    return html.Div(dcc.Graph(id=f"{component_id_prefix}_elite_score_hist", figure=fig, config={'displayModeBar': False}), className="elite-chart-container")

def _create_gamma_wall_plot(
    strike_data_models: List[ProcessedStrikeLevelMetricsV2_5],
    current_price: Optional[float],
    component_id_prefix: str,
    config_manager: ConfigManagerV2_5,
    timestamp: Optional[datetime],
    symbol: str
) -> Optional[html.Div]:
    if not strike_data_models or current_price is None:
        fig = create_empty_figure(title="Gamma Wall Analysis", reason="Data N/A")
    else:
        try:
            df_strike = pd.DataFrame([s.model_dump() for s in strike_data_models])
            # Ensure the correct column name from EliteImpactColumns for strike magnetism
            magnetism_col = EliteImpactColumns.STRIKE_MAGNETISM_INDEX
            strike_col = EliteConvexValueColumns.STRIKE # Assuming this is the strike column name

            if df_strike.empty or magnetism_col not in df_strike.columns or strike_col not in df_strike.columns:
                fig = create_empty_figure(title="Gamma Wall Analysis", reason="Required data missing (Strike Magnetism or Strike)")
            else:
                elite_settings = _get_elite_dashboard_settings(config_manager)
                chart_settings = elite_settings.get("gamma_wall_plot", {})

                df_strike = df_strike.sort_values(by=strike_col)
                strikes = df_strike[strike_col]
                gamma_impact = df_strike[magnetism_col]

                fig = go.Figure(data=[go.Scatter(
                    x=strikes, y=gamma_impact, mode='lines+markers',
                    line=dict(color=chart_settings.get("line_color", 'orange')),
                    marker=dict(size=chart_settings.get("marker_size", 6)),
                    text=[f"Strike: {s}<br>Magnetism: {gi:.2f}" for s, gi in zip(strikes, gamma_impact)],
                    hoverinfo='text'
                )])
                fig.add_vline(x=current_price, line_dash="solid", line_color="yellow", annotation_text="Current Price")
                fig.update_layout(
                    title_text='Gamma Wall Analysis (Strike Magnetism)',
                    xaxis_title='Strike Price',
                    yaxis_title='Strike Magnetism Index',
                    template=PLOTLY_TEMPLATE,
                    height=chart_settings.get("height", 350)
                )
                fig = apply_dark_theme_template(fig)
                if timestamp: fig = add_timestamp_annotation(fig, timestamp)
        except Exception as e:
            logger.error(f"Error creating Gamma Wall plot: {e}")
            fig = create_empty_figure(title="Gamma Wall Analysis", reason=f"Plotting error: {e}")
    return html.Div(dcc.Graph(id=f"{component_id_prefix}_gamma_wall", figure=fig, config={'displayModeBar': False}), className="elite-chart-container")

def _create_flow_momentum_heatmap(
    options_data_models: List[ProcessedContractMetricsV2_5],
    component_id_prefix: str,
    config_manager: ConfigManagerV2_5,
    timestamp: Optional[datetime],
    symbol: str,
    num_options_to_display: int = 20
) -> Optional[html.Div]:
    if not options_data_models:
        fig = create_empty_figure(title="Flow Momentum Heatmap", reason="Data N/A")
    else:
        try:
            df_options = pd.DataFrame([s.model_dump() for s in options_data_models])
            momentum_cols = [
                EliteImpactColumns.FLOW_VELOCITY_5M, EliteImpactColumns.FLOW_VELOCITY_15M,
                EliteImpactColumns.FLOW_ACCELERATION, EliteImpactColumns.MOMENTUM_PERSISTENCE
            ]
            # Ensure all required momentum columns are present
            missing_cols = [col for col in momentum_cols if col not in df_options.columns]
            if df_options.empty or missing_cols:
                fig = create_empty_figure(title="Flow Momentum Heatmap", reason=f"Momentum data missing: {', '.join(missing_cols) if missing_cols else 'Empty DataFrame'}")
            else:
                elite_settings = _get_elite_dashboard_settings(config_manager)
                chart_settings = elite_settings.get("flow_momentum_heatmap", {})

                # Select a subset of options (e.g., by highest OI or volume if available)
                # For demo, just take head if too many
                # In a real scenario, one might sort by a significance metric
                sort_col = EliteConvexValueColumns.OI if EliteConvexValueColumns.OI in df_options.columns else EliteImpactColumns.ELITE_IMPACT_SCORE
                if sort_col in df_options.columns and len(df_options) > num_options_to_display :
                     df_display = df_options.nlargest(num_options_to_display, sort_col)
                else:
                     df_display = df_options.head(num_options_to_display)


                momentum_data = df_display[momentum_cols].fillna(0)

                y_labels = [
                    f"${r.get(EliteConvexValueColumns.STRIKE, 'N/A')} {r.get(EliteConvexValueColumns.OPT_KIND, 'N/A')[0].upper()}"
                    if r.get(EliteConvexValueColumns.OPT_KIND) else f"${r.get(EliteConvexValueColumns.STRIKE, 'N/A')}"
                    for idx, r in df_display.iterrows()
                ]

                fig = go.Figure(data=go.Heatmap(
                    z=momentum_data.values.T,
                    x=y_labels,
                    y=[col.replace('flow_','').replace('_',' ').title() for col in momentum_cols],
                    colorscale=chart_settings.get("colorscale", 'RdYlBu_r'),
                    zmid=0,
                    colorbar_title="Momentum Strength"
                ))
                fig.update_layout(
                    title_text='Flow Momentum Analysis',
                    xaxis_title='Option (Strike Type)',
                    yaxis_title='Momentum Metric',
                    template=PLOTLY_TEMPLATE,
                    height=chart_settings.get("height", 400),
                    xaxis_tickangle=-45
                )
                fig = apply_dark_theme_template(fig)
                if timestamp: fig = add_timestamp_annotation(fig, timestamp)
        except Exception as e:
            logger.error(f"Error creating Flow Momentum Heatmap: {e}")
            fig = create_empty_figure(title="Flow Momentum Heatmap", reason=f"Plotting error: {e}")
    return html.Div(dcc.Graph(id=f"{component_id_prefix}_flow_momentum_heatmap", figure=fig, config={'displayModeBar': False}), className="elite-chart-container")

def _create_vol_pressure_vs_delta_scatter(
    options_data_models: List[ProcessedContractMetricsV2_5],
    component_id_prefix: str,
    config_manager: ConfigManagerV2_5,
    timestamp: Optional[datetime],
    symbol: str
) -> Optional[html.Div]:
    if not options_data_models:
        fig = create_empty_figure(title="Volatility Pressure vs Delta", reason="Data N/A")
    else:
        try:
            df_options = pd.DataFrame([s.model_dump() for s in options_data_models])
            req_cols = [EliteImpactColumns.VOLATILITY_PRESSURE_INDEX, EliteImpactColumns.REGIME_ADJUSTED_DELTA]
            if df_options.empty or not all(col in df_options.columns for col in req_cols):
                fig = create_empty_figure(title="Volatility Pressure vs Delta", reason="Required data missing")
            else:
                elite_settings = _get_elite_dashboard_settings(config_manager)
                chart_settings = elite_settings.get("vol_pressure_scatter", {})

                vol_pressure = df_options[EliteImpactColumns.VOLATILITY_PRESSURE_INDEX]
                delta_exposure = df_options[EliteImpactColumns.REGIME_ADJUSTED_DELTA]
                signal_strength = df_options.get(EliteImpactColumns.SIGNAL_STRENGTH, pd.Series(0.5, index=df_options.index))

                fig = go.Figure(data=[go.Scatter(
                    x=delta_exposure, y=vol_pressure, mode='markers',
                    marker=dict(
                        size=chart_settings.get("marker_size", 8),
                        color=signal_strength,
                        colorscale=chart_settings.get("colorscale", 'Plasma'),
                        showscale=True,
                        colorbar_title="Signal Strength"
                    ),
                    text=[f"Delta Exp: {d:.2f}<br>Vol Pressure: {vp:.2f}<br>Signal: {ss:.2f}" for d,vp,ss in zip(delta_exposure, vol_pressure, signal_strength)],
                    hoverinfo='text'
                )])
                fig.update_layout(
                    title_text='Volatility Pressure vs Delta Exposure',
                    xaxis_title='Regime Adjusted Delta Exposure',
                    yaxis_title='Volatility Pressure Index',
                    template=PLOTLY_TEMPLATE,
                    height=chart_settings.get("height", 350)
                )
                fig = apply_dark_theme_template(fig)
                if timestamp: fig = add_timestamp_annotation(fig, timestamp)
        except Exception as e:
            logger.error(f"Error creating Vol Pressure Scatter: {e}")
            fig = create_empty_figure(title="Volatility Pressure vs Delta", reason=f"Plotting error: {e}")
    return html.Div(dcc.Graph(id=f"{component_id_prefix}_vol_pressure_scatter", figure=fig, config={'displayModeBar': False}), className="elite-chart-container")

def _create_top_impact_levels_bar(
    options_data_models: List[ProcessedContractMetricsV2_5],
    component_id_prefix: str,
    config_manager: ConfigManagerV2_5,
    timestamp: Optional[datetime],
    symbol: str,
    n_levels: int = 10
) -> Optional[html.Div]:
    if not options_data_models:
        fig = create_empty_figure(title=f"Top {n_levels} Impact Levels", reason="Data N/A")
    else:
        try:
            df_options = pd.DataFrame([s.model_dump() for s in options_data_models])
            # Ensure ELITE_IMPACT_SCORE and STRIKE are present
            if df_options.empty or EliteImpactColumns.ELITE_IMPACT_SCORE not in df_options.columns or EliteConvexValueColumns.STRIKE not in df_options.columns:
                 fig = create_empty_figure(title=f"Top {n_levels} Impact Levels", reason="Required data for ranking missing")
            else:
                elite_settings = _get_elite_dashboard_settings(config_manager)
                chart_settings = elite_settings.get("top_impact_bar", {})

                # Calculate a combined score for sorting if not already present
                # This should ideally be pre-calculated if it's the primary sort metric
                df_options['ranking_score'] = (
                    abs(df_options[EliteImpactColumns.ELITE_IMPACT_SCORE].fillna(0)) *
                    df_options.get(EliteImpactColumns.SIGNAL_STRENGTH, pd.Series(1.0, index=df_options.index)).fillna(1.0) *
                    df_options.get(EliteImpactColumns.PREDICTION_CONFIDENCE, pd.Series(0.5, index=df_options.index)).fillna(0.5)
                )
                top_n = df_options.nlargest(n_levels, 'ranking_score')

                strike_labels = [
                    f"${r.get(EliteConvexValueColumns.STRIKE, 'N/A'):.0f}{r.get(EliteConvexValueColumns.OPT_KIND, 'N/A')[0].upper()}"
                    if EliteConvexValueColumns.OPT_KIND in r and isinstance(r.get(EliteConvexValueColumns.OPT_KIND), str) and r.get(EliteConvexValueColumns.OPT_KIND)
                    else f"${r.get(EliteConvexValueColumns.STRIKE, 'N/A'):.0f}"
                    for idx, r in top_n.iterrows()
                ]

                fig = go.Figure(data=[go.Bar(
                    x=strike_labels,
                    y=top_n[EliteImpactColumns.ELITE_IMPACT_SCORE],
                    marker_color=chart_settings.get("bar_color", 'gold'),
                    text=top_n[EliteImpactColumns.ELITE_IMPACT_SCORE].round(2),
                    textposition='auto'
                )])
                fig.update_layout(
                    title_text=f'Top {n_levels} Impact Levels by Elite Score',
                    xaxis_title='Strike (Type)',
                    yaxis_title='Elite Impact Score',
                    template=PLOTLY_TEMPLATE,
                    height=chart_settings.get("height", 350),
                    xaxis_tickangle=-45
                )
                fig = apply_dark_theme_template(fig)
                if timestamp: fig = add_timestamp_annotation(fig, timestamp)
        except Exception as e:
            logger.error(f"Error creating Top Impact Levels Bar: {e}")
            fig = create_empty_figure(title=f"Top {n_levels} Impact Levels", reason=f"Plotting error: {e}")
    return html.Div(dcc.Graph(id=f"{component_id_prefix}_top_impact_bar", figure=fig, config={'displayModeBar': False}), className="elite-chart-container")

def create_elite_charts_layout(
    bundle: FinalAnalysisBundleV2_5,
    config_manager: ConfigManagerV2_5,
    main_component_id_prefix: str = "elite-v10" # Prefix for all chart IDs in this layout
) -> html.Div:
    """
    Creates a layout section containing only the new Elite V10 charts.
    This function is intended to be called and its output integrated into the main dashboard.
    """
    if not bundle or not bundle.processed_data_bundle:
        return html.Div(dbc.Alert("Elite V10 chart data not available.", color="warning"))

    processed_data = bundle.processed_data_bundle
    strike_data_models = processed_data.strike_level_data_with_metrics if processed_data.strike_level_data_with_metrics else []
    # options_data_models = processed_data.options_data_with_metrics # For charts needing contract-level data

    symbol = bundle.target_symbol or "Unknown"
    bundle_timestamp = bundle.bundle_timestamp
    current_price = processed_data.underlying_data_enriched.price if processed_data.underlying_data_enriched else None

    # For other charts, create similar functions:
    # _create_elite_score_histogram(...)
    # _create_gamma_wall_plot(...)
    # _create_flow_momentum_heatmap(...)
    # _create_vol_pressure_vs_delta_scatter(...)
    # _create_top_impact_levels_bar(...)

    layout_children = [
        html.H3("Elite V10 Analytics Charts", className="text-elite-primary mb-3")
    ]

    if not strike_data_models and not (hasattr(processed_data, 'options_data_with_metrics') and processed_data.options_data_with_metrics):
        layout_children.append(dbc.Alert("No processed options data available for Elite V10 charts.", color="info"))
    else:
        sdag_chart = create_sdag_vs_strike_scatter(
            strike_data_models, current_price,
            f"{main_component_id_prefix}_sdag",
            config_manager, bundle_timestamp, symbol
        )
        if sdag_chart:
             layout_children.append(dbc.Row([dbc.Col(sdag_chart, md=12)], className="mb-3"))

        # Placeholder for other charts to be added here
        # Example:
        # elite_score_hist_chart = _create_elite_score_histogram(...)
        # layout_children.append(dbc.Row([dbc.Col(elite_score_hist_chart, md=6), ...]))


    return html.Div(layout_children)

# Placeholder for the other chart creation functions:
def _create_elite_score_histogram(
    processed_data: ProcessedDataBundleV2_5, # Assuming this bundle contains necessary data
    component_id_prefix: str,
    config_manager: ConfigManagerV2_5,
    timestamp: Optional[datetime],
    symbol: str
) -> Optional[html.Div]:
    if not processed_data or not processed_data.options_data_with_metrics:
        fig = create_empty_figure(title="Elite Score Distribution", reason="Data N/A")
    else:
        try:
            # Convert Pydantic models to DataFrame for easier processing
            df_options = pd.DataFrame([s.model_dump() for s in processed_data.options_data_with_metrics])
            if df_options.empty or EliteImpactColumns.ELITE_IMPACT_SCORE not in df_options.columns:
                fig = create_empty_figure(title="Elite Score Distribution", reason="Elite Impact Score data missing")
            else:
                elite_settings = _get_elite_dashboard_settings(config_manager)
                chart_settings = elite_settings.get("elite_score_histogram", {})
                elite_scores = df_options[EliteImpactColumns.ELITE_IMPACT_SCORE].dropna()

                fig = go.Figure(data=[go.Histogram(x=elite_scores,
                                                 marker_color=chart_settings.get("color", 'cyan'),
                                                 xbins=dict(size=chart_settings.get("bin_size", 0.1)))])
                fig.add_vline(x=elite_scores.mean(), line_dash="dash", line_color="red",
                              annotation_text=f"Mean: {elite_scores.mean():.3f}")
                fig.update_layout(
                    title_text='Elite Impact Score Distribution',
                    xaxis_title='Elite Impact Score',
                    yaxis_title='Frequency',
                    template=PLOTLY_TEMPLATE,
                    height=chart_settings.get("height", 350)
                )
                fig = apply_dark_theme_template(fig)
                if timestamp: fig = add_timestamp_annotation(fig, timestamp)
        except Exception as e:
            logger.error(f"Error creating Elite Score Histogram: {e}")
            fig = create_empty_figure(title="Elite Score Distribution", reason=f"Plotting error: {e}")
    return html.Div(dcc.Graph(id=f"{component_id_prefix}_elite_score_hist", figure=fig, config={'displayModeBar': False}))

def _create_gamma_wall_plot(
    strike_data_models: List[ProcessedStrikeLevelMetricsV2_5],
    current_price: Optional[float],
    component_id_prefix: str,
    config_manager: ConfigManagerV2_5,
    timestamp: Optional[datetime],
    symbol: str
) -> Optional[html.Div]:
    if not strike_data_models or current_price is None:
        fig = create_empty_figure(title="Gamma Wall Analysis", reason="Data N/A")
    else:
        try:
            df_strike = pd.DataFrame([s.model_dump() for s in strike_data_models])
            if df_strike.empty or EliteImpactColumns.STRIKE_MAGNETISM_INDEX not in df_strike.columns or EliteConvexValueColumns.STRIKE not in df_strike.columns:
                fig = create_empty_figure(title="Gamma Wall Analysis", reason="Required data missing")
            else:
                elite_settings = _get_elite_dashboard_settings(config_manager)
                chart_settings = elite_settings.get("gamma_wall_plot", {})

                df_strike = df_strike.sort_values(by=EliteConvexValueColumns.STRIKE)
                strikes = df_strike[EliteConvexValueColumns.STRIKE]
                gamma_impact = df_strike[EliteImpactColumns.STRIKE_MAGNETISM_INDEX]

                fig = go.Figure(data=[go.Scatter(x=strikes, y=gamma_impact, mode='lines+markers',
                                                 line=dict(color=chart_settings.get("line_color", 'orange')),
                                                 marker=dict(size=chart_settings.get("marker_size", 6)))])
                fig.add_vline(x=current_price, line_dash="solid", line_color="yellow", annotation_text="Current Price")
                fig.update_layout(
                    title_text='Gamma Wall Analysis (Strike Magnetism)',
                    xaxis_title='Strike Price',
                    yaxis_title='Strike Magnetism Index',
                    template=PLOTLY_TEMPLATE,
                    height=chart_settings.get("height", 350)
                )
                fig = apply_dark_theme_template(fig)
                if timestamp: fig = add_timestamp_annotation(fig, timestamp)
        except Exception as e:
            logger.error(f"Error creating Gamma Wall plot: {e}")
            fig = create_empty_figure(title="Gamma Wall Analysis", reason=f"Plotting error: {e}")
    return html.Div(dcc.Graph(id=f"{component_id_prefix}_gamma_wall", figure=fig, config={'displayModeBar': False}))

def _create_flow_momentum_heatmap(
    options_data_models: List[ProcessedContractMetricsV2_5], # Takes contract level data
    component_id_prefix: str,
    config_manager: ConfigManagerV2_5,
    timestamp: Optional[datetime],
    symbol: str,
    num_options_to_display: int = 20
) -> Optional[html.Div]:
    if not options_data_models:
        fig = create_empty_figure(title="Flow Momentum Heatmap", reason="Data N/A")
    else:
        try:
            df_options = pd.DataFrame([s.model_dump() for s in options_data_models])
            momentum_cols = [
                EliteImpactColumns.FLOW_VELOCITY_5M, EliteImpactColumns.FLOW_VELOCITY_15M,
                EliteImpactColumns.FLOW_ACCELERATION, EliteImpactColumns.MOMENTUM_PERSISTENCE
            ]
            if df_options.empty or not all(col in df_options.columns for col in momentum_cols):
                fig = create_empty_figure(title="Flow Momentum Heatmap", reason="Momentum data missing")
            else:
                elite_settings = _get_elite_dashboard_settings(config_manager)
                chart_settings = elite_settings.get("flow_momentum_heatmap", {})

                # Select a subset of options for display if too many (e.g., by highest OI or volume)
                if len(df_options) > num_options_to_display and EliteConvexValueColumns.OI in df_options.columns:
                    df_display = df_options.nlargest(num_options_to_display, EliteConvexValueColumns.OI)
                else:
                    df_display = df_options.head(num_options_to_display)

                momentum_data = df_display[momentum_cols].fillna(0)

                # For heatmap, it's better to show strike and type rather than index
                y_labels = [f"{r[EliteConvexValueColumns.STRIKE]} {r[EliteConvexValueColumns.OPT_KIND][0].upper()}" for i,r in df_display.iterrows()]


                fig = go.Figure(data=go.Heatmap(
                    z=momentum_data.values.T, # Transpose for correct orientation
                    x=y_labels, # Use custom labels for x-axis (options)
                    y=[col.replace('flow_','').replace('_',' ').title() for col in momentum_cols], # Y-axis are momentum types
                    colorscale=chart_settings.get("colorscale", 'RdYlBu_r'),
                    zmid=0, # Center color scale at 0
                    colorbar_title="Momentum Strength"
                ))
                fig.update_layout(
                    title_text='Flow Momentum Analysis (Top Options)',
                    xaxis_title='Option (Strike Type)',
                    yaxis_title='Momentum Metric',
                    template=PLOTLY_TEMPLATE,
                    height=chart_settings.get("height", 400),
                    xaxis_tickangle=-45
                )
                fig = apply_dark_theme_template(fig)
                if timestamp: fig = add_timestamp_annotation(fig, timestamp)
        except Exception as e:
            logger.error(f"Error creating Flow Momentum Heatmap: {e}")
            fig = create_empty_figure(title="Flow Momentum Heatmap", reason=f"Plotting error: {e}")
    return html.Div(dcc.Graph(id=f"{component_id_prefix}_flow_momentum_heatmap", figure=fig, config={'displayModeBar': False}))


def _create_vol_pressure_vs_delta_scatter(
    options_data_models: List[ProcessedContractMetricsV2_5], # Takes contract level data
    component_id_prefix: str,
    config_manager: ConfigManagerV2_5,
    timestamp: Optional[datetime],
    symbol: str
) -> Optional[html.Div]:
    if not options_data_models:
        fig = create_empty_figure(title="Volatility Pressure vs Delta", reason="Data N/A")
    else:
        try:
            df_options = pd.DataFrame([s.model_dump() for s in options_data_models])
            req_cols = [EliteImpactColumns.VOLATILITY_PRESSURE_INDEX, EliteImpactColumns.REGIME_ADJUSTED_DELTA]
            if df_options.empty or not all(col in df_options.columns for col in req_cols):
                fig = create_empty_figure(title="Volatility Pressure vs Delta", reason="Required data missing")
            else:
                elite_settings = _get_elite_dashboard_settings(config_manager)
                chart_settings = elite_settings.get("vol_pressure_scatter", {})

                vol_pressure = df_options[EliteImpactColumns.VOLATILITY_PRESSURE_INDEX]
                delta_exposure = df_options[EliteImpactColumns.REGIME_ADJUSTED_DELTA]
                signal_strength = df_options.get(EliteImpactColumns.SIGNAL_STRENGTH, pd.Series(0.5, index=df_options.index))

                fig = go.Figure()
                scatter_trace = go.Scatter(
                    x=delta_exposure, y=vol_pressure, mode='markers',
                    marker=dict(
                        size=chart_settings.get("marker_size", 8),
                        color=signal_strength,
                        colorscale=chart_settings.get("colorscale", 'Plasma'),
                        showscale=True,
                        colorbar_title="Signal Strength"
                    ),
                    text=[f"Delta Exp: {d:.2f}<br>Vol Pressure: {vp:.2f}<br>Signal: {ss:.2f}" for d,vp,ss in zip(delta_exposure, vol_pressure, signal_strength)],
                    hoverinfo='text'
                )
                fig.add_trace(scatter_trace)
                fig.update_layout(
                    title_text='Volatility Pressure vs Delta Exposure',
                    xaxis_title='Regime Adjusted Delta Exposure',
                    yaxis_title='Volatility Pressure Index',
                    template=PLOTLY_TEMPLATE,
                    height=chart_settings.get("height", 350)
                )
                fig = apply_dark_theme_template(fig)
                if timestamp: fig = add_timestamp_annotation(fig, timestamp)
        except Exception as e:
            logger.error(f"Error creating Vol Pressure Scatter: {e}")
            fig = create_empty_figure(title="Volatility Pressure vs Delta", reason=f"Plotting error: {e}")
    return html.Div(dcc.Graph(id=f"{component_id_prefix}_vol_pressure_scatter", figure=fig, config={'displayModeBar': False}))

def _create_top_impact_levels_bar(
    options_data_models: List[ProcessedContractMetricsV2_5], # Takes contract level data
    component_id_prefix: str,
    config_manager: ConfigManagerV2_5,
    timestamp: Optional[datetime],
    symbol: str,
    n_levels: int = 10
) -> Optional[html.Div]:
    if not options_data_models:
        fig = create_empty_figure(title=f"Top {n_levels} Impact Levels", reason="Data N/A")
    else:
        try:
            df_options = pd.DataFrame([s.model_dump() for s in options_data_models])
            if df_options.empty or EliteImpactColumns.ELITE_IMPACT_SCORE not in df_options.columns or EliteConvexValueColumns.STRIKE not in df_options.columns:
                fig = create_empty_figure(title=f"Top {n_levels} Impact Levels", reason="Required data missing")
            else:
                elite_settings = _get_elite_dashboard_settings(config_manager)
                chart_settings = elite_settings.get("top_impact_bar", {})

                # Calculate a combined score for sorting if not already present
                if 'combined_score_for_ranking' not in df_options.columns:
                     df_options['combined_score_for_ranking'] = (abs(df_options[EliteImpactColumns.ELITE_IMPACT_SCORE].fillna(0)) *
                                                               df_options.get(EliteImpactColumns.SIGNAL_STRENGTH, pd.Series(1.0, index=df_options.index)).fillna(1.0) *
                                                               df_options.get(EliteImpactColumns.PREDICTION_CONFIDENCE, pd.Series(0.5, index=df_options.index)).fillna(0.5))

                top_n = df_options.nlargest(n_levels, 'combined_score_for_ranking')

                # Create strike labels like "$4500C" or "$4450P"
                strike_labels = [
                    f"${r[EliteConvexValueColumns.STRIKE]:.0f}{r[EliteConvexValueColumns.OPT_KIND][0].upper()}"
                    if EliteConvexValueColumns.OPT_KIND in r and r[EliteConvexValueColumns.OPT_KIND]
                    else f"${r[EliteConvexValueColumns.STRIKE]:.0f}"
                    for i, r in top_n.iterrows()
                ]

                fig = go.Figure(data=[go.Bar(
                    x=strike_labels,
                    y=top_n[EliteImpactColumns.ELITE_IMPACT_SCORE],
                    marker_color=chart_settings.get("bar_color", 'gold')
                )])
                fig.update_layout(
                    title_text=f'Top {n_levels} Impact Levels by Elite Score',
                    xaxis_title='Strike (Type)',
                    yaxis_title='Elite Impact Score',
                    template=PLOTLY_TEMPLATE,
                    height=chart_settings.get("height", 350),
                    xaxis_tickangle=-45
                )
                fig = apply_dark_theme_template(fig)
                if timestamp: fig = add_timestamp_annotation(fig, timestamp)
        except Exception as e:
            logger.error(f"Error creating Top Impact Levels Bar: {e}")
            fig = create_empty_figure(title=f"Top {n_levels} Impact Levels", reason=f"Plotting error: {e}")
    return html.Div(dcc.Graph(id=f"{component_id_prefix}_top_impact_bar", figure=fig, config={'displayModeBar': False}))


if __name__ == '__main__':
    # This block can be used for testing this module independently if needed.
    # Create mock ConfigManagerV2_5 and FinalAnalysisBundleV2_5 for testing.
    logger.info("main_dashboard_display_elite_charts_v10.py running in standalone test mode.")
    # Add test code here if desired.
    pass
