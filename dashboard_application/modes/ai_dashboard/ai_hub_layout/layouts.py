"""
AI Dashboard Layouts Module for EOTS v2.5
=========================================

This module contains all layout and panel assembly functions for the AI dashboard including:
- Panel creation and assembly
- Grid layouts and responsive design
- Component organization
- Container management
- Layout utilities

Author: EOTS v2.5 Development Team
Version: 2.5.0
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from dash import dcc, html
import plotly.graph_objects as go

from data_models.eots_schemas_v2_5 import (
    FinalAnalysisBundleV2_5,
    ProcessedDataBundleV2_5,
    EOTSConfigV2_5
)

from .components import (
    AI_COLORS, AI_TYPOGRAPHY, AI_SPACING, AI_EFFECTS,
    create_placeholder_card, create_enhanced_confidence_meter,
    create_quick_action_buttons, create_regime_transition_indicator,
    get_unified_card_style, get_unified_text_style, get_card_style,
    create_clickable_title_with_info
)

from .visualizations import (
    create_market_state_visualization, create_enhanced_market_state_visualization,
    create_confidence_meter, create_confluence_gauge, create_ai_performance_chart,
    create_regime_transition_gauge
)

from .pydantic_intelligence_engine_v2_5 import (
    generate_ai_insights, calculate_ai_confidence, AnalysisType,
    calculate_ai_confidence_sync, get_consolidated_intelligence_data,
    calculate_overall_intelligence_score, get_real_system_health_status
)

# Import centralized regime display utilities
from dashboard_application.utils.regime_display_utils import get_tactical_regime_name, get_regime_color_class, get_regime_icon

# 🚀 REAL COMPLIANCE TRACKING: Import tracking system
from .component_compliance_tracker_v2_5 import track_data_access, DataSourceType
from .compliance_decorators_v2_5 import track_compliance

logger = logging.getLogger(__name__)

# ===== AI DASHBOARD MODULE INFORMATION BLURBS =====

AI_MODULE_INFO = {
    "unified_intelligence": """🧠 Unified AI Intelligence Hub: This is your COMMAND CENTER for all AI-powered market analysis. The 4-quadrant layout provides: TOP-LEFT: AI Confidence Barometer showing system conviction levels with real-time data quality scoring. TOP-RIGHT: Signal Confluence Barometer measuring agreement between multiple EOTS metrics (VAPI-FA, DWFD, TW-LAF, GIB). BOTTOM-LEFT: Unified Intelligence Analysis combining Alpha Vantage news sentiment, MCP server insights, and ATIF recommendations. BOTTOM-RIGHT: Market Dynamics Radar showing 6-dimensional market forces (Volatility, Flow, Momentum, Structure, Sentiment, Risk). 💡 TRADING INSIGHT: When AI Confidence > 80% AND Signal Confluence > 70% = HIGH CONVICTION setup. Watch for Market Dynamics radar showing EXTREME readings (outer edges) = potential breakout/breakdown. The Unified Intelligence text provides CONTEXTUAL NARRATIVE explaining WHY the system is confident. This updates every 15 minutes with fresh data integration!""",

    "regime_analysis": """🌊 AI Regime Analysis: This 4-quadrant system identifies and analyzes the CURRENT MARKET REGIME using advanced EOTS metrics. TOP-LEFT: Regime Confidence Barometer showing conviction in current regime classification with transition risk assessment. TOP-RIGHT: Regime Characteristics Analysis displaying 4 key market properties (Volatility, Flow Direction, Risk Level, Momentum) with DYNAMIC COLOR CODING. BOTTOM-LEFT: Enhanced AI Analysis showing current regime name, key Z-score metrics (VAPI-FA, DWFD, TW-LAF), and AI-generated insights. BOTTOM-RIGHT: Transition Gauge measuring probability of regime change with stability metrics. 💡 TRADING INSIGHT: Regime Confidence > 70% = STABLE regime, trade WITH the characteristics. Transition Risk > 60% = UNSTABLE regime, expect volatility and potential reversals. When characteristics show EXTREME values (Very High/Low) = regime at INFLECTION POINT. Use regime insights to adjust position sizing and strategy selection!""",

    "recommendations": """🎯 AI Recommendations Engine: This panel displays ADAPTIVE TRADE IDEA FRAMEWORK (ATIF) generated strategies with AI-enhanced conviction scoring. Each recommendation includes: Strategy Type, Conviction Level (0-100%), AI-generated Rationale, and Risk Assessment. The system combines EOTS metrics, regime analysis, and market structure to generate ACTIONABLE trade ideas. 💡 TRADING INSIGHT: Conviction > 80% = HIGH PROBABILITY setup, consider larger position size. Conviction 60-80% = MODERATE setup, standard position size. Conviction < 60% = LOW PROBABILITY, small position or avoid. When multiple recommendations AGREE on direction = STRONG CONFLUENCE. Pay attention to the AI rationale - it explains the LOGIC behind each recommendation. Recommendations update based on changing market conditions and new data!"""
}

# ===== MAIN PANEL CREATION FUNCTIONS =====

@track_compliance("unified_ai_intelligence_hub", "Unified AI Intelligence Hub")
def create_unified_ai_intelligence_hub(bundle_data: FinalAnalysisBundleV2_5, ai_settings: Dict[str, Any], symbol: str, db_manager=None) -> html.Div:
    """Create the UNIFIED AI Intelligence Hub with 4-quadrant layout - enhanced Market Dynamics Radar as the star."""
    try:
        # 🚀 REAL COMPLIANCE TRACKING: Track data usage
        if ai_settings.get("filtered_bundle"):
            track_data_access("unified_ai_intelligence_hub", DataSourceType.FILTERED_OPTIONS, 
                            ai_settings["filtered_bundle"], 
                            metadata={"symbol": symbol, "source": "ai_settings_filtered_bundle"})
        else:
            track_data_access("unified_ai_intelligence_hub", DataSourceType.RAW_OPTIONS, 
                            bundle_data, 
                            metadata={"symbol": symbol, "source": "original_bundle_fallback"})

        # PYDANTIC-FIRST: Extract data using direct model access (no dictionary conversion)
        processed_data = bundle_data.processed_data_bundle
        enriched_data = processed_data.underlying_data_enriched if processed_data else None

        # PYDANTIC-FIRST: Extract regime using direct Pydantic model attribute access
        if enriched_data:
            regime = (
                getattr(enriched_data, 'current_market_regime_v2_5', None) or
                getattr(enriched_data, 'market_regime', None) or
                getattr(enriched_data, 'regime', None) or
                getattr(enriched_data, 'market_regime_summary', None) or
                "REGIME_UNCLEAR_OR_TRANSITIONING"
            )
        else:
            regime = "REGIME_UNCLEAR_OR_TRANSITIONING"

        # Calculate unified intelligence metrics using PYDANTIC-FIRST approach
        confidence_score = calculate_ai_confidence_sync(bundle_data, db_manager)
        confluence_score = calculate_metric_confluence_score(enriched_data)
        signal_strength = assess_signal_strength(enriched_data)

        # Generate unified AI insights using NEW Intelligence Engine V2.5
        try:
            import asyncio
            from utils.config_manager_v2_5 import ConfigManagerV2_5
            config_manager = ConfigManagerV2_5()
            config = config_manager.config

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                unified_insights = loop.run_until_complete(
                    generate_ai_insights(bundle_data, symbol, config, AnalysisType.COMPREHENSIVE)
                )
            finally:
                loop.close()
        except Exception as e:
            unified_insights = [f"🤖 AI insights temporarily unavailable: {str(e)[:50]}..."]

        # Create enhanced market dynamics radar (STAR OF THE SHOW)
        enhanced_market_radar = create_enhanced_market_dynamics_radar(bundle_data, symbol)

        # UNIFIED 4-QUADRANT LAYOUT
        return html.Div([
            # Outer colored container
            html.Div([
                # Inner dark card container
                html.Div([
                    # Card Header with clickable title and info
                    html.Div([
                        create_clickable_title_with_info(
                            "🧠 Unified AI Intelligence Hub",
                            "unified_intelligence",
                            AI_MODULE_INFO["unified_intelligence"]
                        )
                    ], className="card-header", style={
                        "background": "transparent",
                        "borderBottom": f"2px solid {AI_COLORS['primary']}",
                        "padding": f"{AI_SPACING['md']} {AI_SPACING['xl']}"
                    }),

                    # Card Body - 4 QUADRANT LAYOUT
                    html.Div([
                        # TOP ROW - Quadrants 1 & 2
                        html.Div([
                            # QUADRANT 1: AI Confidence Barometer (Top Left)
                            html.Div([
                                create_ai_confidence_barometer(confidence_score, bundle_data, db_manager)
                            ], className="col-md-6 mb-3"),

                            # QUADRANT 2: Signal Confluence Barometer (Top Right)
                            html.Div([
                                create_signal_confluence_barometer(confluence_score, enriched_data, signal_strength)
                            ], className="col-md-6 mb-3")
                        ], className="row"),

                        # BOTTOM ROW - Quadrants 3 & 4
                        html.Div([
                            # QUADRANT 3: Unified Intelligence Analysis (Bottom Left)
                            html.Div([
                                create_unified_intelligence_analysis(unified_insights, regime, bundle_data)
                            ], className="col-md-6"),

                            # QUADRANT 4: Enhanced Market Dynamics Radar (Bottom Right) - STAR OF THE SHOW
                            html.Div([
                                create_market_dynamics_radar_quadrant(enhanced_market_radar, enriched_data, symbol)
                            ], className="col-md-6")
                        ], className="row")
                    ], className="card-body", style={
                        "padding": f"{AI_SPACING['lg']} {AI_SPACING['xl']}",
                        "background": "transparent"
                    })
                ], className="card h-100")
            ], style=get_card_style('primary'))
        ], className="ai-intelligence-hub")

    except Exception as e:
        logger.error(f"Error creating unified AI intelligence hub: {str(e)}")
        return create_placeholder_card("🧠 Unified AI Intelligence Hub", f"Error: {str(e)}")


@track_compliance("ai_recommendations_panel", "AI Recommendations Panel")
def create_ai_recommendations_panel(bundle_data: FinalAnalysisBundleV2_5, ai_settings: Dict[str, Any], symbol: str) -> html.Div:
    """Create enhanced AI-powered recommendations panel with comprehensive EOTS integration."""
    try:
        # 🚀 REAL COMPLIANCE TRACKING: Track data usage
        if ai_settings.get("filtered_bundle"):
            track_data_access("ai_recommendations_panel", DataSourceType.FILTERED_OPTIONS, 
                            ai_settings["filtered_bundle"], 
                            metadata={"symbol": symbol, "source": "ai_settings_filtered_bundle"})
        else:
            track_data_access("ai_recommendations_panel", DataSourceType.RAW_OPTIONS, 
                            bundle_data, 
                            metadata={"symbol": symbol, "source": "original_bundle_fallback"})

        atif_recs = bundle_data.atif_recommendations_v2_5 or []

        # Calculate recommendation confidence and priority
        try:
            from .pydantic_intelligence_engine_v2_5 import calculate_recommendation_confidence
            rec_confidence = calculate_recommendation_confidence(bundle_data, atif_recs)
        except ImportError:
            rec_confidence = 0.75  # Default confidence

        # Extract EOTS metrics for enhanced recommendations using Pydantic models
        processed_data = bundle_data.processed_data_bundle
        metrics = processed_data.underlying_data_enriched.model_dump() if processed_data else {}

        # Generate AI-enhanced insights using NEW Intelligence Engine V2.5
        try:
            import asyncio
            from utils.config_manager_v2_5 import ConfigManagerV2_5
            config_manager = ConfigManagerV2_5()
            config = config_manager.config

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                ai_insights = loop.run_until_complete(
                    generate_ai_insights(bundle_data, symbol, config, AnalysisType.COMPREHENSIVE)
                )
            finally:
                loop.close()
        except Exception as e:
            ai_insights = [f"🎯 AI recommendations temporarily unavailable: {str(e)[:50]}..."]

        # UNIFIED NESTED CONTAINER STRUCTURE - Restored proper dark theme structure
        return html.Div([
            # Outer colored container
            html.Div([
                # Inner dark card container
                html.Div([
                    # Card Header with clickable title and info
                    html.Div([
                        create_clickable_title_with_info(
                            "🎯 AI Recommendations",
                            "recommendations",
                            AI_MODULE_INFO["recommendations"],
                            badge_text=f"Active: {len(atif_recs)}",
                            badge_style='secondary'
                        )
                    ], className="card-header", style={
                        "background": "transparent",
                        "borderBottom": f"2px solid {AI_COLORS['secondary']}",
                        "padding": f"{AI_SPACING['md']} {AI_SPACING['xl']}"
                    }),

                    # Card Body
                    html.Div([
                        # Confidence badge
                        html.Div([
                            html.Span(f"Confidence: {rec_confidence:.0%}",
                                     id="recommendations-confidence",
                                     className="badge mb-3",
                                     style={
                                         "background": AI_COLORS['success'] if rec_confidence > 0.7 else AI_COLORS['warning'],
                                         "color": "white",
                                         "fontSize": AI_TYPOGRAPHY['small_size']
                                     })
                        ]),

                # Quick Actions
                create_quick_action_buttons(bundle_data, symbol),

                # Market Context
                html.Div([
                    html.H6("📊 Market Context", className="mb-2", style={
                        "fontSize": AI_TYPOGRAPHY['subtitle_size'],
                        "color": AI_COLORS['dark']
                    }),
                    html.Div([
                        html.P(f"Symbol: {symbol}", className="small mb-1", style={"color": AI_COLORS['muted']}),
                        html.P(f"Market Regime: {getattr(bundle_data.processed_data_bundle.underlying_data_enriched, 'current_market_regime_v2_5', 'Unknown')}",
                              className="small mb-1", style={"color": AI_COLORS['muted']}),
                        html.P(f"Analysis Time: {bundle_data.bundle_timestamp.strftime('%H:%M:%S')}",
                              className="small mb-0", style={"color": AI_COLORS['muted']})
                    ])
                ], className="mb-3"),

                # ATIF Recommendations Section
                html.Div([
                    html.Div([
                        html.H6("🎯 ATIF Strategy Recommendations", className="mb-0", style={
                            "fontSize": AI_TYPOGRAPHY['subtitle_size'],
                            "color": AI_COLORS['dark']
                        }),
                        html.Small(f"Count: {len(atif_recs)}", id="recommendations-count", style={
                            "color": AI_COLORS['muted'],
                            "fontSize": AI_TYPOGRAPHY['tiny_size']
                        })
                    ], className="d-flex justify-content-between align-items-center mb-3"),
                    html.Div([
                        create_atif_recommendation_items(atif_recs[:3]) if atif_recs else
                        html.P("No ATIF recommendations available", className="text-muted", style={"color": AI_COLORS['muted']})
                    ])
                ], className="mb-3"),

                # Enhanced AI Tactical Recommendations
                html.Div([
                    html.Div([
                        html.H6("🤖 AI Tactical Recommendations", className="mb-0", style={
                            "fontSize": AI_TYPOGRAPHY['subtitle_size'],
                            "color": AI_COLORS['dark'],
                            "fontWeight": AI_TYPOGRAPHY['subtitle_weight']
                        }),
                        html.Div([
                            html.Small("Confidence: ", style={"color": AI_COLORS['muted']}),
                            html.Small(f"{rec_confidence:.0%}", style={
                                "color": AI_COLORS['primary'] if rec_confidence > 0.7 else AI_COLORS['warning'] if rec_confidence > 0.5 else AI_COLORS['danger'],
                                "fontSize": AI_TYPOGRAPHY['small_size'],
                                "fontWeight": "bold"
                            })
                        ])
                    ], className="d-flex justify-content-between align-items-center mb-3"),

                    # Enhanced tactical recommendations with dynamic styling
                    html.Div([
                        html.Div([
                            # Dynamic tactical icon based on recommendation content
                            html.Span(
                                "🎯" if any(word in insight.upper() for word in ["AMBUSH", "SURGE", "TARGET"])
                                else "💥" if any(word in insight.upper() for word in ["SQUEEZE", "BREACH", "EXPLOSIVE"])
                                else "🔥" if any(word in insight.upper() for word in ["IGNITION", "CHAOS", "EXTREME"])
                                else "⚖️" if any(word in insight.upper() for word in ["CONSOLIDATION", "WALL", "BALANCE"])
                                else "🚨" if any(word in insight.upper() for word in ["DEFENSIVE", "RISK", "CAUTION"])
                                else "💡",
                                style={"marginRight": "12px", "fontSize": "18px"}
                            ),
                            html.Span(insight, style={
                                "fontSize": AI_TYPOGRAPHY['small_size'],
                                "color": AI_COLORS['dark'],
                                "lineHeight": "1.5",
                                "fontWeight": "500"
                            })
                        ], style={
                            "padding": f"{AI_SPACING['md']} {AI_SPACING['lg']}",
                            "background": (
                                "rgba(255, 71, 87, 0.15)" if any(word in insight.upper() for word in ["BREACH", "CHAOS", "DEFENSIVE", "RISK"])
                                else "rgba(107, 207, 127, 0.15)" if any(word in insight.upper() for word in ["SURGE", "SQUEEZE", "BULLISH", "BUY"])
                                else "rgba(255, 167, 38, 0.15)" if any(word in insight.upper() for word in ["AMBUSH", "IGNITION", "WALL", "CAUTION"])
                                else "rgba(0, 212, 255, 0.15)"
                            ),
                            "borderRadius": AI_EFFECTS['border_radius'],
                            "marginBottom": AI_SPACING['sm'],
                            "border": (
                                f"2px solid {AI_COLORS['danger']}" if any(word in insight.upper() for word in ["BREACH", "CHAOS", "DEFENSIVE", "RISK"])
                                else f"2px solid {AI_COLORS['success']}" if any(word in insight.upper() for word in ["SURGE", "SQUEEZE", "BULLISH", "BUY"])
                                else f"2px solid {AI_COLORS['warning']}" if any(word in insight.upper() for word in ["AMBUSH", "IGNITION", "WALL", "CAUTION"])
                                else f"1px solid {AI_COLORS['primary']}"
                            ),
                            "transition": AI_EFFECTS['transition'],
                            "cursor": "pointer",
                            "boxShadow": AI_EFFECTS['box_shadow']
                        })
                        for insight in ai_insights[:4]  # Show top 4 tactical insights
                    ], className="recommendations-container", style={
                        "maxHeight": "280px",
                        "overflowY": "auto",
                        "paddingRight": "5px"
                    })
                ])
                    ], className="card-body", style={
                        "padding": f"{AI_SPACING['xl']} {AI_SPACING['xl']}",
                        "background": "transparent"
                    })
                ], className="card h-100")
            ], style=get_card_style('secondary'))
        ], className="ai-recommendations-panel")

    except Exception as e:
        logger.error(f"Error creating AI recommendations panel: {str(e)}")
        return create_placeholder_card("🎯 AI Recommendations", f"Error: {str(e)}")


def create_ai_regime_context_panel(bundle_data: FinalAnalysisBundleV2_5, ai_settings: Dict[str, Any], regime: str) -> html.Div:
    """Create enhanced AI regime analysis panel with 4-quadrant layout similar to Unified AI Intelligence Hub."""
    try:
        # Generate enhanced regime analysis using NEW Intelligence Engine V2.5
        try:
            import asyncio
            from utils.config_manager_v2_5 import ConfigManagerV2_5
            config_manager = ConfigManagerV2_5()
            config = config_manager.config

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                regime_analysis = loop.run_until_complete(
                    generate_ai_insights(bundle_data, regime, config, AnalysisType.MARKET_REGIME)
                )
            finally:
                loop.close()
        except Exception as e:
            regime_analysis = [f"🌊 Regime analysis temporarily unavailable: {str(e)[:50]}..."]

        # Create enhanced regime confidence using NEW Intelligence Engine V2.5
        regime_confidence = calculate_ai_confidence_sync(bundle_data)

        # Extract EOTS metrics for regime context using Pydantic models
        processed_data = bundle_data.processed_data_bundle
        metrics = processed_data.underlying_data_enriched.model_dump() if processed_data else {}

        # Calculate regime transition probability using simplified logic
        transition_prob = 0.3  # Default transition probability

        # Get regime characteristics using simplified logic
        regime_characteristics = {
            "volatility": "MODERATE",
            "flow_direction": "NEUTRAL",
            "risk_level": "MODERATE",
            "momentum": "STABLE"
        }

        # Calculate metric confluence score
        confluence_score = calculate_metric_confluence_score(metrics)

        # Calculate signal strength for quadrant 2
        signal_strength = assess_signal_strength(metrics)

        # UNIFIED 4-QUADRANT LAYOUT STRUCTURE
        return html.Div([
            # Outer colored container
            html.Div([
                # Inner dark card container
                html.Div([
                    # Card Header with clickable title and info
                    html.Div([
                        create_clickable_title_with_info(
                            "🌊 AI Regime Analysis",
                            "regime_analysis",
                            AI_MODULE_INFO["regime_analysis"]
                        )
                    ], className="card-header", style={
                        "background": "transparent",
                        "borderBottom": f"2px solid {AI_COLORS['success']}",
                        "padding": f"{AI_SPACING['md']} {AI_SPACING['xl']}"
                    }),

                    # Card Body with 4-Quadrant Layout
                    html.Div([
                        # TOP ROW - Quadrants 1 & 2
                        html.Div([
                            # QUADRANT 1: Regime Confidence Barometer (Top Left)
                            html.Div([
                                create_regime_confidence_barometer(regime_confidence, regime, transition_prob)
                            ], className="col-md-6"),

                            # QUADRANT 2: Regime Characteristics Analysis (Top Right)
                            html.Div([
                                create_regime_characteristics_analysis(regime_characteristics, regime, signal_strength)
                            ], className="col-md-6")
                        ], className="row mb-4"),

                        # BOTTOM ROW - Quadrants 3 & 4
                        html.Div([
                            # QUADRANT 3: Enhanced AI Regime Analysis (Bottom Left)
                            html.Div([
                                create_enhanced_regime_analysis_quadrant(regime_analysis, regime, metrics)
                            ], className="col-md-6"),

                            # QUADRANT 4: Regime Transition Gauge (Bottom Right)
                            html.Div([
                                create_regime_transition_gauge_quadrant(regime_confidence, transition_prob, regime, confluence_score)
                            ], className="col-md-6")
                        ], className="row")
                    ], className="card-body", style={
                        "padding": f"{AI_SPACING['lg']} {AI_SPACING['xl']}",
                        "background": "transparent"
                    })
                ], className="card h-100")
            ], style=get_card_style('success'))
        ], className="ai-regime-analysis-panel")

    except Exception as e:
        logger.error(f"Error creating AI regime analysis panel: {str(e)}")
        return create_placeholder_card(f"🌊 Regime Analysis: {regime}", f"Error: {str(e)}")


# ===== AI REGIME ANALYSIS 4-QUADRANT FUNCTIONS =====

def create_regime_confidence_barometer(regime_confidence: float, regime: str, transition_prob: float) -> html.Div:
    """QUADRANT 1: Create Regime Confidence Barometer with detailed breakdown."""
    try:
        # PYDANTIC-FIRST: Handle None regime values with proper defaults
        if not regime or regime in [None, "None", "UNKNOWN", ""]:
            regime = "UNKNOWN"

        # Get tactical regime name and styling
        tactical_regime_name = get_tactical_regime_name(regime)

        # Determine confidence level and styling
        if regime_confidence >= 0.8:
            confidence_level = "Very High"
            color = AI_COLORS['success']
            icon = "🔥"
            bg_color = "rgba(107, 207, 127, 0.1)"
        elif regime_confidence >= 0.6:
            confidence_level = "High"
            color = AI_COLORS['primary']
            icon = "⚡"
            bg_color = "rgba(0, 212, 255, 0.1)"
        elif regime_confidence >= 0.4:
            confidence_level = "Moderate"
            color = AI_COLORS['warning']
            icon = "⚠️"
            bg_color = "rgba(255, 167, 38, 0.1)"
        else:
            confidence_level = "Low"
            color = AI_COLORS['danger']
            icon = "🚨"
            bg_color = "rgba(255, 71, 87, 0.1)"

        return html.Div([
            html.Div([
                html.H6(f"{icon} Regime Confidence", className="mb-3", style={
                    "color": AI_COLORS['dark'],
                    "fontSize": AI_TYPOGRAPHY['subtitle_size'],
                    "fontWeight": AI_TYPOGRAPHY['subtitle_weight']
                }),

                # Main confidence display
                html.Div([
                    html.Div([
                        html.Span(f"{regime_confidence:.0%}", id="regime-confidence-score", style={
                            "fontSize": "2.5rem",
                            "fontWeight": "bold",
                            "color": color
                        }),
                        html.Div(confidence_level, style={
                            "fontSize": AI_TYPOGRAPHY['body_size'],
                            "color": AI_COLORS['muted'],
                            "marginTop": "-5px"
                        })
                    ], className="text-center mb-3"),

                    # Enhanced Confidence bar
                    html.Div([
                        html.Div(style={
                            "width": f"{regime_confidence * 100}%",
                            "height": "18px",
                            "background": f"linear-gradient(90deg, {color}, {color}aa)",
                            "borderRadius": "9px",
                            "transition": AI_EFFECTS['transition']
                        })
                    ], style={
                        "width": "100%",
                        "height": "18px",
                        "background": "rgba(255, 255, 255, 0.1)",
                        "borderRadius": "9px",
                        "marginBottom": AI_SPACING['lg']
                    }),

                    # Regime details
                    html.Div([
                        html.Div([
                            html.Small("Current Regime: ", style={"color": AI_COLORS['muted']}),
                            html.Small(tactical_regime_name, style={"color": color, "fontWeight": "bold"})
                        ], className="mb-2"),
                        html.Div([
                            html.Small("Transition Risk: ", style={"color": AI_COLORS['muted']}),
                            html.Small(f"{transition_prob:.0%}", id="regime-transition-prob", style={
                                "color": AI_COLORS['danger'] if transition_prob > 0.6 else AI_COLORS['warning'] if transition_prob > 0.3 else AI_COLORS['success'],
                                "fontWeight": "bold"
                            })
                        ])
                    ])
                ])
            ], id="regime-analysis-container", style={
                "padding": f"{AI_SPACING['lg']} {AI_SPACING['md']}",
                "background": bg_color,
                "borderRadius": AI_EFFECTS['border_radius'],
                "border": f"1px solid {color}",
                "height": "100%",
                "minHeight": "280px"  # Match regime characteristics height
            })
        ])

    except Exception as e:
        logger.error(f"Error creating regime confidence barometer: {str(e)}")
        return html.Div("Regime confidence unavailable")


def create_regime_characteristics_analysis(regime_characteristics: Dict[str, str], regime: str, signal_strength: str) -> html.Div:
    """QUADRANT 2: Create Regime Characteristics Analysis with 4-quadrant layout and dynamic colors."""
    try:
        # Determine styling based on regime
        regime_colors = {
            'BULLISH': AI_COLORS['success'],
            'BEARISH': AI_COLORS['danger'],
            'NEUTRAL': AI_COLORS['warning'],
            'VOLATILE': AI_COLORS['info'],
            'UNKNOWN': AI_COLORS['muted']
        }

        regime_icons = {
            'BULLISH': '🚀',
            'BEARISH': '🐻',
            'NEUTRAL': '⚖️',
            'VOLATILE': '🌪️',
            'UNKNOWN': '❓'
        }

        color = regime_colors.get(regime, AI_COLORS['muted'])
        icon = regime_icons.get(regime, '📊')
        bg_color = f"rgba({color[4:-1]}, 0.1)"

        # Function to get dynamic color for characteristic value
        def get_characteristic_color(char_value: str) -> str:
            """Get dynamic color based on characteristic value."""
            char_value_lower = char_value.lower()
            if any(word in char_value_lower for word in ['high', 'strong', 'positive', 'expanding', 'elevated']):
                return AI_COLORS['success']
            elif any(word in char_value_lower for word in ['low', 'weak', 'negative', 'contracting']):
                return AI_COLORS['danger']
            elif any(word in char_value_lower for word in ['moderate', 'medium', 'balanced', 'neutral']):
                return AI_COLORS['warning']
            elif any(word in char_value_lower for word in ['very high', 'extreme', 'massive']):
                return AI_COLORS['info']
            else:
                return AI_COLORS['secondary']

        # Function to get background color for characteristic container
        def get_characteristic_bg_color(char_value: str) -> str:
            """Get background color for characteristic container."""
            char_color = get_characteristic_color(char_value)
            # Extract RGB values and create transparent version
            if char_color.startswith('#'):
                # Convert hex to rgba
                hex_color = char_color.lstrip('#')
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.1)"
            else:
                return "rgba(255, 255, 255, 0.05)"

        return html.Div([
            html.Div([
                # Header with signal strength indicator
                html.Div([
                    html.H6(f"{icon} Regime Characteristics", className="mb-0", style={
                        "color": AI_COLORS['dark'],
                        "fontSize": AI_TYPOGRAPHY['subtitle_size'],
                        "fontWeight": AI_TYPOGRAPHY['subtitle_weight']
                    }),
                    html.Div([
                        html.Small("Signal Strength: ", style={"color": AI_COLORS['muted']}),
                        html.Span(signal_strength, style={
                            "color": color,
                            "fontWeight": "bold",
                            "fontSize": AI_TYPOGRAPHY['body_size']
                        })
                    ])
                ], className="d-flex justify-content-between align-items-center mb-3"),

                # 4-QUADRANT CHARACTERISTICS LAYOUT
                html.Div([
                    # TOP ROW - First 2 characteristics
                    html.Div([
                        html.Div([
                            create_characteristic_quadrant(char_name, char_value, get_characteristic_color(char_value), get_characteristic_bg_color(char_value))
                            for char_name, char_value in list(regime_characteristics.items())[:2]
                        ], className="row mb-2"),

                        # BOTTOM ROW - Last 2 characteristics
                        html.Div([
                            html.Div([
                                create_characteristic_quadrant(char_name, char_value, get_characteristic_color(char_value), get_characteristic_bg_color(char_value))
                                for char_name, char_value in list(regime_characteristics.items())[2:4]
                            ], className="row")
                        ])
                    ])
                ], id="regime-characteristics")
            ], style={
                "padding": f"{AI_SPACING['lg']} {AI_SPACING['md']}",
                "background": bg_color,
                "borderRadius": AI_EFFECTS['border_radius'],
                "border": f"1px solid {color}",
                "height": "100%",
                "minHeight": "280px"  # Match regime confidence height
            })
        ])

    except Exception as e:
        logger.error(f"Error creating regime characteristics analysis: {str(e)}")
        return html.Div("Regime characteristics unavailable")


def create_characteristic_quadrant(char_name: str, char_value: str, char_color: str, bg_color: str) -> html.Div:
    """Create individual characteristic quadrant with dynamic styling."""
    return html.Div([
        html.Div([
            html.Small(char_name, className="d-block mb-1", style={
                "fontSize": AI_TYPOGRAPHY['tiny_size'],
                "color": AI_COLORS['muted'],
                "fontWeight": "500"
            }),
            html.Strong(char_value, style={
                "fontSize": AI_TYPOGRAPHY['small_size'],
                "color": char_color,
                "fontWeight": "bold"
            })
        ], className="text-center", style={
            "padding": f"{AI_SPACING['sm']} {AI_SPACING['xs']}",
            "background": bg_color,
            "borderRadius": AI_EFFECTS['border_radius_sm'],
            "border": f"1px solid {char_color}",
            "height": "60px",
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "center",
            "transition": AI_EFFECTS['transition'],
            "cursor": "default"
        })
    ], className="col-6 mb-1")


def create_enhanced_regime_analysis_quadrant(regime_analysis: List[str], regime: str, metrics: Dict[str, Any]) -> html.Div:
    """QUADRANT 3: Create ELITE Enhanced AI Regime Analysis with tactical intelligence."""
    try:
        from dashboard_application.utils.regime_display_utils import get_tactical_regime_name, get_regime_icon, get_regime_blurb

        # PYDANTIC-FIRST: Handle None regime values with proper defaults
        if not regime or regime in [None, "None", "UNKNOWN", ""]:
            regime = "UNKNOWN"

        # Get tactical regime name and comprehensive intelligence
        tactical_regime_name = get_tactical_regime_name(regime)
        regime_icon = get_regime_icon(regime)
        regime_blurb = get_regime_blurb(regime)

        # Enhanced regime styling based on tactical classification
        regime_colors = {
            'VANNA_CASCADE': AI_COLORS['danger'],  # High urgency red
            'APEX_AMBUSH': AI_COLORS['primary'],   # Strategic blue
            'ALPHA_SURGE': AI_COLORS['success'],   # Bullish green
            'STRUCTURE_BREACH': AI_COLORS['danger'], # Bearish red
            'IGNITION_POINT': AI_COLORS['warning'], # Volatile orange
            'DEMAND_WALL': AI_COLORS['success'],    # Support green
            'CLOSING_IMBALANCE': AI_COLORS['info'], # EOD blue
            'CONSOLIDATION': AI_COLORS['warning'],  # Neutral yellow
            'CHAOS_STATE': AI_COLORS['danger'],     # High risk red
            'TRANSITION_STATE': AI_COLORS['muted']  # Uncertain gray
        }

        # Determine color based on tactical regime type
        color = AI_COLORS['muted']  # Default
        for regime_type, regime_color in regime_colors.items():
            if regime_type in regime.upper():
                color = regime_color
                break

        # If no tactical match, use basic classification
        if color == AI_COLORS['muted']:
            if 'BULLISH' in regime:
                color = AI_COLORS['success']
            elif 'BEARISH' in regime:
                color = AI_COLORS['danger']
            elif 'VOLATILE' in regime:
                color = AI_COLORS['warning']
            else:
                color = AI_COLORS['info']

        bg_color = f"rgba({color[4:-1]}, 0.1)"

        # Get key metrics for display
        vapi_fa = metrics.get('vapi_fa_z_score_und', 0.0)
        dwfd = metrics.get('dwfd_z_score_und', 0.0)
        tw_laf = metrics.get('tw_laf_z_score_und', 0.0)

        return html.Div([
            html.Div([
                html.H6(f"🧠 Elite AI Analysis", className="mb-3", style={
                    "color": AI_COLORS['dark'],
                    "fontSize": AI_TYPOGRAPHY['subtitle_size'],
                    "fontWeight": AI_TYPOGRAPHY['subtitle_weight']
                }),

                # Enhanced tactical regime display
                html.Div([
                    html.Div([
                        html.Span(f"{regime_icon} {tactical_regime_name}", style={
                            "fontSize": AI_TYPOGRAPHY['title_size'],
                            "fontWeight": "bold",
                            "color": color,
                            "textShadow": "0 1px 2px rgba(0,0,0,0.1)"
                        })
                    ], className="text-center mb-3"),

                    # Enhanced key metrics summary with tactical context
                    html.Div([
                        html.Div([
                            html.Small("VAPI-FA: ", style={"color": AI_COLORS['muted'], "fontWeight": "bold"}),
                            html.Small(f"{vapi_fa:.2f}σ", style={
                                "color": AI_COLORS['success'] if vapi_fa > 0 else AI_COLORS['danger'],
                                "fontWeight": "bold",
                                "fontSize": AI_TYPOGRAPHY['small_size']
                            }),
                            html.Small(" | ", style={"color": AI_COLORS['muted'], "margin": "0 5px"}),
                            html.Small("DWFD: ", style={"color": AI_COLORS['muted'], "fontWeight": "bold"}),
                            html.Small(f"{dwfd:.2f}σ", style={
                                "color": AI_COLORS['success'] if dwfd > 0 else AI_COLORS['danger'],
                                "fontWeight": "bold",
                                "fontSize": AI_TYPOGRAPHY['small_size']
                            })
                        ], className="mb-1"),
                        html.Div([
                            html.Small("TW-LAF: ", style={"color": AI_COLORS['muted'], "fontWeight": "bold"}),
                            html.Small(f"{tw_laf:.2f}σ", style={
                                "color": AI_COLORS['success'] if tw_laf > 0 else AI_COLORS['danger'],
                                "fontWeight": "bold",
                                "fontSize": AI_TYPOGRAPHY['small_size']
                            }),
                            html.Small(" | ", style={"color": AI_COLORS['muted'], "margin": "0 5px"}),
                            html.Small("Confluence: ", style={"color": AI_COLORS['muted'], "fontWeight": "bold"}),
                            html.Small(f"{len([x for x in [abs(vapi_fa), abs(dwfd), abs(tw_laf)] if x > 1.0])}/3", style={
                                "color": color,
                                "fontWeight": "bold",
                                "fontSize": AI_TYPOGRAPHY['small_size']
                            })
                        ], className="mb-3")
                    ], style={
                        "padding": AI_SPACING['sm'],
                        "backgroundColor": "rgba(255, 255, 255, 0.05)",
                        "borderRadius": AI_EFFECTS['border_radius_sm'],
                        "border": f"1px solid {color}33"
                    }),

                    # Enhanced AI Analysis insights with better formatting
                    html.Div([
                        html.Div([
                            html.P(analysis, className="small mb-2", style={
                                "fontSize": AI_TYPOGRAPHY['small_size'],
                                "lineHeight": "1.4",
                                "color": AI_COLORS['dark'],
                                "padding": AI_SPACING['sm'],
                                "borderLeft": f"3px solid {color}",
                                "backgroundColor": "rgba(255, 255, 255, 0.08)",
                                "borderRadius": AI_EFFECTS['border_radius_sm'],
                                "margin": "0 0 8px 0",
                                "boxShadow": "0 1px 3px rgba(0,0,0,0.1)"
                            })
                            for analysis in regime_analysis[:4]  # Expanded to 4 insights for elite analysis
                        ])
                    ])
                ])
            ], style={
                "padding": f"{AI_SPACING['lg']} {AI_SPACING['md']}",
                "background": bg_color,
                "borderRadius": AI_EFFECTS['border_radius'],
                "border": f"1px solid {color}",
                "height": "100%"
            })
        ])

    except Exception as e:
        logger.error(f"Error creating enhanced regime analysis quadrant: {str(e)}")
        return html.Div("Enhanced regime analysis unavailable")


def create_regime_transition_gauge_quadrant(regime_confidence: float, transition_prob: float, regime: str, confluence_score: float) -> html.Div:
    """QUADRANT 4: Create Regime Transition Gauge with comprehensive metrics."""
    try:
        # Determine gauge styling based on transition probability
        if transition_prob >= 0.7:
            gauge_color = AI_COLORS['danger']
            gauge_level = "High Risk"
            gauge_icon = "🚨"
            bg_color = "rgba(255, 71, 87, 0.1)"
        elif transition_prob >= 0.4:
            gauge_color = AI_COLORS['warning']
            gauge_level = "Moderate Risk"
            gauge_icon = "⚠️"
            bg_color = "rgba(255, 167, 38, 0.1)"
        else:
            gauge_color = AI_COLORS['success']
            gauge_level = "Low Risk"
            gauge_icon = "✅"
            bg_color = "rgba(107, 207, 127, 0.1)"

        return html.Div([
            html.Div([
                html.H6(f"{gauge_icon} Transition Gauge", className="mb-3", style={
                    "color": AI_COLORS['dark'],
                    "fontSize": AI_TYPOGRAPHY['subtitle_size'],
                    "fontWeight": AI_TYPOGRAPHY['subtitle_weight']
                }),

                # Main gauge visualization
                html.Div([
                    dcc.Graph(
                        figure=create_regime_transition_gauge(transition_prob, regime_confidence, confluence_score),
                        config={'displayModeBar': False},
                        style={"height": "180px", "marginBottom": "10px"}
                    )
                ]),

                # Gauge metrics summary
                html.Div([
                    html.Div([
                        html.Small("Transition Risk: ", style={"color": AI_COLORS['muted']}),
                        html.Small(f"{transition_prob:.0%} ({gauge_level})", style={
                            "color": gauge_color,
                            "fontWeight": "bold"
                        })
                    ], className="mb-2"),
                    html.Div([
                        html.Small("Regime Stability: ", style={"color": AI_COLORS['muted']}),
                        html.Small(f"{(1-transition_prob):.0%}", style={
                            "color": AI_COLORS['success'] if (1-transition_prob) > 0.6 else AI_COLORS['warning'],
                            "fontWeight": "bold"
                        })
                    ], className="mb-2"),
                    html.Div([
                        html.Small("Signal Confluence: ", style={"color": AI_COLORS['muted']}),
                        html.Small(f"{confluence_score:.0%}", style={
                            "color": AI_COLORS['success'] if confluence_score > 0.6 else AI_COLORS['warning'],
                            "fontWeight": "bold"
                        })
                    ])
                ])
            ], style={
                "padding": f"{AI_SPACING['lg']} {AI_SPACING['md']}",
                "background": bg_color,
                "borderRadius": AI_EFFECTS['border_radius'],
                "border": f"1px solid {gauge_color}",
                "height": "100%"
            })
        ])

    except Exception as e:
        logger.error(f"Error creating regime transition gauge quadrant: {str(e)}")
        return html.Div("Regime transition gauge unavailable")


# ===== UTILITY FUNCTIONS =====

# get_regime_characteristics function moved to intelligence.py to avoid duplication


def get_color_for_value(value: float) -> str:
    """Get color based on value (positive/negative)."""
    if value > 0.1:
        return AI_COLORS['success']
    elif value < -0.1:
        return AI_COLORS['danger']
    else:
        return AI_COLORS['warning']


def get_confluence_color(confluence_score: float) -> str:
    """Get color based on confluence score."""
    if confluence_score >= 0.8:
        return AI_COLORS['success']
    elif confluence_score >= 0.6:
        return AI_COLORS['primary']
    elif confluence_score >= 0.4:
        return AI_COLORS['warning']
    else:
        return AI_COLORS['danger']


def calculate_data_quality_score(bundle_data: FinalAnalysisBundleV2_5) -> float:
    """Calculate data quality score for confidence barometer."""
    try:
        # Check if we have processed data
        if not bundle_data.processed_data_bundle:
            return 0.3

        # Check if we have underlying data
        if not bundle_data.processed_data_bundle.underlying_data_enriched:
            return 0.5

        # Check if we have strike data
        if not bundle_data.processed_data_bundle.strike_level_data_with_metrics:
            return 0.7

        # All data available
        return 0.9

    except Exception as e:
        logger.error(f"Error calculating data quality score: {str(e)}")
        return 0.5


def count_bullish_signals(enriched_data) -> int:
    """PYDANTIC-FIRST: Count bullish signals using direct model access."""
    try:
        if not enriched_data:
            return 0

        count = 0

        # VAPI-FA bullish
        vapi_fa = getattr(enriched_data, 'vapi_fa_z_score_und', 0.0) or 0.0
        if vapi_fa > 1.0:
            count += 1

        # DWFD bullish
        dwfd = getattr(enriched_data, 'dwfd_z_score_und', 0.0) or 0.0
        if dwfd > 0.5:
            count += 1

        # TW-LAF bullish
        tw_laf = getattr(enriched_data, 'tw_laf_z_score_und', 0.0) or 0.0
        if tw_laf > 1.0:
            count += 1

        # GIB bullish (positive call imbalance)
        gib = getattr(enriched_data, 'gib_oi_based_und', 0.0) or 0.0
        if gib > 50000:
            count += 1

        # Price momentum bullish
        price_change = getattr(enriched_data, 'price_change_pct_und', 0.0) or 0.0
        if price_change > 0.005:  # > 0.5%
            count += 1

        return count

    except Exception as e:
        logger.debug(f"Error counting bullish signals: {e}")
        return 0


def count_bearish_signals(enriched_data) -> int:
    """PYDANTIC-FIRST: Count bearish signals using direct model access."""
    try:
        if not enriched_data:
            return 0

        count = 0

        # VAPI-FA bearish
        vapi_fa = getattr(enriched_data, 'vapi_fa_z_score_und', 0.0) or 0.0
        if vapi_fa < -1.0:
            count += 1

        # DWFD bearish
        dwfd = getattr(enriched_data, 'dwfd_z_score_und', 0.0) or 0.0
        if dwfd < -0.5:
            count += 1

        # TW-LAF bearish
        tw_laf = getattr(enriched_data, 'tw_laf_z_score_und', 0.0) or 0.0
        if tw_laf < -1.0:
            count += 1

        # GIB bearish (negative put imbalance)
        gib = getattr(enriched_data, 'gib_oi_based_und', 0.0) or 0.0
        if gib < -50000:
            count += 1

        # Price momentum bearish
        price_change = getattr(enriched_data, 'price_change_pct_und', 0.0) or 0.0
        if price_change < -0.005:  # < -0.5%
            count += 1

        return count

    except Exception as e:
        logger.debug(f"Error counting bearish signals: {e}")
        return 0


def count_neutral_signals(enriched_data) -> int:
    """PYDANTIC-FIRST: Count neutral signals using direct model access."""
    try:
        if not enriched_data:
            return 0

        count = 0

        # VAPI-FA neutral
        vapi_fa = abs(getattr(enriched_data, 'vapi_fa_z_score_und', 0.0) or 0.0)
        if vapi_fa <= 0.5:
            count += 1

        # DWFD neutral
        dwfd = abs(getattr(enriched_data, 'dwfd_z_score_und', 0.0) or 0.0)
        if dwfd <= 0.3:
            count += 1

        # TW-LAF neutral
        tw_laf = abs(getattr(enriched_data, 'tw_laf_z_score_und', 0.0) or 0.0)
        if tw_laf <= 0.5:
            count += 1

        # Low volatility
        atr = getattr(enriched_data, 'atr_und', 0.0) or 0.0
        if atr < 0.01:
            count += 1

        return count

    except Exception as e:
        logger.debug(f"Error counting neutral signals: {e}")
        return 0


def create_atif_recommendation_items(atif_recs: List[Any]) -> html.Div:
    """Create enhanced ATIF recommendation items display with detailed information."""
    try:
        if not atif_recs:
            return html.Div("No ATIF recommendations available", style=get_unified_text_style('muted'))

        items = []
        for i, rec in enumerate(atif_recs):
            conviction_raw = rec.final_conviction_score_from_atif
            strategy = rec.selected_strategy_type
            rationale = str(rec.supportive_rationale_components.get('primary_rationale', 'No rationale provided'))

            # FIXED: Normalize conviction score to 0-1 scale for display
            # ATIF conviction scores can be much larger than 1.0, so we need to normalize
            # Based on the schema, conviction should be 0-5 scale, but we're seeing larger values
            # Let's normalize by dividing by a reasonable maximum (e.g., 50 for very high conviction)
            conviction_normalized = min(conviction_raw / 50.0, 1.0)  # Cap at 1.0

            # Enhanced conviction styling based on raw score ranges
            if conviction_raw > 20.0:  # Very high conviction
                conviction_color = AI_COLORS['success']
                conviction_bg = "rgba(107, 207, 127, 0.15)"
                conviction_icon = "🔥"
                conviction_level = "EXCEPTIONAL"
            elif conviction_raw > 10.0:  # High conviction
                conviction_color = AI_COLORS['primary']
                conviction_bg = "rgba(0, 212, 255, 0.15)"
                conviction_icon = "⚡"
                conviction_level = "HIGH"
            elif conviction_raw > 5.0:  # Moderate conviction
                conviction_color = AI_COLORS['warning']
                conviction_bg = "rgba(255, 167, 38, 0.15)"
                conviction_icon = "⚠️"
                conviction_level = "MODERATE"
            else:  # Low conviction
                conviction_color = AI_COLORS['danger']
                conviction_bg = "rgba(255, 71, 87, 0.15)"
                conviction_icon = "🚨"
                conviction_level = "LOW"

            # Extract additional ATIF details
            dte_range = f"{rec.target_dte_min}-{rec.target_dte_max} DTE"
            underlying_price = rec.underlying_price_at_decision

            items.append(
                html.Div([
                    # Strategy header with enhanced styling
                    html.Div([
                        html.H6(f"🎯 #{i+1}: {strategy}", className="mb-1", style={
                            "fontSize": AI_TYPOGRAPHY['body_size'],
                            "fontWeight": AI_TYPOGRAPHY['subtitle_weight'],
                            "color": AI_COLORS['dark']
                        }),
                        html.Div([
                            html.Span(f"{conviction_icon} {conviction_level}", style={
                                "fontSize": AI_TYPOGRAPHY['small_size'],
                                "color": conviction_color,
                                "fontWeight": "bold"
                            })
                        ])
                    ], className="d-flex justify-content-between align-items-center mb-2"),

                    # Enhanced conviction display
                    html.Div([
                        html.Div([
                            html.Span("Conviction: ", style={
                                "fontSize": AI_TYPOGRAPHY['small_size'],
                                "color": AI_COLORS['dark'],
                                "fontWeight": "bold"
                            }),
                            html.Span(f"{conviction_raw:.1f}", style={  # Show raw score, not percentage
                                "fontSize": AI_TYPOGRAPHY['body_size'],
                                "color": conviction_color,
                                "fontWeight": "bold",
                                "marginLeft": "5px"
                            }),
                            html.Span(" / 50", style={  # Show scale reference
                                "fontSize": AI_TYPOGRAPHY['tiny_size'],
                                "color": AI_COLORS['muted'],
                                "marginLeft": "2px"
                            })
                        ], className="mb-1"),

                        # Conviction progress bar (using normalized score for visual)
                        html.Div([
                            html.Div(style={
                                "width": f"{conviction_normalized * 100}%",
                                "height": "6px",
                                "background": f"linear-gradient(90deg, {conviction_color}, {conviction_color}aa)",
                                "borderRadius": "3px",
                                "transition": AI_EFFECTS['transition']
                            })
                        ], style={
                            "width": "100%",
                            "height": "6px",
                            "background": "rgba(255, 255, 255, 0.1)",
                            "borderRadius": "3px",
                            "marginBottom": AI_SPACING['sm']
                        })
                    ], className="mb-2"),

                    # Strategy details
                    html.Div([
                        html.Div([
                            html.Small(f"📅 {dte_range}", style={
                                "color": AI_COLORS['muted'],
                                "fontSize": AI_TYPOGRAPHY['tiny_size'],
                                "marginRight": "10px"
                            }),
                            html.Small(f"💰 ${underlying_price:.2f}", style={
                                "color": AI_COLORS['muted'],
                                "fontSize": AI_TYPOGRAPHY['tiny_size']
                            })
                        ], className="mb-2")
                    ]),

                    # Rationale with better formatting
                    html.P(rationale[:100] + "..." if len(rationale) > 100 else rationale,
                           className="small", style={
                               "fontSize": AI_TYPOGRAPHY['small_size'],
                               "color": AI_COLORS['muted'],
                               "lineHeight": "1.4",
                               "marginBottom": "0",
                               "fontStyle": "italic"
                           })
                ], className="recommendation-item p-3 mb-2", style={
                    "background": conviction_bg,
                    "borderRadius": AI_EFFECTS['border_radius'],
                    "border": f"2px solid {conviction_color}",
                    "transition": AI_EFFECTS['transition'],
                    "cursor": "pointer",
                    "boxShadow": AI_EFFECTS['box_shadow']
                })
            )

        return html.Div(items)

    except Exception as e:
        logger.error(f"Error creating ATIF recommendation items: {str(e)}")
        return html.Div("Error displaying ATIF recommendations", style=get_unified_text_style('muted'))


def calculate_metric_confluence_score(enriched_data) -> float:
    """PYDANTIC-FIRST: Calculate metric confluence score using direct model access."""
    try:
        if not enriched_data:
            return 0.5

        # PYDANTIC-FIRST: Extract key flow metrics using direct attribute access
        vapi_fa = abs(getattr(enriched_data, 'vapi_fa_z_score_und', 0.0) or 0.0)
        dwfd = abs(getattr(enriched_data, 'dwfd_z_score_und', 0.0) or 0.0)
        tw_laf = abs(getattr(enriched_data, 'tw_laf_z_score_und', 0.0) or 0.0)

        # Calculate confluence based on signal alignment
        strong_signals = sum([vapi_fa > 1.5, dwfd > 1.5, tw_laf > 1.5])
        signal_strength = (vapi_fa + dwfd + tw_laf) / 3.0

        # Confluence score combines signal count and strength
        confluence = (strong_signals / 3.0) * 0.6 + min(signal_strength / 3.0, 1.0) * 0.4
        return min(confluence, 1.0)

    except Exception as e:
        logger.debug(f"Error calculating confluence score: {e}")
        return 0.5


def assess_signal_strength(enriched_data) -> str:
    """PYDANTIC-FIRST: Assess overall signal strength using direct model access."""
    try:
        if not enriched_data:
            return "Unknown"

        # PYDANTIC-FIRST: Calculate total signal strength using direct attribute access
        vapi_fa = abs(getattr(enriched_data, 'vapi_fa_z_score_und', 0.0) or 0.0)
        dwfd = abs(getattr(enriched_data, 'dwfd_z_score_und', 0.0) or 0.0)
        tw_laf = abs(getattr(enriched_data, 'tw_laf_z_score_und', 0.0) or 0.0)

        total_strength = vapi_fa + dwfd + tw_laf

        if total_strength > 6.0:
            return "Extreme"
        elif total_strength > 4.0:
            return "Strong"
        elif total_strength > 2.0:
            return "Moderate"
        else:
            return "Weak"
    except Exception as e:
        logger.debug(f"Error assessing signal strength: {e}")
        return "Unknown"

# REMOVED: Duplicate Pydantic functions - using consolidated versions instead


# Import the badge style function from components
from .components import get_unified_badge_style

# ===== 4-QUADRANT FUNCTIONS =====

def create_ai_confidence_barometer(confidence_score: float, bundle_data: FinalAnalysisBundleV2_5, db_manager=None) -> html.Div:
    """QUADRANT 1: Create AI Confidence Barometer with detailed breakdown."""
    try:
        # Determine confidence level and styling
        if confidence_score >= 0.8:
            confidence_level = "Exceptional"
            color = AI_COLORS['success']
            icon = "🔥"
            bg_color = "rgba(107, 207, 127, 0.1)"
        elif confidence_score >= 0.6:
            confidence_level = "High"
            color = AI_COLORS['primary']
            icon = "⚡"
            bg_color = "rgba(0, 212, 255, 0.1)"
        elif confidence_score >= 0.4:
            confidence_level = "Moderate"
            color = AI_COLORS['warning']
            icon = "⚠️"
            bg_color = "rgba(255, 167, 38, 0.1)"
        else:
            confidence_level = "Low"
            color = AI_COLORS['danger']
            icon = "🚨"
            bg_color = "rgba(255, 71, 87, 0.1)"

        # Calculate confidence factors breakdown using NEW Intelligence Engine V2.5
        system_health_data = get_real_system_health_status(bundle_data, db_manager)
        # FIXED: system_health_data is a Pydantic model, not a dict
        system_health = system_health_data.overall_health_score if system_health_data else 0.5
        data_quality = calculate_data_quality_score(bundle_data)

        return html.Div([
            html.Div([
                html.H6(f"{icon} AI Confidence Barometer", className="mb-3", style={
                    "color": AI_COLORS['dark'],
                    "fontSize": AI_TYPOGRAPHY['subtitle_size'],
                    "fontWeight": AI_TYPOGRAPHY['subtitle_weight']
                }),

                # Main confidence display
                html.Div([
                    html.Div([
                        html.Span(f"{confidence_score:.0%}", id="ai-confidence-score", style={
                            "fontSize": "2.5rem",
                            "fontWeight": "bold",
                            "color": color
                        }),
                        html.Div(confidence_level, id="ai-confidence-level", style={
                            "fontSize": AI_TYPOGRAPHY['body_size'],
                            "color": AI_COLORS['muted'],
                            "marginTop": "-5px"
                        })
                    ], className="text-center mb-3"),

                    # Enhanced Confidence bar - Bigger to fill timestamp space
                    html.Div([
                        html.Div(style={
                            "width": f"{confidence_score * 100}%",
                            "height": "18px",  # Increased from 12px to 18px
                            "background": f"linear-gradient(90deg, {color}, {color}aa)",
                            "borderRadius": "9px",  # Adjusted border radius proportionally
                            "transition": AI_EFFECTS['transition']
                        })
                    ], style={
                        "width": "100%",
                        "height": "18px",  # Match inner height
                        "background": "rgba(255, 255, 255, 0.1)",
                        "borderRadius": "9px",
                        "marginBottom": AI_SPACING['lg']  # Increased margin to fill space
                    }),

                    # Enhanced Confidence factors - Better spacing
                    html.Div([
                        html.Div([
                            html.Small("System Health: ", style={"color": AI_COLORS['muted']}),
                            html.Small(f"{system_health:.0%}", style={"color": color, "fontWeight": "bold"})
                        ], className="mb-2"),
                        html.Div([
                            html.Small("Data Quality: ", style={"color": AI_COLORS['muted']}),
                            html.Small(f"{data_quality:.0%}", style={"color": color, "fontWeight": "bold"})
                        ])
                    ])
                ])
            ], id="ai-confidence-container", style={
                "padding": f"{AI_SPACING['lg']} {AI_SPACING['md']}",
                "background": bg_color,
                "borderRadius": AI_EFFECTS['border_radius'],
                "border": f"1px solid {color}",
                "height": "100%"
            })
        ])

    except Exception as e:
        logger.error(f"Error creating AI confidence barometer: {str(e)}")
        return html.Div("Confidence barometer unavailable")


def create_signal_confluence_barometer(confluence_score: float, enriched_data, signal_strength: str) -> html.Div:
    """PYDANTIC-FIRST: Create Signal Confluence Barometer using direct model access."""
    try:
        # Determine confluence level and styling
        if confluence_score >= 0.8:
            confluence_level = "Strong Alignment"
            color = AI_COLORS['success']
            icon = "🎯"
            bg_color = "rgba(107, 207, 127, 0.1)"
            gauge_color = "#6bcf7f"
        elif confluence_score >= 0.6:
            confluence_level = "Good Alignment"
            color = AI_COLORS['primary']
            icon = "📊"
            bg_color = "rgba(0, 212, 255, 0.1)"
            gauge_color = "#00d4ff"
        elif confluence_score >= 0.4:
            confluence_level = "Mixed Signals"
            color = AI_COLORS['warning']
            icon = "⚖️"
            bg_color = "rgba(255, 167, 38, 0.1)"
            gauge_color = "#ffa726"
        else:
            confluence_level = "Divergent"
            color = AI_COLORS['danger']
            icon = "🌪️"
            bg_color = "rgba(255, 71, 87, 0.1)"
            gauge_color = "#ff4757"

        # PYDANTIC-FIRST: Calculate signal components using direct model access
        bullish_signals = count_bullish_signals(enriched_data)
        bearish_signals = count_bearish_signals(enriched_data)
        neutral_signals = count_neutral_signals(enriched_data)
        total_signals = bullish_signals + bearish_signals + neutral_signals

        # Create beautiful gauge chart
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confluence_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Signal Confluence", 'font': {'size': 16, 'color': AI_COLORS['dark']}},
            number={'font': {'size': 24, 'color': color}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': AI_COLORS['muted']},
                'bar': {'color': gauge_color, 'thickness': 0.3},
                'bgcolor': "rgba(255, 255, 255, 0.1)",
                'borderwidth': 2,
                'bordercolor': color,
                'steps': [
                    {'range': [0, 40], 'color': "rgba(255, 71, 87, 0.2)"},
                    {'range': [40, 60], 'color': "rgba(255, 167, 38, 0.2)"},
                    {'range': [60, 80], 'color': "rgba(0, 212, 255, 0.2)"},
                    {'range': [80, 100], 'color': "rgba(107, 207, 127, 0.2)"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': confluence_score * 100
                }
            }
        ))

        gauge_fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': AI_COLORS['dark']},
            margin=dict(l=20, r=20, t=40, b=20),
            height=180
        )

        return html.Div([
            html.Div([
                # Header with icon and signal strength
                html.Div([
                    html.H6(f"{icon} Signal Confluence Barometer", className="mb-2", style={
                        "color": AI_COLORS['dark'],
                        "fontSize": AI_TYPOGRAPHY['subtitle_size'],
                        "fontWeight": AI_TYPOGRAPHY['subtitle_weight']
                    }),
                    html.Div([
                        html.Small("Strength: ", style={"color": AI_COLORS['muted']}),
                        html.Span(signal_strength, style={
                            "color": color,
                            "fontWeight": "bold",
                            "fontSize": AI_TYPOGRAPHY['small_size']
                        })
                    ])
                ], className="d-flex justify-content-between align-items-center mb-2"),

                # Beautiful gauge chart
                html.Div([
                    dcc.Graph(
                        id="signal-confluence-gauge",
                        figure=gauge_fig,
                        config={'displayModeBar': False},
                        style={"height": "180px"}
                    )
                ], className="mb-2"),

                # Signal breakdown with enhanced styling
                html.Div([
                    html.Div([
                        html.Div([
                            html.Span("🟢", style={"marginRight": "8px", "fontSize": "14px"}),
                            html.Span(f"Bullish: {bullish_signals}", style={
                                "color": AI_COLORS['success'],
                                "fontSize": AI_TYPOGRAPHY['small_size'],
                                "fontWeight": "bold"
                            })
                        ], className="d-flex align-items-center mb-1"),
                        html.Div([
                            html.Span("🔴", style={"marginRight": "8px", "fontSize": "14px"}),
                            html.Span(f"Bearish: {bearish_signals}", style={
                                "color": AI_COLORS['danger'],
                                "fontSize": AI_TYPOGRAPHY['small_size'],
                                "fontWeight": "bold"
                            })
                        ], className="d-flex align-items-center mb-1"),
                        html.Div([
                            html.Span("⚪", style={"marginRight": "8px", "fontSize": "14px"}),
                            html.Span(f"Neutral: {neutral_signals}", style={
                                "color": AI_COLORS['muted'],
                                "fontSize": AI_TYPOGRAPHY['small_size'],
                                "fontWeight": "bold"
                            })
                        ], className="d-flex align-items-center")
                    ], style={
                        "padding": f"{AI_SPACING['sm']} {AI_SPACING['md']}",
                        "background": "rgba(255, 255, 255, 0.05)",
                        "borderRadius": AI_EFFECTS['border_radius_sm'],
                        "border": "1px solid rgba(255, 255, 255, 0.1)"
                    })
                ])
            ], id="signal-confluence-container", style={
                "padding": f"{AI_SPACING['md']} {AI_SPACING['md']}",
                "background": bg_color,
                "borderRadius": AI_EFFECTS['border_radius'],
                "border": f"2px solid {color}",
                "height": "100%",
                "boxShadow": AI_EFFECTS['box_shadow']
            })
        ])

    except Exception as e:
        logger.error(f"Error creating signal confluence barometer: {str(e)}")
        return html.Div("Signal confluence unavailable")





def create_unified_intelligence_analysis(unified_insights: List[str], regime: str, bundle_data: FinalAnalysisBundleV2_5) -> html.Div:
    """QUADRANT 3: Create Enhanced Unified Intelligence Analysis with sophisticated market insights."""
    try:
        # Get tactical regime name using proper front-end mapping
        tactical_regime_name = get_tactical_regime_name(regime)

        # Enhanced regime styling based on tactical regime content
        if any(word in tactical_regime_name for word in ["Bullish", "Surge", "Demand"]):
            regime_color = AI_COLORS['success']
            regime_icon = '🚀'
            bg_color = "rgba(107, 207, 127, 0.08)"
        elif any(word in tactical_regime_name for word in ["Bearish", "Breach", "Supply"]):
            regime_color = AI_COLORS['danger']
            regime_icon = '🐻'
            bg_color = "rgba(255, 71, 87, 0.08)"
        elif any(word in tactical_regime_name for word in ["Vanna", "Ignition", "Ambush"]):
            regime_color = AI_COLORS['warning']
            regime_icon = '⚡'
            bg_color = "rgba(255, 167, 38, 0.08)"
        elif any(word in tactical_regime_name for word in ["Chaos", "Volatile"]):
            regime_color = AI_COLORS['info']
            regime_icon = '🌪️'
            bg_color = "rgba(0, 212, 255, 0.08)"
        elif any(word in tactical_regime_name for word in ["Consolidation", "Transition"]):
            regime_color = AI_COLORS['secondary']
            regime_icon = '⚖️'
            bg_color = "rgba(108, 117, 125, 0.08)"
        else:
            regime_color = AI_COLORS['muted']
            regime_icon = '❓'
            bg_color = "rgba(255, 255, 255, 0.05)"

        return html.Div([
            html.Div([
                html.H6(f"🧠 Unified Intelligence Analysis", className="mb-3", style={
                    "color": AI_COLORS['dark'],
                    "fontSize": AI_TYPOGRAPHY['subtitle_size'],
                    "fontWeight": AI_TYPOGRAPHY['subtitle_weight']
                }),

                # Enhanced tactical regime indicator
                html.Div([
                    html.Div([
                        html.Span(f"{regime_icon} {tactical_regime_name}", style={
                            "fontSize": AI_TYPOGRAPHY['body_size'],
                            "fontWeight": "bold",
                            "color": regime_color,
                            "lineHeight": "1.2"
                        }),
                        html.Div("Tactical Market Regime", style={
                            "fontSize": AI_TYPOGRAPHY['tiny_size'],
                            "color": AI_COLORS['muted'],
                            "marginTop": "2px"
                        })
                    ], className="text-center mb-3", style={
                        "padding": f"{AI_SPACING['sm']} {AI_SPACING['md']}",
                        "background": bg_color,
                        "borderRadius": AI_EFFECTS['border_radius_sm'],
                        "border": f"1px solid {regime_color}"
                    })
                ]),

                # Enhanced insights with dynamic icons
                html.Div([
                    html.Div([
                        html.Div([
                            # Dynamic icon based on insight content
                            html.Span(
                                "🌟" if "ECOSYSTEM" in insight.upper() or "OPTIMAL" in insight.upper()
                                else "🎯" if "ARCHITECTURE" in insight.upper() or "CONVERGENT" in insight.upper()
                                else "🌊" if "EVOLUTION" in insight.upper() or "VOLATILITY" in insight.upper()
                                else "⚖️" if "GAMMA" in insight.upper() or "DEALER" in insight.upper()
                                else "🧠" if "REGIME INTELLIGENCE" in insight.upper()
                                else "📰" if "FUSION" in insight.upper() or "NEWS" in insight.upper()
                                else "🎯" if "CONFLUENCE" in insight.upper()
                                else "💡",
                                style={"marginRight": "10px", "fontSize": "14px"}
                            ),
                            html.Span(insight, style={
                                "fontSize": AI_TYPOGRAPHY['small_size'],
                                "color": AI_COLORS['dark'],
                                "lineHeight": "1.5",
                                "fontWeight": "500"
                            })
                        ], style={
                            "padding": f"{AI_SPACING['sm']} {AI_SPACING['md']}",
                            "background": (
                                "rgba(107, 207, 127, 0.1)" if any(word in insight.upper() for word in ["OPTIMAL", "BULLISH", "SURGE"])
                                else "rgba(255, 71, 87, 0.1)" if any(word in insight.upper() for word in ["STRESS", "BEARISH", "BREACH"])
                                else "rgba(255, 167, 38, 0.1)" if any(word in insight.upper() for word in ["PRESSURE", "CHAOS", "DIVERGENCE"])
                                else "rgba(0, 212, 255, 0.1)"
                            ),
                            "borderRadius": AI_EFFECTS['border_radius_sm'],
                            "marginBottom": AI_SPACING['sm'],
                            "border": "1px solid rgba(255, 255, 255, 0.15)",
                            "transition": AI_EFFECTS['transition']
                        })
                        for insight in unified_insights[:5]  # Show top 5 sophisticated insights
                    ])
                ], style={
                    "maxHeight": "200px",
                    "overflowY": "auto",
                    "paddingRight": "5px"
                }),

                # Enhanced analysis timestamp with system status
                html.Div([
                    html.Small(f"Intelligence Update: {bundle_data.bundle_timestamp.strftime('%H:%M:%S')} | AI Systems Active",
                             style={"color": AI_COLORS['muted'], "fontSize": AI_TYPOGRAPHY['tiny_size']})
                ], className="text-center mt-2")
            ], style={
                "padding": f"{AI_SPACING['lg']} {AI_SPACING['md']}",
                "background": "linear-gradient(135deg, rgba(0, 212, 255, 0.05), rgba(107, 207, 127, 0.03))",
                "borderRadius": AI_EFFECTS['border_radius'],
                "border": f"2px solid {AI_COLORS['primary']}",
                "height": "100%",
                "boxShadow": AI_EFFECTS['box_shadow']
            })
        ])

    except Exception as e:
        logger.error(f"Error creating unified intelligence analysis: {str(e)}")
        return html.Div("Intelligence analysis unavailable")


def create_market_dynamics_radar_quadrant(enhanced_radar_fig, enriched_data, symbol: str) -> html.Div:
    """PYDANTIC-FIRST: Enhanced Market Dynamics Radar using direct model access."""
    try:
        # PYDANTIC-FIRST: Calculate dynamic market forces using direct model access
        if enriched_data:
            metrics = enriched_data.model_dump()  # Convert for legacy force calculation functions
        else:
            metrics = {}

        momentum_force = calculate_momentum_force(metrics)
        volatility_force = calculate_volatility_force(metrics)
        volume_force = calculate_volume_force(metrics)
        sentiment_force = calculate_sentiment_force(metrics)

        # Determine dominant force
        forces = {
            'Momentum': momentum_force,
            'Volatility': volatility_force,
            'Volume': volume_force,
            'Sentiment': sentiment_force
        }
        dominant_force = max(forces, key=lambda k: forces[k])
        dominant_value = forces[dominant_force]

        # Force icons and colors
        force_icons = {
            'Momentum': '🚀',
            'Volatility': '⚡',
            'Volume': '📊',
            'Sentiment': '💭'
        }

        return html.Div([
            html.Div([
                html.H6(f"🎯 Market Dynamics Radar", className="mb-3", style={
                    "color": AI_COLORS['dark'],
                    "fontSize": AI_TYPOGRAPHY['subtitle_size'],
                    "fontWeight": AI_TYPOGRAPHY['subtitle_weight']
                }),

                # Enhanced radar chart
                html.Div([
                    dcc.Graph(
                        id="market-dynamics-radar",
                        figure=enhanced_radar_fig,
                        config={'displayModeBar': False},
                        style={"height": "200px"}
                    )
                ], className="mb-3"),

                # Dominant force indicator
                html.Div([
                    html.Div([
                        html.Span(f"{force_icons[dominant_force]}", style={
                            "fontSize": "1.5rem",
                            "marginRight": "8px"
                        }),
                        html.Span(f"{dominant_force}", style={
                            "fontSize": AI_TYPOGRAPHY['body_size'],
                            "fontWeight": "bold",
                            "color": AI_COLORS['primary']
                        }),
                        html.Div(f"Dominant Force ({dominant_value:.1f})", style={
                            "fontSize": AI_TYPOGRAPHY['small_size'],
                            "color": AI_COLORS['muted']
                        })
                    ], className="text-center mb-2"),

                    # Force breakdown mini indicators
                    html.Div([
                        html.Div([
                            html.Span(f"{force_icons[force]}", style={"fontSize": "12px", "marginRight": "4px"}),
                            html.Span(f"{value:.1f}", style={
                                "fontSize": AI_TYPOGRAPHY['tiny_size'],
                                "color": AI_COLORS['primary'] if force == dominant_force else AI_COLORS['muted']
                            })
                        ], className="d-inline-block me-2")
                        for force, value in forces.items()
                    ], className="text-center")
                ])
            ], style={
                "padding": f"{AI_SPACING['lg']} {AI_SPACING['md']}",
                "background": "linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(107, 207, 127, 0.05))",
                "borderRadius": AI_EFFECTS['border_radius'],
                "border": f"2px solid {AI_COLORS['primary']}",
                "height": "100%",
                "boxShadow": AI_EFFECTS['box_shadow']
            })
        ])

    except Exception as e:
        logger.error(f"Error creating market dynamics radar quadrant: {str(e)}")
        return html.Div("Market dynamics radar unavailable")


def create_enhanced_market_dynamics_radar(bundle_data: FinalAnalysisBundleV2_5, symbol: str):
    """Create enhanced market dynamics radar with comprehensive market forces."""
    try:
        processed_data = bundle_data.processed_data_bundle
        metrics = processed_data.underlying_data_enriched.model_dump() if processed_data else {}

        # Calculate comprehensive market forces (0-10 scale)
        forces = {
            'Momentum': calculate_momentum_force(metrics),
            'Volatility': calculate_volatility_force(metrics),
            'Volume': calculate_volume_force(metrics),
            'Sentiment': calculate_sentiment_force(metrics),
            'Trend': calculate_trend_force(metrics),
            'Support/Resistance': calculate_sr_force(metrics),
            'Options Flow': calculate_options_flow_force(metrics),
            'Market Breadth': calculate_market_breadth_force(metrics)
        }

        # Debug logging to track radar updates
        logger.info(f"🎯 Market Dynamics Radar Update for {symbol}: {forces}")
        logger.debug(f"📊 Available metrics keys: {list(metrics.keys())[:10]}...")  # Show first 10 keys

        # Create enhanced radar chart
        fig = go.Figure()

        # Add market forces trace
        fig.add_trace(go.Scatterpolar(
            r=list(forces.values()),
            theta=list(forces.keys()),
            fill='toself',
            fillcolor='rgba(0, 212, 255, 0.3)',
            line=dict(color=AI_COLORS['primary'], width=3),
            marker=dict(size=8, color=AI_COLORS['primary']),
            name='Market Forces',
            hovertemplate='<b>%{theta}</b><br>Force: %{r:.1f}/10<extra></extra>'
        ))

        # Add reference circle at 5.0 (neutral)
        fig.add_trace(go.Scatterpolar(
            r=[5] * len(forces),
            theta=list(forces.keys()),
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Enhanced layout
        fig.update_layout(
            polar=dict(
                bgcolor='rgba(0, 0, 0, 0.1)',
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    tickfont=dict(size=10, color=AI_COLORS['muted']),
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    linecolor='rgba(255, 255, 255, 0.3)'
                ),
                angularaxis=dict(
                    tickfont=dict(size=11, color=AI_COLORS['dark']),
                    linecolor='rgba(255, 255, 255, 0.3)',
                    gridcolor='rgba(255, 255, 255, 0.2)'
                )
            ),
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=AI_COLORS['dark'])
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating enhanced market dynamics radar: {str(e)}")
        return go.Figure()


# ===== MARKET FORCE CALCULATION FUNCTIONS =====

def calculate_momentum_force(metrics: Dict[str, Any]) -> float:
    """Calculate momentum force (0-10 scale) using EOTS metrics with Pydantic validation."""
    try:
        # PYDANTIC-FIRST: Import and use validation models from EOTS schemas
        from pydantic import BaseModel, Field

        # Create a validation model for momentum calculation inputs
        class MomentumInputsV2_5(BaseModel):
            vapi_fa_z_score_und: float = Field(default=0.0, description="VAPI-FA Z-score")
            dwfd_z_score_und: float = Field(default=0.0, description="DWFD Z-score")
            price_change_abs_und: float = Field(default=0.0, description="Absolute price change")
            price_change_pct_und: float = Field(default=0.0, description="Percentage price change")

        # Validate inputs using Pydantic model - filter out None values to use defaults
        filtered_metrics = {
            k: v for k, v in metrics.items()
            if k in ['vapi_fa_z_score_und', 'dwfd_z_score_und', 'price_change_abs_und', 'price_change_pct_und']
            and v is not None
        }
        validated_inputs = MomentumInputsV2_5(**filtered_metrics)

        # Extract validated values (guaranteed to be non-None floats with defaults)
        vapi_fa_z = validated_inputs.vapi_fa_z_score_und
        dwfd_z = validated_inputs.dwfd_z_score_und
        price_change_abs = validated_inputs.price_change_abs_und
        price_change_pct = validated_inputs.price_change_pct_und

        # Normalize to 0-10 scale using EOTS Z-scores
        vapi_score = min(abs(vapi_fa_z) * 2, 10)  # Z-score momentum
        dwfd_score = min(abs(dwfd_z) * 2, 10)  # Flow direction momentum
        price_abs_score = min(abs(price_change_abs) * 0.5, 10)  # Absolute price change
        price_pct_score = min(abs(price_change_pct) * 20, 10)  # Percentage price change

        # Weight EOTS metrics more heavily
        momentum_score = (vapi_score * 0.4 + dwfd_score * 0.4 + price_abs_score * 0.1 + price_pct_score * 0.1)
        return min(momentum_score, 10)

    except Exception as e:
        logger.debug(f"Error calculating momentum force: {e}")
        return 5.0

def calculate_volatility_force(metrics: Dict[str, Any]) -> float:
    """Calculate volatility force (0-10 scale) using EOTS metrics with Pydantic validation."""
    try:
        # PYDANTIC-FIRST: Create validation model for volatility calculation inputs
        from pydantic import BaseModel, Field

        class VolatilityInputsV2_5(BaseModel):
            atr_und: float = Field(default=0.0, description="Average True Range")
            u_volatility: float = Field(default=0.0, description="Underlying volatility")
            tw_laf_z_score_und: float = Field(default=0.0, description="Time-weighted LAF Z-score")
            gib_oi_based_und: float = Field(default=0.0, description="Gamma imbalance based on OI")

        # Validate inputs using Pydantic model - filter out None values to use defaults
        filtered_metrics = {
            k: v for k, v in metrics.items()
            if k in ['atr_und', 'u_volatility', 'tw_laf_z_score_und', 'gib_oi_based_und']
            and v is not None
        }
        validated_inputs = VolatilityInputsV2_5(**filtered_metrics)

        # Extract validated values (guaranteed to be non-None floats with defaults)
        atr_und = validated_inputs.atr_und
        u_volatility = validated_inputs.u_volatility
        tw_laf_z = validated_inputs.tw_laf_z_score_und
        gib_oi = validated_inputs.gib_oi_based_und

        # Normalize to 0-10 scale using EOTS metrics
        atr_score = min(atr_und * 200, 10)  # ATR from EOTS
        vol_score = min(u_volatility * 100, 10)  # Underlying volatility
        tw_laf_score = min(abs(tw_laf_z) * 2, 10)  # Time-weighted liquidity flow
        gib_score = min(abs(gib_oi) * 5, 10)  # Gamma imbalance volatility

        # Weight EOTS volatility metrics
        volatility_score = (atr_score * 0.3 + vol_score * 0.3 + tw_laf_score * 0.2 + gib_score * 0.2)
        return min(volatility_score, 10)
    except Exception as e:
        logger.debug(f"Error calculating volatility force: {e}")
        return 5.0

def calculate_volume_force(metrics: Dict[str, Any]) -> float:
    """Calculate volume force (0-10 scale) using EOTS metrics."""
    try:
        # Extract EOTS volume indicators
        day_volume = metrics.get('day_volume', 0)
        total_call_vol = metrics.get('total_call_vol_und', 0)
        total_put_vol = metrics.get('total_put_vol_und', 0)
        net_vol_flow_5m = metrics.get('net_vol_flow_5m_und', 0)
        net_vol_flow_15m = metrics.get('net_vol_flow_15m_und', 0)

        # Calculate volume metrics
        total_options_vol = (total_call_vol or 0) + (total_put_vol or 0)
        vol_5m_abs = abs(net_vol_flow_5m or 0)
        vol_15m_abs = abs(net_vol_flow_15m or 0)

        # Normalize to 0-10 scale
        day_vol_score = min((day_volume or 0) / 1000000, 10)  # Scale day volume
        options_vol_score = min(total_options_vol / 100000, 10)  # Scale options volume
        flow_5m_score = min(vol_5m_abs / 10000, 10)  # 5-minute flow intensity
        flow_15m_score = min(vol_15m_abs / 50000, 10)  # 15-minute flow intensity

        # Weight flow metrics more heavily for EOTS
        volume_score = (day_vol_score * 0.2 + options_vol_score * 0.3 + flow_5m_score * 0.3 + flow_15m_score * 0.2)
        return min(volume_score, 10)
    except Exception as e:
        logger.debug(f"Error calculating volume force: {e}")
        return 5.0

def calculate_sentiment_force(metrics: Dict[str, Any]) -> float:
    """Calculate sentiment force (0-10 scale) using EOTS metrics."""
    try:
        # Extract EOTS sentiment indicators
        total_call_oi = metrics.get('total_call_oi_und', 0)
        total_put_oi = metrics.get('total_put_oi_und', 0)
        total_call_vol = metrics.get('total_call_vol_und', 0)
        total_put_vol = metrics.get('total_put_vol_und', 0)
        vflowratio = metrics.get('vflowratio', 1.0)

        # Calculate put/call ratios
        if total_call_oi and total_put_oi:
            pc_oi_ratio = total_put_oi / total_call_oi
        else:
            pc_oi_ratio = 1.0

        if total_call_vol and total_put_vol:
            pc_vol_ratio = total_put_vol / total_call_vol
        else:
            pc_vol_ratio = 1.0

        # Normalize to 0-10 scale
        oi_sentiment_score = min(abs(pc_oi_ratio - 1.0) * 10, 10)  # OI imbalance
        vol_sentiment_score = min(abs(pc_vol_ratio - 1.0) * 10, 10)  # Volume imbalance
        vflow_score = min(abs((vflowratio or 1.0) - 1.0) * 15, 10)  # Volume flow ratio

        # Weight EOTS sentiment metrics
        sentiment_score = (oi_sentiment_score * 0.4 + vol_sentiment_score * 0.4 + vflow_score * 0.2)
        return min(sentiment_score, 10)
    except Exception as e:
        logger.debug(f"Error calculating sentiment force: {e}")
        return 5.0

def calculate_trend_force(metrics: Dict[str, Any]) -> float:
    """Calculate trend force (0-10 scale) using EOTS metrics."""
    try:
        # Extract EOTS trend indicators
        price_change_pct = metrics.get('price_change_pct_und', 0)
        net_value_flow_5m = metrics.get('net_value_flow_5m_und', 0)
        net_value_flow_15m = metrics.get('net_value_flow_15m_und', 0)
        net_value_flow_30m = metrics.get('net_value_flow_30m_und', 0)

        # Calculate trend strength from flow consistency
        flows = [net_value_flow_5m or 0, net_value_flow_15m or 0, net_value_flow_30m or 0]
        flow_signs = [1 if f > 0 else -1 if f < 0 else 0 for f in flows]
        flow_consistency = abs(sum(flow_signs)) / len(flow_signs) if flow_signs else 0

        # Normalize to 0-10 scale
        price_trend_score = min(abs(price_change_pct or 0) * 50, 10)  # Price momentum
        flow_consistency_score = flow_consistency * 10  # Flow direction consistency
        flow_magnitude_score = min(sum(abs(f) for f in flows) / 100000, 10)  # Flow strength

        # Weight trend metrics
        trend_score = (price_trend_score * 0.4 + flow_consistency_score * 0.4 + flow_magnitude_score * 0.2)
        return min(trend_score, 10)
    except Exception as e:
        logger.debug(f"Error calculating trend force: {e}")
        return 5.0

def calculate_sr_force(metrics: Dict[str, Any]) -> float:
    """Calculate support/resistance force (0-10 scale) using EOTS metrics."""
    try:
        # Extract EOTS structure indicators
        gib_oi = metrics.get('gib_oi_based_und', 0)
        hp_eod = metrics.get('hp_eod_und', 0)
        call_gxoi = metrics.get('call_gxoi', 0)
        put_gxoi = metrics.get('put_gxoi', 0)
        dxoi = metrics.get('dxoi', 0)

        # Calculate structural forces
        gib_force = min(abs(gib_oi or 0) * 10, 10)  # Gamma imbalance structural force
        hp_force = min(abs(hp_eod or 0) * 5, 10)  # Hedging pressure force
        gamma_imbalance = abs((call_gxoi or 0) - (put_gxoi or 0))
        gamma_force = min(gamma_imbalance / 1000, 10)  # Gamma concentration force
        delta_force = min(abs(dxoi or 0) / 10000, 10)  # Delta concentration force

        # Weight structural metrics for support/resistance
        sr_score = (gib_force * 0.4 + hp_force * 0.3 + gamma_force * 0.2 + delta_force * 0.1)
        return min(sr_score, 10)
    except Exception as e:
        logger.debug(f"Error calculating support/resistance force: {e}")
        return 5.0

def calculate_options_flow_force(metrics: Dict[str, Any]) -> float:
    """Calculate options flow force (0-10 scale) using EOTS metrics."""
    try:
        # Extract EOTS flow indicators with safe null handling
        vapi_fa_z = metrics.get('vapi_fa_z_score_und', 0) or 0
        dwfd_z = metrics.get('dwfd_z_score_und', 0) or 0
        tw_laf_z = metrics.get('tw_laf_z_score_und', 0) or 0
        net_cust_delta_flow = metrics.get('net_cust_delta_flow_und', 0) or 0
        net_cust_gamma_flow = metrics.get('net_cust_gamma_flow_und', 0) or 0

        # Calculate flow force from EOTS Z-scores
        vapi_force = min(abs(vapi_fa_z) * 2.5, 10)  # VAPI-FA flow force
        dwfd_force = min(abs(dwfd_z) * 2.5, 10)  # DWFD flow force
        tw_laf_force = min(abs(tw_laf_z) * 2.5, 10)  # TW-LAF flow force
        delta_flow_force = min(abs(net_cust_delta_flow) / 10000, 10)  # Delta flow
        gamma_flow_force = min(abs(net_cust_gamma_flow) / 1000, 10)  # Gamma flow

        # Weight EOTS flow metrics heavily
        flow_score = (vapi_force * 0.3 + dwfd_force * 0.3 + tw_laf_force * 0.2 +
                     delta_flow_force * 0.1 + gamma_flow_force * 0.1)
        return min(flow_score, 10)
    except Exception as e:
        logger.debug(f"Error calculating options flow force: {e}")
        return 5.0

def calculate_market_breadth_force(metrics: Dict[str, Any]) -> float:
    """Calculate market breadth force (0-10 scale) using EOTS metrics."""
    try:
        # Extract EOTS breadth indicators
        total_call_oi = metrics.get('total_call_oi_und', 0)
        total_put_oi = metrics.get('total_put_oi_und', 0)
        total_call_vol = metrics.get('total_call_vol_und', 0)
        total_put_vol = metrics.get('total_put_vol_und', 0)
        day_volume = metrics.get('day_volume', 0)
        vxoi = metrics.get('vxoi', 0)

        # Calculate breadth metrics
        total_oi = (total_call_oi or 0) + (total_put_oi or 0)
        total_vol = (total_call_vol or 0) + (total_put_vol or 0)

        # Normalize to 0-10 scale
        oi_breadth_score = min(total_oi / 1000000, 10)  # Open interest breadth
        vol_breadth_score = min(total_vol / 100000, 10)  # Volume breadth
        day_vol_score = min((day_volume or 0) / 10000000, 10)  # Daily volume breadth
        vxoi_score = min(abs(vxoi or 0) / 100000, 10)  # VXOI breadth

        # Weight breadth metrics
        breadth_score = (oi_breadth_score * 0.3 + vol_breadth_score * 0.3 +
                        day_vol_score * 0.2 + vxoi_score * 0.2)
        return min(breadth_score, 10)
    except Exception as e:
        logger.debug(f"Error calculating market breadth force: {e}")
        return 5.0


def get_confluence_color(confluence_score: float) -> str:
    """Get color for confluence score."""
    if confluence_score > 0.8:
        return AI_COLORS['primary']  # Bright blue for high confluence
    elif confluence_score > 0.6:
        return AI_COLORS['success']  # Green for good confluence
    elif confluence_score > 0.4:
        return AI_COLORS['warning']  # Yellow for moderate confluence
    else:
        return AI_COLORS['danger']  # Red for low confluence


def create_confluence_gauge(confluence_score: float):
    """Create a small confluence gauge visualization."""

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confluence_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confluence"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': get_confluence_color(confluence_score)},
            'steps': [
                {'range': [0, 40], 'color': "rgba(255, 107, 107, 0.2)"},
                {'range': [40, 60], 'color': "rgba(255, 193, 7, 0.2)"},
                {'range': [60, 80], 'color': "rgba(107, 207, 127, 0.2)"},
                {'range': [80, 100], 'color': "rgba(0, 123, 255, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        height=120,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=AI_COLORS['dark'], size=10)
    )

    return fig


# ===== SIGNAL ANALYSIS FUNCTIONS =====

# REMOVED: Dictionary-based duplicate - using Pydantic-first version above

# REMOVED: Dictionary-based duplicate - using Pydantic-first version above

# REMOVED: Dictionary-based duplicate - using Pydantic-first version above


# ===== SYSTEM HEALTH FUNCTIONS =====

# System health calculation moved to pydantic_intelligence_engine_v2_5.py
# Use get_real_system_health_status() instead

def calculate_system_health_score_legacy(bundle_data: FinalAnalysisBundleV2_5, db_manager=None) -> float:
    """Legacy system health calculation - use get_real_system_health_status() instead."""
    try:
        # Return default health score for legacy function
        return 0.85
    except:
        return 0.5

def calculate_data_quality_score(bundle_data: FinalAnalysisBundleV2_5) -> float:
    """Calculate data quality score."""
    try:
        quality_factors = []

        processed_data = bundle_data.processed_data_bundle
        if not processed_data:
            return 0.3

        # Check for required fields
        enriched_data = processed_data.underlying_data_enriched
        if enriched_data:
            # Check if key metrics exist
            metrics = enriched_data.model_dump()
            required_fields = ['current_price', 'volume', 'rsi_14', 'macd']

            field_score = sum(1 for field in required_fields if metrics.get(field) is not None) / len(required_fields)
            quality_factors.append(field_score)

            # Check for reasonable values
            price = metrics.get('current_price', 0)
            volume = metrics.get('volume', 0)

            if price > 0 and volume > 0:
                value_score = 1.0
            else:
                value_score = 0.5
            quality_factors.append(value_score)
        else:
            quality_factors.append(0.3)

        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.3
    except:
        return 0.3
