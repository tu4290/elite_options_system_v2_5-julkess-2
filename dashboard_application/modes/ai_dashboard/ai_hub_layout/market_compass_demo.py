"""
Standalone Market Compass Demo
=============================

This demo shows the legendary Market Compass visualization using sample EOTS data.
It creates a 12-dimensional radar chart with multi-timeframe layers.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime
import math

# AI Colors (matching your system)
AI_COLORS = {
    'primary': '#42A5F5',
    'success': '#6BCF7F', 
    'warning': '#FFA726',
    'danger': '#FF4757',
    'info': '#42A5F5',
    'muted': '#6C757D'
}

def create_legendary_market_compass_demo():
    """Create the legendary 12-dimensional Market Compass with sample data."""
    
    # Sample EOTS metrics (simulating real trading data)
    metrics_data = {
        # Inner Ring - Core EOTS Metrics
        'VAPI-FA': 1.8,      # Volatility-Adjusted Premium Intensity
        'DWFD': -1.2,        # Delta-Weighted Flow Divergence  
        'TW-LAF': 0.9,       # Time-Weighted Liquidity-Adjusted Flow
        'VRI 2.0': 1.5,      # Volatility Risk Index (normalized)
        'A-DAG': -0.7,       # Adaptive Delta-Adjusted Gamma
        'GIB': 2.1,          # Gamma Imbalance Barometer
        
        # Outer Ring - Custom Formulas
        'LWPAI': 1.3,        # Liquidity-Weighted Price Action Indicator
        'VABAI': -0.8,       # Volatility-Adjusted Bid/Ask Imbalance
        'AOFM': 1.6,         # Aggressive Order Flow Momentum
        'LIDB': 0.4,         # Liquidity-Implied Directional Bias
        'SVR': 1.1,          # Spread-to-Volatility Ratio
        'TPDLF': -1.4        # Theoretical Price Deviation with Liquidity Filter
    }
    
    # Multi-timeframe data (4 layers: 5m, 15m, 1h, 4h)
    timeframes = ['5m', '15m', '1h', '4h']
    timeframe_colors = ['rgba(66, 165, 245, 0.8)', 'rgba(107, 207, 127, 0.6)', 
                       'rgba(255, 167, 38, 0.4)', 'rgba(255, 71, 87, 0.3)']
    
    # Create the figure
    fig = go.Figure()
    
    # Add each timeframe layer
    for i, (timeframe, color) in enumerate(zip(timeframes, timeframe_colors)):
        # Simulate timeframe variations
        variation = 1.0 - (i * 0.15)  # Each timeframe slightly different
        
        metrics_names = list(metrics_data.keys())
        values = [metrics_data[name] * variation for name in metrics_names]
        
        # Close the radar chart by adding first point at the end
        theta = metrics_names + [metrics_names[0]]
        r = values + [values[0]]
        
        # Add radar trace
        fig.add_trace(go.Scatterpolar(
            r=r,
            theta=theta,
            fill='toself',
            fillcolor=color,
            line=dict(color=color.replace('rgba', 'rgb').replace(', 0.', ', 1.'), width=2),
            name=f'{timeframe} Timeframe',
            hovertemplate='<b>%{theta}</b><br>Value: %{r:.2f}<br>Timeframe: ' + timeframe + '<extra></extra>'
        ))
    
    # Add extreme reading indicators (pulsing effect simulation)
    extreme_metrics = [(name, value) for name, value in metrics_data.items() if abs(value) > 1.5]
    
    for name, value in extreme_metrics:
        # Add a marker for extreme readings
        fig.add_trace(go.Scatterpolar(
            r=[abs(value) * 1.1],
            theta=[name],
            mode='markers',
            marker=dict(
                size=15,
                color=AI_COLORS['danger'] if abs(value) > 2.0 else AI_COLORS['warning'],
                symbol='star',
                line=dict(width=2, color='white')
            ),
            name=f'Extreme: {name}',
            showlegend=False,
            hovertemplate=f'<b>EXTREME READING</b><br>{name}: {value:.2f}<extra></extra>'
        ))
    
    # Update layout for the legendary compass look
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0, 0, 0, 0.1)',
            radialaxis=dict(
                visible=True,
                range=[-3, 3],
                tickfont=dict(size=10, color='white'),
                gridcolor='rgba(255, 255, 255, 0.2)',
                linecolor='rgba(255, 255, 255, 0.3)'
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='white', family='Arial Black'),
                gridcolor='rgba(255, 255, 255, 0.3)',
                linecolor='rgba(255, 255, 255, 0.5)'
            )
        ),
        title=dict(
            text='🧭 LEGENDARY MARKET COMPASS - SPY',
            x=0.5,
            y=0.95,
            font=dict(size=20, color='white', family='Arial Black')
        ),
        paper_bgcolor='rgba(15, 23, 42, 0.95)',  # Dark blue background
        plot_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),
        height=600,
        width=800,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(color='white', size=10)
        )
    )
    
    return fig

def create_compass_metrics_summary():
    """Create a summary of compass metrics for display."""
    
    metrics_summary = {
        'Flow Intelligence': {
            'VAPI-FA': {'value': 1.8, 'status': 'Extreme Bullish', 'color': AI_COLORS['success']},
            'DWFD': {'value': -1.2, 'status': 'Bearish Flow', 'color': AI_COLORS['warning']},
            'TW-LAF': {'value': 0.9, 'status': 'Moderate Bull', 'color': AI_COLORS['info']}
        },
        'Volatility & Gamma': {
            'VRI 2.0': {'value': 1.5, 'status': 'Elevated Risk', 'color': AI_COLORS['warning']},
            'A-DAG': {'value': -0.7, 'status': 'Negative Gamma', 'color': AI_COLORS['info']},
            'GIB': {'value': 2.1, 'status': 'Extreme Imbalance', 'color': AI_COLORS['danger']}
        },
        'Custom Formulas': {
            'LWPAI': {'value': 1.3, 'status': 'Strong Signal', 'color': AI_COLORS['success']},
            'VABAI': {'value': -0.8, 'status': 'Bid Pressure', 'color': AI_COLORS['info']},
            'AOFM': {'value': 1.6, 'status': 'Aggressive Flow', 'color': AI_COLORS['warning']}
        }
    }
    
    return metrics_summary

def create_regime_analysis():
    """Create regime analysis based on compass data."""
    
    analysis = {
        'current_regime': 'BULL TRENDING',
        'confidence': 87,
        'transition_risk': 23,
        'key_drivers': [
            'Extreme VAPI-FA reading (1.8) indicates strong institutional buying',
            'GIB showing extreme gamma imbalance (2.1) - potential acceleration',
            'AOFM confirms aggressive bullish order flow (1.6)',
            'Bearish DWFD (-1.2) suggests some profit-taking but not dominant'
        ],
        'risk_factors': [
            'High VRI 2.0 (1.5) indicates elevated volatility risk',
            'Negative A-DAG (-0.7) shows dealer gamma positioning stress',
            'TPDLF divergence (-1.4) suggests potential mean reversion'
        ],
        'opportunities': [
            'Strong confluence across 5m, 15m, 1h timeframes',
            'LWPAI and AOFM alignment suggests sustained momentum',
            'Low transition risk (23%) indicates regime stability'
        ]
    }
    
    return analysis

if __name__ == "__main__":
    # Create the compass
    compass_fig = create_legendary_market_compass_demo()
    
    # Save as HTML for viewing
    compass_fig.write_html("/home/ubuntu/market_compass_demo.html", 
                          include_plotlyjs='cdn',
                          config={'displayModeBar': True, 'displaylogo': False})
    
    # Also save as PNG for quick preview
    compass_fig.write_image("/home/ubuntu/market_compass_demo.png", 
                           width=800, height=600, scale=2)
    
    print("🧭 LEGENDARY MARKET COMPASS DEMO CREATED!")
    print("📁 Files created:")
    print("   - market_compass_demo.html (Interactive)")
    print("   - market_compass_demo.png (Static image)")
    
    # Create metrics summary
    metrics = create_compass_metrics_summary()
    analysis = create_regime_analysis()
    
    print("\n📊 COMPASS METRICS SUMMARY:")
    for category, metrics_dict in metrics.items():
        print(f"\n{category}:")
        for metric, data in metrics_dict.items():
            print(f"  {metric}: {data['value']:.1f} ({data['status']})")
    
    print(f"\n🌊 REGIME ANALYSIS:")
    print(f"Current Regime: {analysis['current_regime']}")
    print(f"Confidence: {analysis['confidence']}%")
    print(f"Transition Risk: {analysis['transition_risk']}%")
    
    print("\n🎯 KEY INSIGHTS:")
    for insight in analysis['key_drivers'][:2]:
        print(f"  • {insight}")

