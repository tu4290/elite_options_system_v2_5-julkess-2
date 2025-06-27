"""
Simple MOE Trading Compass
=========================

COLOR = WHAT TO DO
SHAPE = WHERE THE ACTION IS

Takes 1000 data points, makes it simple as fuck.
"""

import plotly.graph_objects as go
import numpy as np
import math

class SimpleMOE:
    """Simple MOE that makes trading decisions visual."""
    
    def __init__(self):
        self.action_color = "GREEN"
        self.focus_metric = "VAPI-FA"
        self.confidence = 0.8
    
    def analyze_simple(self, metrics):
        """Simple analysis: what to do + where to focus."""
        
        # Find the strongest signal
        strongest_metric = max(metrics.items(), key=lambda x: abs(x[1]))
        self.focus_metric = strongest_metric[0]
        strongest_value = strongest_metric[1]
        
        # Determine action based on conditions
        vapi_fa = metrics.get('VAPI-FA', 0)
        gib = metrics.get('GIB', 0)
        vri = metrics.get('VRI 2.0', 0)
        
        # Simple rules
        if abs(gib) > 2.0 and abs(vri) > 1.5:
            self.action_color = "RED"  # Volatility explosion
        elif abs(vri) < 0.5 and abs(vapi_fa) < 0.5:
            self.action_color = "BLUE"  # Sell premium
        elif abs(vapi_fa) > 1.5:
            self.action_color = "GREEN"  # Trend continuation
        else:
            self.action_color = "ORANGE"  # Danger/caution
        
        self.confidence = min(abs(strongest_value) / 3.0, 1.0)
    
    def get_action_meaning(self):
        """What the color means."""
        meanings = {
            "RED": "🔥 BUY OPTIONS - Volatility explosion coming",
            "BLUE": "💰 SELL PREMIUM - Volatility dying", 
            "GREEN": "🚀 RIDE TREND - Momentum continuing",
            "ORANGE": "⚠️ REDUCE SIZE - Danger zone"
        }
        return meanings.get(self.action_color, "Unknown")
    
    def get_focus_meaning(self):
        """Where the action is."""
        meanings = {
            "VAPI-FA": "📊 FLOW-BASED - Institutional activity",
            "DWFD": "🧠 SMART MONEY - Big player moves",
            "VRI 2.0": "⚡ VOLATILITY - Risk/reward setup",
            "GIB": "🎯 GAMMA - Acceleration potential",
            "LWPAI": "📈 MOMENTUM - Trend strength",
            "AOFM": "💥 AGGRESSIVE - High conviction flow"
        }
        return meanings.get(self.focus_metric, "Unknown")

def create_simple_compass():
    """Create simple compass: color=action, shape=focus."""
    
    # Sample data
    metrics = {
        'VAPI-FA': 1.8,    # Strong
        'DWFD': -0.5,      # Weak
        'VRI 2.0': 1.2,    # Moderate
        'GIB': 2.3,        # EXTREME
        'LWPAI': 1.1,      # Moderate
        'AOFM': 1.4        # Strong
    }
    
    # MOE analysis
    moe = SimpleMOE()
    moe.analyze_simple(metrics)
    
    # Color mapping
    colors = {
        "RED": "#FF4444",      # Buy options
        "BLUE": "#4488FF",     # Sell premium  
        "GREEN": "#44FF88",    # Ride trend
        "ORANGE": "#FFAA44"    # Danger
    }
    
    base_color = colors[moe.action_color]
    
    # Create shape that bulges towards focus metric
    metrics_names = list(metrics.keys())
    focus_index = metrics_names.index(moe.focus_metric)
    
    # Base values
    values = list(metrics.values())
    
    # Bulge towards focus metric
    bulge_factor = 1.5  # How much to emphasize
    for i, val in enumerate(values):
        if i == focus_index:
            values[i] = val * bulge_factor  # Emphasize focus metric
        else:
            values[i] = val * 0.8  # De-emphasize others
    
    # Create figure
    fig = go.Figure()
    
    # Close the shape
    theta = metrics_names + [metrics_names[0]]
    r = values + [values[0]]
    
    # Add the shape
    # Convert hex to rgba properly
    r_val = int(base_color[1:3], 16)
    g_val = int(base_color[3:5], 16) 
    b_val = int(base_color[5:7], 16)
    
    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=theta,
        fill='toself',
        fillcolor=f"rgba({r_val}, {g_val}, {b_val}, 0.7)",
        line=dict(color=base_color, width=4),
        name=f"Action: {moe.action_color}",
        hovertemplate='<b>%{theta}</b><br>Value: %{r:.2f}<br>Action: ' + moe.get_action_meaning() + '<extra></extra>'
    ))
    
    # Add focus indicator
    fig.add_trace(go.Scatterpolar(
        r=[values[focus_index] * 1.1],
        theta=[moe.focus_metric],
        mode='markers',
        marker=dict(
            size=30,
            color=base_color,
            symbol='star',
            line=dict(width=3, color='white')
        ),
        name=f"Focus: {moe.focus_metric}",
        showlegend=False,
        hovertemplate=f'<b>🎯 FOCUS HERE</b><br>{moe.get_focus_meaning()}<extra></extra>'
    ))
    
    # Layout
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0, 0, 0, 0.1)',
            radialaxis=dict(
                visible=True,
                range=[0, 4],
                tickfont=dict(size=12, color='white'),
                gridcolor='rgba(255, 255, 255, 0.3)'
            ),
            angularaxis=dict(
                tickfont=dict(size=14, color='white', family='Arial Black'),
                gridcolor='rgba(255, 255, 255, 0.4)',
                rotation=90,
                direction='clockwise'
            )
        ),
        title=dict(
            text=f'🤖 SIMPLE MOE: {moe.get_action_meaning()}',
            x=0.5,
            y=0.95,
            font=dict(size=18, color=base_color, family='Arial Black')
        ),
        paper_bgcolor='rgba(15, 23, 42, 0.95)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),
        height=600,
        width=600,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig, moe

if __name__ == "__main__":
    # Create simple compass
    fig, moe = create_simple_compass()
    
    # Save files
    fig.write_html("/home/ubuntu/simple_moe_compass.html")
    fig.write_image("/home/ubuntu/simple_moe_compass.png", width=600, height=600, scale=2)
    
    print("🤖 SIMPLE MOE COMPASS CREATED!")
    print("\n📋 SIMPLE RULES:")
    print("COLOR = WHAT TO DO:")
    print("  🔴 RED = Buy options (volatility explosion)")
    print("  🔵 BLUE = Sell premium (volatility dying)")
    print("  🟢 GREEN = Ride trend (momentum continuing)")
    print("  🟠 ORANGE = Reduce size (danger zone)")
    
    print("\nSHAPE = WHERE THE ACTION IS:")
    print("  Bulges towards the metric with the opportunity")
    
    print(f"\n🎯 CURRENT READING:")
    print(f"Action: {moe.get_action_meaning()}")
    print(f"Focus: {moe.get_focus_meaning()}")
    print(f"Confidence: {moe.confidence:.0%}")
    
    print(f"\n💡 TRANSLATION:")
    print(f"The compass is {moe.action_color} and bulging towards {moe.focus_metric}")
    print(f"This means: {moe.get_action_meaning()}")
    print(f"Because: {moe.get_focus_meaning()}")
    print(f"Confidence: {moe.confidence:.0%}")

