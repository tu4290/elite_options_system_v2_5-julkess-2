"""
AI-Controlled Adaptive Hexagon Market Compass
============================================

The MOE dynamically controls both SHAPE deformation and COLOR intensity
based on real-time market analysis. The hexagon morphs and changes colors
to reflect market conditions.
"""

import plotly.graph_objects as go
import numpy as np
import math
from datetime import datetime
import random

# AI Colors with intensity variations
AI_COLORS = {
    'primary': '#42A5F5',
    'success': '#6BCF7F', 
    'warning': '#FFA726',
    'danger': '#FF4757',
    'info': '#42A5F5',
    'muted': '#6C757D'
}

class MarketRegimeMOE:
    """MOE that controls compass shape and color based on market analysis."""
    
    def __init__(self):
        self.current_regime = "ANALYZING"
        self.volatility_level = 0.5
        self.trend_strength = 0.0
        self.flow_intensity = 0.0
        self.gamma_pressure = 0.0
        
    def analyze_market_conditions(self, metrics_data):
        """Analyze market and determine shape/color modifications."""
        
        # Extract key metrics
        vapi_fa = metrics_data.get('VAPI-FA', 0)
        dwfd = metrics_data.get('DWFD', 0)
        vri = metrics_data.get('VRI 2.0', 0)
        gib = metrics_data.get('GIB', 0)
        lwpai = metrics_data.get('LWPAI', 0)
        aofm = metrics_data.get('AOFM', 0)
        
        # Calculate market characteristics
        self.volatility_level = min(abs(vri) / 2.0, 1.0)
        self.trend_strength = (abs(vapi_fa) + abs(dwfd)) / 4.0
        self.flow_intensity = (abs(lwpai) + abs(aofm)) / 4.0
        self.gamma_pressure = min(abs(gib) / 3.0, 1.0)
        
        # Determine regime
        if abs(vapi_fa) > 1.5 and vapi_fa > 0:
            self.current_regime = "BULL_MOMENTUM"
        elif abs(vapi_fa) > 1.5 and vapi_fa < 0:
            self.current_regime = "BEAR_MOMENTUM"
        elif self.volatility_level > 0.7:
            self.current_regime = "HIGH_VOLATILITY"
        elif self.trend_strength < 0.3:
            self.current_regime = "CONSOLIDATION"
        else:
            self.current_regime = "TRANSITIONING"
    
    def get_shape_deformation(self):
        """Calculate how to deform the hexagon based on market conditions."""
        
        deformations = {}
        
        if self.current_regime == "BULL_MOMENTUM":
            # Stretch upward, compress downward
            deformations = {
                'VAPI-FA': 1.3,    # Stretch top
                'DWFD': 0.7,       # Compress right
                'VRI 2.0': 1.0,    # Normal
                'GIB': 1.2,        # Slight stretch
                'LWPAI': 0.8,      # Slight compress
                'AOFM': 1.1        # Slight stretch
            }
            
        elif self.current_regime == "BEAR_MOMENTUM":
            # Compress upward, stretch downward
            deformations = {
                'VAPI-FA': 0.7,    # Compress top
                'DWFD': 1.3,       # Stretch right
                'VRI 2.0': 1.0,    # Normal
                'GIB': 0.8,        # Compress
                'LWPAI': 1.2,      # Stretch
                'AOFM': 0.9        # Slight compress
            }
            
        elif self.current_regime == "HIGH_VOLATILITY":
            # Jagged, irregular shape
            deformations = {
                'VAPI-FA': 1.0 + random.uniform(-0.3, 0.3),
                'DWFD': 1.0 + random.uniform(-0.3, 0.3),
                'VRI 2.0': 1.4,    # Stretch volatility
                'GIB': 1.0 + random.uniform(-0.2, 0.2),
                'LWPAI': 1.0 + random.uniform(-0.3, 0.3),
                'AOFM': 1.0 + random.uniform(-0.3, 0.3)
            }
            
        elif self.current_regime == "CONSOLIDATION":
            # Compressed, tight shape
            deformations = {
                'VAPI-FA': 0.6,
                'DWFD': 0.6,
                'VRI 2.0': 0.5,
                'GIB': 0.7,
                'LWPAI': 0.6,
                'AOFM': 0.6
            }
            
        else:  # TRANSITIONING
            # Asymmetric, unbalanced shape
            deformations = {
                'VAPI-FA': 1.1,
                'DWFD': 0.9,
                'VRI 2.0': 1.2,
                'GIB': 0.8,
                'LWPAI': 1.0,
                'AOFM': 1.1
            }
        
        return deformations
    
    def get_color_scheme(self):
        """Get color scheme based on market regime and intensity."""
        
        base_colors = {
            'BULL_MOMENTUM': '#00FF88',      # Bright green
            'BEAR_MOMENTUM': '#FF4444',      # Bright red
            'HIGH_VOLATILITY': '#FFAA00',    # Orange
            'CONSOLIDATION': '#4488FF',      # Blue
            'TRANSITIONING': '#AA44FF'       # Purple
        }
        
        base_color = base_colors.get(self.current_regime, '#6C757D')
        
        # Calculate intensity (0.3 to 1.0)
        intensity = 0.3 + (self.flow_intensity * 0.7)
        
        # Create color variations for timeframes
        colors = {
            '15m': f"rgba({self._hex_to_rgb(base_color)}, {intensity})",
            '1h': f"rgba({self._hex_to_rgb(base_color)}, {intensity * 0.7})",
            '4h': f"rgba({self._hex_to_rgb(base_color)}, {intensity * 0.4})"
        }
        
        return colors, base_color
    
    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB string."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"{r}, {g}, {b}"
    
    def get_regime_description(self):
        """Get human-readable regime description."""
        descriptions = {
            'BULL_MOMENTUM': f"🚀 Bull Momentum (Intensity: {self.flow_intensity:.1f})",
            'BEAR_MOMENTUM': f"🐻 Bear Momentum (Intensity: {self.flow_intensity:.1f})",
            'HIGH_VOLATILITY': f"⚡ High Volatility (Level: {self.volatility_level:.1f})",
            'CONSOLIDATION': f"📊 Consolidation (Strength: {self.trend_strength:.1f})",
            'TRANSITIONING': f"🔄 Transitioning (Uncertainty: {1-self.trend_strength:.1f})"
        }
        return descriptions.get(self.current_regime, "❓ Unknown")

def create_moe_controlled_compass():
    """Create MOE-controlled adaptive hexagon compass."""
    
    # Sample market data
    metrics_data = {
        'VAPI-FA': 1.8,      # Strong bullish flow
        'DWFD': -0.8,        # Some bearish pressure
        'VRI 2.0': 1.2,      # Moderate volatility
        'GIB': 2.1,          # High gamma imbalance
        'LWPAI': 1.4,        # Strong custom signal
        'AOFM': 1.6          # Aggressive flow
    }
    
    # Initialize MOE and analyze market
    moe = MarketRegimeMOE()
    moe.analyze_market_conditions(metrics_data)
    
    # Get MOE-controlled modifications
    shape_deformations = moe.get_shape_deformation()
    colors, base_color = moe.get_color_scheme()
    
    # Create the figure
    fig = go.Figure()
    
    # Timeframes with MOE-controlled colors
    timeframes = ['15m', '1h', '4h']
    
    # Add each timeframe layer with MOE modifications
    for i, timeframe in enumerate(timeframes):
        metrics_names = list(metrics_data.keys())
        
        # Apply MOE shape deformations
        values = []
        for name in metrics_names:
            base_value = metrics_data[name]
            deformation = shape_deformations.get(name, 1.0)
            modified_value = base_value * deformation
            values.append(modified_value)
        
        # Close the hexagon
        theta = metrics_names + [metrics_names[0]]
        r = values + [values[0]]
        
        # Add trace with MOE-controlled color
        fig.add_trace(go.Scatterpolar(
            r=r,
            theta=theta,
            fill='toself',
            fillcolor=colors[timeframe],
            line=dict(
                color=base_color, 
                width=3
            ),
            name=f'{timeframe} (MOE)',
            hovertemplate='<b>%{theta}</b><br>Value: %{r:.2f}<br>Timeframe: ' + timeframe + '<br>MOE Regime: ' + moe.current_regime + '<extra></extra>'
        ))
    
    # Add extreme indicators with regime-based colors
    extreme_metrics = [(name, value) for name, value in metrics_data.items() if abs(value) > 1.5]
    
    for name, value in extreme_metrics:
        # Use regime color for extreme indicators
        fig.add_trace(go.Scatterpolar(
            r=[abs(value) * shape_deformations.get(name, 1.0) * 1.1],
            theta=[name],
            mode='markers',
            marker=dict(
                size=25,
                color=base_color,
                symbol='star',
                line=dict(width=3, color='white')
            ),
            name=f'MOE Alert: {name}',
            showlegend=False,
            hovertemplate=f'<b>🤖 MOE ALERT</b><br>{name}: {value:.2f}<br>Regime: {moe.current_regime}<extra></extra>'
        ))
    
    # Update layout with MOE regime info
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0, 0, 0, 0.1)',
            radialaxis=dict(
                visible=True,
                range=[-3, 3],
                tickfont=dict(size=12, color='white'),
                gridcolor='rgba(255, 255, 255, 0.3)',
                linecolor='rgba(255, 255, 255, 0.4)'
            ),
            angularaxis=dict(
                tickfont=dict(size=14, color='white', family='Arial Black'),
                gridcolor='rgba(255, 255, 255, 0.4)',
                linecolor='rgba(255, 255, 255, 0.6)',
                rotation=90,
                direction='clockwise'
            )
        ),
        title=dict(
            text=f'🤖 MOE COMPASS - {moe.get_regime_description()}',
            x=0.5,
            y=0.95,
            font=dict(size=20, color=base_color, family='Arial Black')
        ),
        paper_bgcolor='rgba(15, 23, 42, 0.95)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),
        height=600,
        width=600,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.05,
            xanchor="center",
            x=0.5,
            font=dict(color='white', size=12)
        )
    )
    
    return fig, moe

def create_moe_analysis_summary(moe):
    """Create MOE analysis summary."""
    
    return {
        'regime': moe.current_regime,
        'regime_description': moe.get_regime_description(),
        'shape_analysis': {
            'volatility_level': f"{moe.volatility_level:.1%}",
            'trend_strength': f"{moe.trend_strength:.1%}",
            'flow_intensity': f"{moe.flow_intensity:.1%}",
            'gamma_pressure': f"{moe.gamma_pressure:.1%}"
        },
        'moe_decisions': [
            f"🎨 Color Scheme: {moe.current_regime} regime colors",
            f"🔧 Shape Deformation: Based on {moe.current_regime} pattern",
            f"⚡ Intensity Level: {moe.flow_intensity:.1%} market energy",
            f"📊 Volatility Adjustment: {moe.volatility_level:.1%} chaos factor"
        ]
    }

if __name__ == "__main__":
    # Create MOE-controlled compass
    compass_fig, moe = create_moe_controlled_compass()
    
    # Save files
    compass_fig.write_html("/home/ubuntu/moe_compass_demo.html", 
                          include_plotlyjs='cdn',
                          config={'displayModeBar': True, 'displaylogo': False})
    
    compass_fig.write_image("/home/ubuntu/moe_compass_demo.png", 
                           width=600, height=600, scale=2)
    
    print("🤖 MOE-CONTROLLED COMPASS CREATED!")
    print("📁 Files created:")
    print("   - moe_compass_demo.html (Interactive)")
    print("   - moe_compass_demo.png (Static image)")
    
    # Show MOE analysis
    analysis = create_moe_analysis_summary(moe)
    
    print(f"\n🧠 MOE ANALYSIS:")
    print(f"Detected Regime: {analysis['regime_description']}")
    
    print(f"\n📊 SHAPE ANALYSIS:")
    for key, value in analysis['shape_analysis'].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n🤖 MOE DECISIONS:")
    for decision in analysis['moe_decisions']:
        print(f"  {decision}")
    
    print(f"\n🎯 COMPASS BEHAVIOR:")
    print(f"  • Shape morphed based on {analysis['regime']} pattern")
    print(f"  • Colors reflect regime intensity and confidence")
    print(f"  • Real-time adaptation to market conditions")
    print(f"  • AI-driven visual intelligence")

