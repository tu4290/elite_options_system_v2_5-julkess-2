# dashboard_application/layout_manager_v2_5.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE MASTER LAYOUT DEFINITION

from dash import dcc, html
import dash_bootstrap_components as dbc
import datetime

from dashboard_application import ids
from utils.config_manager_v2_5 import ConfigManagerV2_5

def create_control_panel(config: ConfigManagerV2_5) -> dbc.Card:
    """Creates the control panel with symbol input, fetch button, and settings."""
    try:
        # Get default values from config
        vis_defaults = config.config.visualization_settings.dashboard.get('defaults', {})
        default_symbol = vis_defaults.get('symbol', 'SPY')
        default_refresh = vis_defaults.get('refresh_interval_seconds', 30)
        default_dte_min = vis_defaults.get('dte_min', 0)
        default_dte_max = vis_defaults.get('dte_max', 45)
        default_price_range = vis_defaults.get('price_range_percent', 20)
        
        print(f"🎛️ Control Panel Config: defaults={vis_defaults}")
        print(f"🎛️ Control Panel: symbol={default_symbol}, refresh={default_refresh}")
        
    except Exception as e:
        print(f"❌ Error reading config in control panel: {e}")
        # Fallback values
        default_symbol = 'SPY'
        default_refresh = 30
        default_dte_min = 0
        default_dte_max = 45
        default_price_range = 20
    
    control_panel = dbc.Card([
        dbc.CardHeader(html.H5("🎛️ EOTS Control Panel", className="mb-0")),
        dbc.CardBody([
            # Row 1: Main Controls
            dbc.Row([
                dbc.Col([
                    dbc.Label("Symbol:", html_for=ids.ID_SYMBOL_INPUT, className="form-label"),
                    dbc.InputGroup([
                        dbc.Input(
                            id=ids.ID_SYMBOL_INPUT,
                            type="text",
                            value=default_symbol,
                            placeholder="Enter symbol (e.g., SPY)",
                            className="form-control"
                        )
                    ], className="control-input-group mb-2")
                ], width=2),
                dbc.Col([
                    dbc.Label("DTE Range:", html_for="dte-range-input", className="form-label"),
                    dbc.InputGroup([
                        dbc.Input(
                            id="dte-min-input",
                            type="number",
                            value=default_dte_min,
                            placeholder="Min",
                            min=0,
                            max=365,
                            className="form-control me-1"
                        ),
                        dbc.InputGroupText("to", className="bg-elite-surface text-elite-secondary"),
                        dbc.Input(
                            id="dte-max-input",
                            type="number",
                            value=default_dte_max,
                            placeholder="Max",
                            min=0,
                            max=365,
                            className="form-control"
                        )
                    ], size="sm", className="control-input-group mb-2")
                ], width=2),
                dbc.Col([
                    dbc.Label("Price Range %:", html_for="price-range-input", className="form-label"),
                    dbc.InputGroup([
                        dbc.Input(
                            id="price-range-input",
                            type="number",
                            value=default_price_range,
                            placeholder="±%",
                            min=1,
                            max=100,
                            step=1,
                            className="form-control"
                        ),
                        dbc.InputGroupText("%", className="bg-elite-surface text-elite-secondary")
                    ], size="sm", className="control-input-group mb-2")
                ], width=2),
                dbc.Col([
                    dbc.Label("Refresh:", html_for=ids.ID_REFRESH_INTERVAL_DROPDOWN, className="form-label"),
                    html.Div([
                        dcc.Dropdown(
                            id=ids.ID_REFRESH_INTERVAL_DROPDOWN,
                            options=[
                                {"label": "15s", "value": 15},
                                {"label": "30s", "value": 30},
                                {"label": "1m", "value": 60},
                                {"label": "2m", "value": 120},
                                {"label": "5m", "value": 300},
                                {"label": "Off", "value": 999999999}
                            ],
                            value=default_refresh,
                            style={
                                'backgroundColor': 'var(--elite-bg-primary)',
                                'border': '1px solid var(--elite-border-primary)',
                                'borderRadius': 'var(--elite-radius-md)'
                            }
                        )
                    ], className="control-input-group mb-2")
                ], width=2),
                dbc.Col([
                    dbc.Label("Actions:", className="d-block"),
                    dbc.Button(
                        "🚀 Fetch Data",
                        id=ids.ID_MANUAL_REFRESH_BUTTON,
                        size="sm",
                        className="btn-elite-primary mb-2 elite-focus-visible",
                        style={"width": "100%"}
                    )
                ], width=2),
                dbc.Col([
                    dbc.Label("Status:", className="d-block"),
                    html.Div(id="status-indicator", children=[
                        dbc.Badge("Ready", color="secondary", className="me-1"),
                        html.Small("Enter symbol and click Fetch Data", className="text-muted")
                    ])
                ], width=2)
            ], align="center"),
            
            # Row 2: STATUS UPDATE Section
            html.Hr(className="my-3"),
            dbc.Row([
                dbc.Col([
                    html.H6("📊 STATUS UPDATE", className="mb-2 text-primary"),
                    html.Div(id="status-update-display", children=[
                        dbc.Row([
                            dbc.Col([
                                html.Small("Symbol:", className="text-muted d-block"),
                                html.Span("---", id="current-symbol", className="fw-bold")
                            ], width=2),
                            dbc.Col([
                                html.Small("DTE Range:", className="text-muted d-block"),
                                html.Span("-- to --", id="current-dte-range", className="fw-bold")
                            ], width=2),
                            dbc.Col([
                                html.Small("Price Range:", className="text-muted d-block"),
                                html.Span("±--%", id="current-price-range", className="fw-bold")
                            ], width=2),
                            dbc.Col([
                                html.Small("Contracts:", className="text-muted d-block"),
                                html.Span("---", id="contracts-count", className="fw-bold")
                            ], width=1),
                            dbc.Col([
                                html.Small("Strikes:", className="text-muted d-block"),
                                html.Span("---", id="strikes-count", className="fw-bold")
                            ], width=1),
                            dbc.Col([
                                html.Small("Processing Time:", className="text-muted d-block"),
                                html.Span("---", id="processing-time", className="fw-bold")
                            ], width=2),
                            dbc.Col([
                                html.Small("Last Update:", className="text-muted d-block"),
                                html.Span("--:--:--", id="last-update-time", className="fw-bold")
                            ], width=2)
                        ], className="g-2")
                    ])
                ], width=12)
            ], className="mt-2")
        ])
    ], className="mb-4 elite-control-panel")
    
    print("🎛️ Control panel created successfully")
    return control_panel

def create_header(config: ConfigManagerV2_5) -> dbc.Navbar:
    """Creates the persistent header and navigation bar for the application."""
    
    # Dynamically build navigation links from the Pydantic config model
    modes_config = config.config.visualization_settings.dashboard.get('modes_detail_config', {})
    nav_links = []
    # Ensure 'main' mode is first if it exists
    if 'main' in modes_config:
        nav_links.append(
            dbc.NavLink(
                modes_config['main']['label'], 
                href="/", 
                active="exact",
                className="nav-link-custom"
            )
        )
    for mode, details in modes_config.items():
        if mode != 'main':
            nav_links.append(
                dbc.NavLink(
                    details['label'], 
                    href=f"/{mode}", 
                    active="exact",
                    className="nav-link-custom"
                )
            )

    # Get visualization settings for styling
    vis_settings = config.config.visualization_settings.dashboard.get('defaults', {})
    
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.NavbarBrand("EOTS v2.5 - Elite Options Trading System", className="ms-2"),
                ], width=6),
                dbc.Col([
                    dbc.Nav(nav_links, className="ms-auto", navbar=True)
                ], width=6)
            ], align="center", className="w-100")
        ], fluid=True),
        color="dark",
        dark=True,
        className="mb-3"
    )

def create_master_layout(config: ConfigManagerV2_5) -> html.Div:
    """
    Creates the master layout for the entire Dash application.

    This layout includes core non-visual components for state management and routing,
    and defines the main structure including the header and content area.
    """
    print("🏗️ Creating master layout...")
    
    # Fetch default refresh interval for dcc.Interval
    vis_defaults = config.config.visualization_settings.dashboard.get('defaults', {})
    initial_refresh_seconds = int(vis_defaults.get('refresh_interval_seconds', 60))
    initial_refresh_ms = initial_refresh_seconds * 1000

    # Determine if interval should be disabled if "Off" (very large number) is the default
    interval_disabled = True if initial_refresh_seconds >= 999999999 else False

    print(f"🏗️ Creating control panel...")
    control_panel_component = create_control_panel(config)
    print(f"🏗️ Control panel created: {control_panel_component is not None}")

    # --- Regime Display: Insert directly below control panel ---
    # Try to get a default/placeholder und_data for initial render
    try:
        from data_models import ProcessedUnderlyingAggregatesV2_5 # Updated import
        und_data_placeholder = ProcessedUnderlyingAggregatesV2_5(
            symbol='SPY',
            timestamp=datetime.datetime.now(),
            price=None,
            price_change_abs_und=None,
            price_change_pct_und=None,
            day_open_price_und=None,
            day_high_price_und=None,
            day_low_price_und=None,
            prev_day_close_price_und=None,
            u_volatility=None,
            day_volume=None,
            call_gxoi=None,
            put_gxoi=None,
            gammas_call_buy=None,
            gammas_call_sell=None,
            gammas_put_buy=None,
            gammas_put_sell=None,
            deltas_call_buy=None,
            deltas_call_sell=None,
            deltas_put_buy=None,
            deltas_put_sell=None,
            vegas_call_buy=None,
            vegas_call_sell=None,
            vegas_put_buy=None,
            vegas_put_sell=None,
            thetas_call_buy=None,
            thetas_call_sell=None,
            thetas_put_buy=None,
            thetas_put_sell=None,
            call_vxoi=None,
            put_vxoi=None,
            value_bs=None,
            volm_bs=None,
            deltas_buy=None,
            deltas_sell=None,
            vegas_buy=None,
            vegas_sell=None,
            thetas_buy=None,
            thetas_sell=None,
            volm_call_buy=None,
            volm_put_buy=None,
            volm_call_sell=None,
            volm_put_sell=None,
            value_call_buy=None,
            value_put_buy=None,
            value_call_sell=None,
            value_put_sell=None,
            vflowratio=None,
            dxoi=None,
            gxoi=None,
            vxoi=None,
            txoi=None,
            call_dxoi=None,
            put_dxoi=None,
            tradier_iv5_approx_smv_avg=None,
            total_call_oi_und=None,
            total_put_oi_und=None,
            total_call_vol_und=None,
            total_put_vol_und=None,
            tradier_open=None,
            tradier_high=None,
            tradier_low=None,
            tradier_close=None,
            tradier_volume=None,
            tradier_vwap=None
        )
    except Exception:
        und_data_placeholder = ProcessedUnderlyingAggregatesV2_5(
            symbol='SPY',
            timestamp=datetime.datetime.now(),
            price=None,
            price_change_abs_und=None,
            price_change_pct_und=None,
            day_open_price_und=None,
            day_high_price_und=None,
            day_low_price_und=None,
            prev_day_close_price_und=None,
            u_volatility=None,
            day_volume=None,
            call_gxoi=None,
            put_gxoi=None,
            gammas_call_buy=None,
            gammas_call_sell=None,
            gammas_put_buy=None,
            gammas_put_sell=None,
            deltas_call_buy=None,
            deltas_call_sell=None,
            deltas_put_buy=None,
            deltas_put_sell=None,
            vegas_call_buy=None,
            vegas_call_sell=None,
            vegas_put_buy=None,
            vegas_put_sell=None,
            thetas_call_buy=None,
            thetas_call_sell=None,
            thetas_put_buy=None,
            thetas_put_sell=None,
            call_vxoi=None,
            put_vxoi=None,
            value_bs=None,
            volm_bs=None,
            deltas_buy=None,
            deltas_sell=None,
            vegas_buy=None,
            vegas_sell=None,
            thetas_buy=None,
            thetas_sell=None,
            volm_call_buy=None,
            volm_put_buy=None,
            volm_call_sell=None,
            volm_put_sell=None,
            value_call_buy=None,
            value_put_buy=None,
            value_call_sell=None,
            value_put_sell=None,
            vflowratio=None,
            dxoi=None,
            gxoi=None,
            vxoi=None,
            txoi=None,
            call_dxoi=None,
            put_dxoi=None,
            tradier_iv5_approx_smv_avg=None,
            total_call_oi_und=None,
            total_put_oi_und=None,
            total_call_vol_und=None,
            total_put_vol_und=None,
            tradier_open=None,
            tradier_high=None,
            tradier_low=None,
            tradier_close=None,
            tradier_volume=None,
            tradier_vwap=None
        )
    regime_display_card = _create_regime_display(und_data_placeholder, config)
    # TODO: Wire up regime display to update with live data via callback if needed

    layout = html.Div(
        id='app-container',
        children=[
            dcc.Location(id=ids.ID_URL_LOCATION, refresh=False),
            dcc.Store(id=ids.ID_MAIN_DATA_STORE, storage_type='memory'), # Stores the main analysis bundle
            dcc.Interval(
                id=ids.ID_INTERVAL_LIVE_UPDATE,
                interval=initial_refresh_ms,
                n_intervals=0,
                disabled=interval_disabled # Control if interval timer is active
            ),
            
            create_header(config), # Header is persistent
            
            # Control panel with symbol input and fetch button
            dbc.Container([
                control_panel_component,
                html.Div(regime_display_card, id='regime-display-container'),  # <--- Regime display now in a container
            ], fluid=True),
            
            # Area for status alerts (e.g., data updated, errors)
            html.Div(id=ids.ID_STATUS_ALERT_CONTAINER,
                     style={"position": "fixed", "top": "120px", "right": "10px", "zIndex": "1050", "width": "auto"}),

            # Main content area, dynamically updated by callbacks based on URL
            html.Main(
                id='app-body',
                className='app-body container-fluid p-3', # Use container-fluid for responsive padding
                children=[
                    dbc.Container(id=ids.ID_PAGE_CONTENT, fluid=True, children=[ # Ensure page content also uses fluid container
                        dbc.Spinner(color="primary", children=html.Div("Waiting for initial data fetch..."))
                    ])
                ]
            )
        ]
    )
    
    print("🏗️ Master layout created successfully")
    return layout

def _create_regime_display(und_data, config):
    """Creates the market regime display card."""
    main_dash_settings = getattr(config, 'main_dashboard_settings', lambda: {})() if hasattr(config, 'main_dashboard_settings') else {}
    regime_settings = main_dash_settings.get("regime_display", {})
    regime_title = regime_settings.get("title", "Market Regime")
    regime_blurb = "🧠 Market Regime Engine: Analyzes current market conditions using multiple metrics. Helps determine optimal strategy types and risk parameters. Green = Bullish conditions, Red = Bearish conditions, Yellow = Transitional/Unclear."

    import dash_bootstrap_components as dbc
    from dash import html

    card_body_children = [
        html.H6(f"{regime_title}", className="elite-card-title text-center"),
        dbc.Button(
            "ℹ️ About",
            id="regime-about-toggle",
            color="link",
            size="sm",
            className="p-0 text-elite-secondary",
            style={'font-size': '0.75em'}
        ),
        dbc.Collapse(
            html.Small(regime_blurb, className="text-elite-secondary d-block mb-2", style={'font-size': '0.75em'}),
            id="regime-about-collapse",
            is_open=False
        )
    ]

    if not und_data or not hasattr(und_data, 'current_market_regime_v2_5'):
        card_body_children.append(dbc.Alert("Regime data unavailable.", color="info", className="mt-2 fade-in-up"))
    else:
        regime = getattr(und_data, 'current_market_regime_v2_5', None) or "UNKNOWN"
        if "BULL" in regime.upper() or "POSITIVE" in regime.upper():
            alert_color = "success"
        elif "BEAR" in regime.upper() or "NEGATIVE" in regime.upper():
            alert_color = "danger"
        elif "UNCLEAR" in regime.upper() or "TRANSITION" in regime.upper():
            alert_color = "warning"
        else:
            alert_color = "info"
        card_body_children.append(
            dbc.Alert(regime.replace("_", " ").title(), color=alert_color, className="mt-2 text-center fade-in-up")
        )

    return dbc.Card(dbc.CardBody(card_body_children, className="elite-card-body"), className="elite-card")