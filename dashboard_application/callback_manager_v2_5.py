# dashboard_application/callback_manager.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE CALLBACK MANAGER

import logging
import json
import importlib
from typing import Any, Optional, List, Dict
import datetime

import dash
from dash import html, Input, Output, State, ctx, no_update, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# EOTS V2.5 Imports
from core_analytics_engine.its_orchestrator_v2_5 import ITSOrchestratorV2_5
from utils.config_manager_v2_5 import ConfigManagerV2_5
from data_models import FinalAnalysisBundleV2_5 # Updated top-level import
from . import ids
from .utils_dashboard_v2_5 import create_empty_figure
from dashboard_application.layout_manager_v2_5 import _create_regime_display

# --- Module-Specific Logger & Global References ---
callback_logger = logging.getLogger(__name__)
ORCHESTRATOR_REF: Optional[ITSOrchestratorV2_5] = None
CONFIG_REF: Optional[ConfigManagerV2_5] = None

def register_v2_5_callbacks(app: dash.Dash, orchestrator: ITSOrchestratorV2_5, config: ConfigManagerV2_5):
    """Registers all v2.5 callbacks with the Dash app instance."""
    global ORCHESTRATOR_REF, CONFIG_REF
    ORCHESTRATOR_REF = orchestrator
    CONFIG_REF = config
    callback_logger.info("Registering EOTS v2.5 authoritative callbacks...")

    # --- Primary Data Fetching and Storage Callback ---
    @app.callback(
        Output(ids.ID_MAIN_DATA_STORE, 'data'),
        Output(ids.ID_STATUS_ALERT_CONTAINER, 'children'),
        Input(ids.ID_MANUAL_REFRESH_BUTTON, 'n_clicks'),
        Input(ids.ID_INTERVAL_LIVE_UPDATE, 'n_intervals'),
        State(ids.ID_SYMBOL_INPUT, 'value'),
        State('dte-min-input', 'value'),
        State('dte-max-input', 'value'),
        State('price-range-input', 'value'),
        prevent_initial_call=True
    )
    def update_analysis_bundle_store(n_clicks: int, n_intervals: int, symbol: str, dte_min: int, dte_max: int, price_range_percent: int) -> tuple:
        """
        The primary data callback. Triggered by manual refresh or a timer.
        Calls the orchestrator to get the latest analysis and stores it.
        """
        callback_logger.info(f"🔍 CALLBACK TRIGGERED: n_clicks={n_clicks}, n_intervals={n_intervals}, symbol={symbol}")
        callback_logger.info(f"🔍 Control Panel Params: dte_min={dte_min}, dte_max={dte_max}, price_range_percent={price_range_percent}")
        callback_logger.info(f"🔍 ORCHESTRATOR_REF exists: {ORCHESTRATOR_REF is not None}")
        callback_logger.info(f"🔍 ctx.triggered: {ctx.triggered}")
        callback_logger.info(f"🔍 ctx.triggered_id: {ctx.triggered_id}")
        
        if not ORCHESTRATOR_REF:
            callback_logger.error(f"❌ ORCHESTRATOR_REF is None - cannot fetch data")
            return no_update, dbc.Alert("System error: Orchestrator not initialized", color="danger", duration=10000)
            
        if not symbol:
            callback_logger.warning(f"❌ No symbol provided - cannot fetch data")
            return no_update, dbc.Alert("Please enter a symbol first", color="warning", duration=4000)

        # Validate and set defaults for control panel parameters
        dte_min = dte_min if dte_min is not None else 0
        dte_max = dte_max if dte_max is not None else 45
        price_range_percent = price_range_percent if price_range_percent is not None else 20

        trigger_id = ctx.triggered_id
        callback_logger.info(f"▶️ Data fetch triggered by '{trigger_id}' for symbol '{symbol}' with DTE range [{dte_min}, {dte_max}] and price range ±{price_range_percent}%.")
        
        try:
            callback_logger.info(f"🚀 Calling orchestrator.run_full_analysis_cycle('{symbol}', dte_min={dte_min}, dte_max={dte_max}, price_range_percent={price_range_percent})...")
            analysis_bundle = ORCHESTRATOR_REF.run_full_analysis_cycle(symbol, dte_min=dte_min, dte_max=dte_max, price_range_percent=price_range_percent)
            callback_logger.info(f"📊 Orchestrator returned: {analysis_bundle is not None}")
            
            if not analysis_bundle:
                callback_logger.error(f"❌ Orchestrator returned None for {symbol}")
                alert = dbc.Alert(f"Orchestrator did not return a bundle for {symbol}.", color="warning", duration=4000)
                return no_update, alert
            
            callback_logger.info(f"✅ Analysis bundle received for {symbol}")
            # Serialize the Pydantic model to JSON for storage
            bundle_json = analysis_bundle.model_dump_json()
            callback_logger.info(f"📦 Bundle serialized successfully - {len(bundle_json)} characters")
            
            status_message = f"Data updated for {symbol} at {analysis_bundle.bundle_timestamp.strftime('%H:%M:%S')}."
            alert = dbc.Alert(status_message, color="success", duration=4000)
            callback_logger.info(f"🎉 Data fetch completed successfully for {symbol}")
            return bundle_json, alert

        except Exception as e:
            callback_logger.error(f"💥 Error running full analysis cycle for {symbol}: {e}", exc_info=True)
            alert = dbc.Alert(f"Failed to fetch data for {symbol}: {str(e)}", color="danger", duration=10000)
            return no_update, alert

    # --- Dynamic Mode and Chart Rendering Callback ---
    @app.callback(
        Output(ids.ID_PAGE_CONTENT, 'children'),
        Input(ids.ID_MAIN_DATA_STORE, 'data'),
        State(ids.ID_URL_LOCATION, 'pathname') # Use URL to determine the mode
    )
    def render_mode_content(bundle_json: Optional[str], pathname: str) -> Any:
        """
        Renders the entire layout for the currently selected mode.
        This is the central UI update callback.
        """
        callback_logger.info(f"🎨 RENDER CALLBACK: bundle_json={'has data' if bundle_json else 'None'}, pathname='{pathname}'")
        
        if not bundle_json:
            callback_logger.warning("❌ No bundle data available - showing wait message")
            return dbc.Alert("Waiting for initial data fetch...", color="info")

        # Determine mode from URL path, default to main
        if not pathname or pathname == '/':
            mode_key = 'main'
        else:
            mode_key = pathname.strip('/').split('/')[0]
            
        callback_logger.info(f"🎨 Determined mode_key: '{mode_key}' from pathname: '{pathname}'")
        
        # Get the full dashboard config and extract modes_detail_config from it
        dashboard_config = CONFIG_REF.config.visualization_settings.dashboard if CONFIG_REF else {}
        modes_config = dashboard_config.get('modes_detail_config', {})
        if callback_logger.isEnabledFor(logging.DEBUG):
            callback_logger.debug(f"🎨 Available modes in config: {list(modes_config.keys())}")
        
        mode_info = modes_config.get(mode_key) or modes_config.get('main')

        if not mode_info:
            callback_logger.error(f"❌ Configuration for mode '{mode_key}' not found")
            return dbc.Alert(f"Configuration for mode '{mode_key}' not found.", color="danger")

        callback_logger.info(f"🎨 Mode info found: {mode_info}")

        try:
            # Dynamically import the required display module
            callback_logger.info(f"🎨 Importing module: dashboard_application.modes.{mode_info['module_name']}")
            display_module = importlib.import_module(f".modes.{mode_info['module_name']}", package='dashboard_application')
            
            # The display module's create_layout function generates the necessary charts and structure
            # It now receives the full data bundle to do its work.
            callback_logger.info(f"🎨 Deserializing bundle JSON ({len(bundle_json)} chars)")
            bundle = FinalAnalysisBundleV2_5.model_validate_json(bundle_json)
            callback_logger.info(f"🎨 Calling {mode_info['module_name']}.create_layout()")
            mode_layout = display_module.create_layout(bundle, CONFIG_REF)
            callback_logger.info(f"🎨 Layout created successfully for mode '{mode_key}'")
            
            return mode_layout
        except ImportError:
            callback_logger.error(f"Could not import display module: {mode_info['module_name']}")
            return dbc.Alert(f"Error loading UI module for mode '{mode_key}'.", color="danger")
        except Exception as e:
            callback_logger.error(f"Error rendering layout for mode '{mode_key}': {e}", exc_info=True)
            return dbc.Alert(f"An unexpected error occurred while rendering the {mode_key} view.", color="danger")
            
    # --- Callback to update Refresh Interval ---
    @app.callback(
        Output(ids.ID_INTERVAL_LIVE_UPDATE, 'interval'),
        Input(ids.ID_REFRESH_INTERVAL_DROPDOWN, 'value')
    )
    def update_refresh_interval(interval_seconds: str) -> int:
        """Updates the dcc.Interval component's refresh rate."""
        return int(interval_seconds) * 1000 if interval_seconds else 60 * 1000

    # --- Status Update Display Callback ---
    @app.callback(
        [
            Output('current-symbol', 'children'),
            Output('current-dte-range', 'children'),
            Output('current-price-range', 'children'),
            Output('contracts-count', 'children'),
            Output('strikes-count', 'children'),
            Output('processing-time', 'children'),
            Output('last-update-time', 'children')
        ],
        [
            Input(ids.ID_MAIN_DATA_STORE, 'data'),
            Input(ids.ID_INTERVAL_LIVE_UPDATE, 'n_intervals')
        ],
        [
            State(ids.ID_SYMBOL_INPUT, 'value'),
            State('dte-min-input', 'value'),
            State('dte-max-input', 'value'),
            State('price-range-input', 'value')
        ],
        prevent_initial_call=True
    )
    def update_status_display(bundle_json: str, n_intervals: int, symbol: str, dte_min: int, dte_max: int, price_range_percent: int) -> tuple:
        """Updates the status display with current analysis information."""
        if not bundle_json:
            return ("---", "-- to --", "±--%", "---", "---", "---", "--:--:--")
        
        try:
            bundle = FinalAnalysisBundleV2_5.model_validate_json(bundle_json)
            
            # Extract information from bundle
            symbol_display = symbol or bundle.target_symbol or "Unknown"
            timestamp = bundle.bundle_timestamp
            
            # Format timestamp
            if timestamp:
                last_update = timestamp.strftime("%H:%M:%S")
            else:
                last_update = "--:--:--"
            
            # Use control panel values for DTE range display
            if dte_min is not None and dte_max is not None:
                if dte_min == dte_max:
                    dte_range = f"{dte_min} DTE"
                else:
                    dte_range = f"{dte_min} to {dte_max}"
            else:
                dte_range = "-- to --"
            
            # Use control panel value for price range display
            if price_range_percent is not None:
                price_range = f"±{price_range_percent}%"
            else:
                price_range = "±--%"
            
            # Get contracts and strikes count from actual data
            if bundle.processed_data_bundle and bundle.processed_data_bundle.options_data_with_metrics:
                contracts_count = len(bundle.processed_data_bundle.options_data_with_metrics)
            else:
                contracts_count = 0
            
            # Get strikes count
            if bundle.processed_data_bundle and bundle.processed_data_bundle.strike_level_data_with_metrics:
                strikes_count = len(bundle.processed_data_bundle.strike_level_data_with_metrics)
            else:
                strikes_count = 0
            
            # Calculate processing time from bundle timestamps
            processing_time_display = "---"
            if (bundle.bundle_timestamp and 
                bundle.processed_data_bundle and 
                bundle.processed_data_bundle.processing_timestamp):
                
                start_time = bundle.processed_data_bundle.processing_timestamp
                end_time = bundle.bundle_timestamp
                
                # Handle timezone compatibility
                if start_time.tzinfo != end_time.tzinfo:
                    if start_time.tzinfo is None:
                        start_time = start_time.replace(tzinfo=end_time.tzinfo)
                    elif end_time.tzinfo is None:
                        end_time = end_time.replace(tzinfo=start_time.tzinfo)
                
                processing_duration = (end_time - start_time).total_seconds()
                
                # Format processing time in a more readable way
                if processing_duration < 0.001:  # Less than 1ms
                    processing_time_display = "<1ms"
                elif processing_duration < 1:  # Less than 1 second, show in milliseconds
                    ms = processing_duration * 1000
                    if ms < 10:
                        processing_time_display = f"{ms:.1f}ms"
                    else:
                        processing_time_display = f"{ms:.0f}ms"
                elif processing_duration < 60:  # Less than 1 minute, show in seconds
                    if processing_duration < 10:
                        processing_time_display = f"{processing_duration:.2f}s"
                    else:
                        processing_time_display = f"{processing_duration:.1f}s"
                else:  # 1 minute or more
                    minutes = int(processing_duration // 60)
                    seconds = processing_duration % 60
                    processing_time_display = f"{minutes}m {seconds:.1f}s"
            
            return (
                symbol_display,
                dte_range,
                price_range,
                str(contracts_count),
                str(strikes_count),
                processing_time_display,
                last_update
            )
            
        except Exception as e:
            callback_logger.error(f"Error updating status display: {e}")
            return ("ERROR", "-- to --", "±--%", "---", "---", "---", "ERROR")

    # --- Collapsible About Section Callbacks ---
    @app.callback(
        Output("regime-about-collapse", "is_open"),
        Input("regime-about-toggle", "n_clicks"),
        State("regime-about-collapse", "is_open"),
        prevent_initial_call=True
    )
    def toggle_regime_about(n_clicks, is_open):
        """Toggle the regime about section."""
        if n_clicks:
            return not is_open
        return is_open

    # Generic callback for all chart about sections using pattern matching
    @app.callback(
        Output({"type": "about-collapse", "index": dash.MATCH}, "is_open"),
        Input({"type": "about-toggle", "index": dash.MATCH}, "n_clicks"),
        State({"type": "about-collapse", "index": dash.MATCH}, "is_open"),
        prevent_initial_call=True
    )
    def toggle_chart_about(n_clicks, is_open):
        """Toggle chart about section independently using pattern matching (MATCH)."""
        if n_clicks:
            return not is_open
        return is_open

    # --- Collapsible About Section Callback (for all cards using the new pattern) ---
    @app.callback(
        Output({"type": "about-collapse", "section": dash.MATCH}, "is_open"),
        Input({"type": "about-toggle-btn", "section": dash.MATCH}, "n_clicks"),
        State({"type": "about-collapse", "section": dash.MATCH}, "is_open"),
        prevent_initial_call=True
    )
    def toggle_about_section(n_clicks, is_open):
        if n_clicks:
            return not is_open
        return is_open

    # --- Regime Display Live Update Callback ---
    @app.callback(
        Output('regime-display-container', 'children'),
        Input(ids.ID_MAIN_DATA_STORE, 'data'),
        prevent_initial_call=False
    )
    def update_regime_display(bundle_json):
        try:
            if not bundle_json:
                # No data yet, show placeholder
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
                return _create_regime_display(und_data_placeholder, config)
            from data_models.eots_schemas_v2_5 import FinalAnalysisBundleV2_5
            bundle = FinalAnalysisBundleV2_5.model_validate_json(bundle_json)
            und_data = getattr(bundle.processed_data_bundle, 'underlying_data_enriched', None)
            if und_data is None:
                # Fallback to placeholder
                from data_models import ProcessedUnderlyingAggregatesV2_5 # Updated import
                und_data = ProcessedUnderlyingAggregatesV2_5(
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
            return _create_regime_display(und_data, config)
        except Exception as e:
            import traceback
            print(f"Error updating regime display: {e}\n{traceback.format_exc()}")
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
            return _create_regime_display(und_data_placeholder, config)

    callback_logger.info("EOTS v2.5 authoritative callbacks registered successfully.")