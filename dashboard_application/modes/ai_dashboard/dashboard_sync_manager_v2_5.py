"""
DASHBOARD SYNCHRONIZATION MANAGER V2.5
======================================

Manages real-time synchronization between AI components and dashboard control panel.
Ensures all AI tools update seamlessly when symbol, DTE, or interval changes.

SYNCHRONIZATION FEATURES:
- Real-time symbol change propagation
- DTE filter synchronization
- Interval timer coordination
- Component state management
- Event-driven updates

Author: EOTS v2.5 Development Team
Version: 2.5.0 - "SEAMLESS SYNCHRONIZATION"
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from enum import Enum

from pydantic import BaseModel, Field
from data_models.eots_schemas_v2_5 import EOTSConfigV2_5, FinalAnalysisBundleV2_5

logger = logging.getLogger(__name__)

# ===== SYNCHRONIZATION MODELS =====

class SyncEventType(str, Enum):
    """Types of synchronization events."""
    SYMBOL_CHANGE = "symbol_change"
    DTE_CHANGE = "dte_change"
    INTERVAL_CHANGE = "interval_change"
    CONFIG_UPDATE = "config_update"
    DATA_REFRESH = "data_refresh"
    COMPONENT_REGISTER = "component_register"
    COMPONENT_UNREGISTER = "component_unregister"

class SyncEvent(BaseModel):
    """Pydantic model for synchronization events."""
    event_type: SyncEventType
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    source_component: Optional[str] = None
    target_components: Optional[List[str]] = None

class ComponentState(BaseModel):
    """Pydantic model for component state tracking."""
    component_id: str
    component_type: str
    last_update: datetime = Field(default_factory=datetime.now)
    current_symbol: str = "SPY"
    current_dte: Optional[int] = None
    refresh_interval: int = 30
    is_active: bool = True
    update_callback: Optional[str] = None  # Callback function name

class DashboardState(BaseModel):
    """Pydantic model for overall dashboard state."""
    current_symbol: str = "SPY"
    current_dte: Optional[int] = None
    refresh_interval: int = 30
    last_global_update: datetime = Field(default_factory=datetime.now)
    active_components: Dict[str, ComponentState] = Field(default_factory=dict)
    pending_updates: List[SyncEvent] = Field(default_factory=list)

# ===== DASHBOARD SYNCHRONIZATION MANAGER =====

class DashboardSyncManagerV2_5:
    """
    DASHBOARD SYNCHRONIZATION MANAGER V2.5
    
    Manages real-time synchronization between all AI components and the dashboard
    control panel to ensure seamless updates across the entire system.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Dashboard state
        self.state = DashboardState()
        
        # Event handlers
        self.event_handlers: Dict[SyncEventType, List[Callable]] = {
            event_type: [] for event_type in SyncEventType
        }
        
        # Component callbacks
        self.component_callbacks: Dict[str, Callable] = {}
        
        # Update queue for batch processing
        self.update_queue: asyncio.Queue = asyncio.Queue()
        
        # Background task for processing updates
        self.update_processor_task: Optional[asyncio.Task] = None
        
        self.logger.info("🔄 Dashboard Sync Manager V2.5 initialized")
    
    async def start(self):
        """Start the synchronization manager."""
        if self.update_processor_task is None:
            self.update_processor_task = asyncio.create_task(self._process_updates())
            self.logger.info("🚀 Dashboard synchronization started")
    
    async def stop(self):
        """Stop the synchronization manager."""
        if self.update_processor_task:
            self.update_processor_task.cancel()
            try:
                await self.update_processor_task
            except asyncio.CancelledError:
                pass
            self.update_processor_task = None
            self.logger.info("⏹️ Dashboard synchronization stopped")
    
    def register_component(self, component_id: str, component_type: str, 
                          update_callback: Optional[Callable] = None) -> ComponentState:
        """Register a component for synchronization."""
        component_state = ComponentState(
            component_id=component_id,
            component_type=component_type,
            current_symbol=self.state.current_symbol,
            current_dte=self.state.current_dte,
            refresh_interval=self.state.refresh_interval
        )
        
        self.state.active_components[component_id] = component_state
        
        if update_callback:
            self.component_callbacks[component_id] = update_callback
        
        self.logger.info(f"📝 Registered component: {component_id} ({component_type})")
        
        # Send registration event
        event = SyncEvent(
            event_type=SyncEventType.COMPONENT_REGISTER,
            data={"component_id": component_id, "component_type": component_type},
            source_component="sync_manager"
        )
        asyncio.create_task(self._queue_event(event))
        
        return component_state
    
    def unregister_component(self, component_id: str):
        """Unregister a component from synchronization."""
        if component_id in self.state.active_components:
            del self.state.active_components[component_id]
        
        if component_id in self.component_callbacks:
            del self.component_callbacks[component_id]
        
        self.logger.info(f"🗑️ Unregistered component: {component_id}")
        
        # Send unregistration event
        event = SyncEvent(
            event_type=SyncEventType.COMPONENT_UNREGISTER,
            data={"component_id": component_id},
            source_component="sync_manager"
        )
        asyncio.create_task(self._queue_event(event))
    
    async def update_symbol(self, new_symbol: str, source_component: Optional[str] = None):
        """Update the current symbol across all components."""
        if new_symbol != self.state.current_symbol:
            old_symbol = self.state.current_symbol
            self.state.current_symbol = new_symbol
            self.state.last_global_update = datetime.now()
            
            # Update all component states
            for component_state in self.state.active_components.values():
                component_state.current_symbol = new_symbol
                component_state.last_update = datetime.now()
            
            self.logger.info(f"🔄 Symbol changed: {old_symbol} → {new_symbol}")
            
            # Send symbol change event
            event = SyncEvent(
                event_type=SyncEventType.SYMBOL_CHANGE,
                data={"old_symbol": old_symbol, "new_symbol": new_symbol},
                source_component=source_component
            )
            await self._queue_event(event)
    
    async def update_dte(self, new_dte: Optional[int], source_component: Optional[str] = None):
        """Update the current DTE filter across all components."""
        if new_dte != self.state.current_dte:
            old_dte = self.state.current_dte
            self.state.current_dte = new_dte
            self.state.last_global_update = datetime.now()
            
            # Update all component states
            for component_state in self.state.active_components.values():
                component_state.current_dte = new_dte
                component_state.last_update = datetime.now()
            
            self.logger.info(f"🔄 DTE changed: {old_dte} → {new_dte}")
            
            # Send DTE change event
            event = SyncEvent(
                event_type=SyncEventType.DTE_CHANGE,
                data={"old_dte": old_dte, "new_dte": new_dte},
                source_component=source_component
            )
            await self._queue_event(event)
    
    async def update_interval(self, new_interval: int, source_component: Optional[str] = None):
        """Update the refresh interval across all components."""
        if new_interval != self.state.refresh_interval:
            old_interval = self.state.refresh_interval
            self.state.refresh_interval = new_interval
            self.state.last_global_update = datetime.now()
            
            # Update all component states
            for component_state in self.state.active_components.values():
                component_state.refresh_interval = new_interval
                component_state.last_update = datetime.now()
            
            self.logger.info(f"🔄 Interval changed: {old_interval}s → {new_interval}s")
            
            # Send interval change event
            event = SyncEvent(
                event_type=SyncEventType.INTERVAL_CHANGE,
                data={"old_interval": old_interval, "new_interval": new_interval},
                source_component=source_component
            )
            await self._queue_event(event)
    
    async def trigger_data_refresh(self, source_component: Optional[str] = None, 
                                  target_components: Optional[List[str]] = None):
        """Trigger a data refresh across specified or all components."""
        self.state.last_global_update = datetime.now()
        
        self.logger.info("🔄 Triggering data refresh")
        
        # Send data refresh event
        event = SyncEvent(
            event_type=SyncEventType.DATA_REFRESH,
            data={"trigger_time": datetime.now().isoformat()},
            source_component=source_component,
            target_components=target_components
        )
        await self._queue_event(event)
    
    def add_event_handler(self, event_type: SyncEventType, handler: Callable):
        """Add an event handler for specific event types."""
        self.event_handlers[event_type].append(handler)
        self.logger.debug(f"Added event handler for {event_type}")
    
    def remove_event_handler(self, event_type: SyncEventType, handler: Callable):
        """Remove an event handler."""
        if handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
            self.logger.debug(f"Removed event handler for {event_type}")
    
    async def _queue_event(self, event: SyncEvent):
        """Queue an event for processing."""
        await self.update_queue.put(event)
    
    async def _process_updates(self):
        """Background task to process update events."""
        while True:
            try:
                # Get event from queue
                event = await self.update_queue.get()
                
                # Process the event
                await self._handle_event(event)
                
                # Mark task as done
                self.update_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing update event: {e}")
    
    async def _handle_event(self, event: SyncEvent):
        """Handle a synchronization event."""
        try:
            # Call registered event handlers
            for handler in self.event_handlers[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    self.logger.error(f"Event handler error: {e}")
            
            # Call component-specific callbacks
            target_components = event.target_components or list(self.state.active_components.keys())
            
            for component_id in target_components:
                if component_id in self.component_callbacks:
                    callback = self.component_callbacks[component_id]
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(event)
                        else:
                            callback(event)
                    except Exception as e:
                        self.logger.error(f"Component callback error for {component_id}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error handling event {event.event_type}: {e}")
    
    def get_current_state(self) -> DashboardState:
        """Get the current dashboard state."""
        return self.state.model_copy()
    
    def get_component_state(self, component_id: str) -> Optional[ComponentState]:
        """Get the state of a specific component."""
        return self.state.active_components.get(component_id)

# ===== GLOBAL SYNC MANAGER INSTANCE =====

_global_sync_manager: Optional[DashboardSyncManagerV2_5] = None

def get_sync_manager() -> DashboardSyncManagerV2_5:
    """Get or create the global synchronization manager."""
    global _global_sync_manager
    if _global_sync_manager is None:
        _global_sync_manager = DashboardSyncManagerV2_5()
    return _global_sync_manager

async def start_dashboard_sync():
    """Start the global dashboard synchronization."""
    sync_manager = get_sync_manager()
    await sync_manager.start()

async def stop_dashboard_sync():
    """Stop the global dashboard synchronization."""
    sync_manager = get_sync_manager()
    await sync_manager.stop()
