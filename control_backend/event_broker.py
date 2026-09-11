"""
control_backend/event_broker.py
Central asynchronous WebSocket event bus and subscriber manager.
Broadcasts typed events to all connected operator dashboards without blocking telemetry or inference.
"""

import asyncio
from collections import deque
import json
import logging
from typing import Dict, List, Optional, Set, Any
from fastapi import WebSocket
from pydantic import BaseModel

logger = logging.getLogger("antigravity.event_broker")


class EventBroker:
    def __init__(self, max_log_history: int = 300):
        self.active_connections: Set[WebSocket] = set()
        self.log_history: deque = deque(maxlen=max_log_history)
        self.latest_status: Optional[Dict[str, Any]] = None
        self.latest_prediction: Optional[Dict[str, Any]] = None
        self.latest_topology: Optional[Dict[str, Any]] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Operator client connected. Active clients: {len(self.active_connections)}")

        # Immediate handshake event
        await self._safe_send(websocket, {
            "type": "connected",
            "message": "Connected to Antigravity Real-Time Attack-Trajectory Bus."
        })

        if self.latest_status:
            await self._safe_send(websocket, self.latest_status)
        if self.latest_topology:
            await self._safe_send(websocket, self.latest_topology)
        if self.latest_prediction:
            await self._safe_send(websocket, self.latest_prediction)
        for log_event in list(self.log_history):
            await self._safe_send(websocket, log_event)

    def clear_prediction(self):
        """Immediately invalidates and clears cached prediction and notifies clients."""
        self.latest_prediction = None
        self.broadcast_sync({
            "type": "ml_reset",
            "ml_status": "standby"
        })

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Operator client disconnected. Remaining clients: {len(self.active_connections)}")

    async def _safe_send(self, websocket: WebSocket, message: Dict[str, Any]):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception:
            self.disconnect(websocket)

    async def broadcast(self, event: Any):
        """Broadcasts an event (dict or Pydantic model) to all active WebSocket clients."""
        if isinstance(event, BaseModel):
            data = event.dict() if hasattr(event, "dict") else event.model_dump()
        elif isinstance(event, dict):
            data = event
        else:
            data = {"type": "raw", "data": str(event)}

        # Update cache
        evt_type = data.get("type")
        if evt_type == "system_status":
            self.latest_status = data
            if not data.get("ml_active") and data.get("ml_status") == "standby":
                self.latest_prediction = None
        elif evt_type == "prediction":
            self.latest_prediction = data
        elif evt_type == "topology_update":
            self.latest_topology = data
        elif evt_type == "ml_reset":
            self.latest_prediction = None
        elif evt_type in ("command_output", "command_started", "command_completed", "error"):
            self.log_history.append(data)

        if not self.active_connections:
            return

        msg_text = json.dumps(data)
        for connection in list(self.active_connections):
            try:
                await connection.send_text(msg_text)
            except Exception:
                self.disconnect(connection)

    def broadcast_sync(self, event: Any):
        """Synchronous wrapper for broadcasting from background threads or callbacks."""
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(event), self.loop)
        else:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(self.broadcast(event), loop)
                else:
                    loop.run_until_complete(self.broadcast(event))
            except RuntimeError:
                pass


broker = EventBroker()
