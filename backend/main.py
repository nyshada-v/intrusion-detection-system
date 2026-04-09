
"""
main.py - FastAPI Server + WebSocket + IDS Orchestrator
=========================================================
Wires together:
  - PacketCapture  (capture.py)   - live packet sniffing
  - IDSInferenceEngine (inference.py) - anomaly detection
  - WebSocket                     - real-time alerts to browser UI
  - FastAPI REST endpoints        - start/stop/status/interfaces

Run (as Administrator):
    cd D:\IDS_Project\backend
    venv\Scripts\activate
    python main.py
Then open: http://localhost:8000
"""
import sys
sys.path.append("D:/temp")
import asyncio
import json
import time
import threading
import uuid
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os

from capture import PacketCapture, list_interfaces
from inference import IDSInferenceEngine

# - App Setup -

app = FastAPI(title="IDS - Network Anomaly Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# - Global State -

engine    : Optional[IDSInferenceEngine] = None
capture   : Optional[PacketCapture]     = None
is_running: bool                         = False

# Alert history (last 100 alerts kept in memory)
alert_history : list[dict] = []
MAX_HISTORY   : int        = 100

# Live stats
stats = {
    "packets_captured" : 0,
    "flows_analysed"   : 0,
    "anomalies_detected": 0,
    "start_time"       : None,
    "status"           : "idle",   # idle | running | stopping
}

# - WebSocket Manager -

class ConnectionManager:
    """Manages all connected WebSocket clients (browser tabs)."""

    def __init__(self):
        self._clients: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._clients.append(ws)
        print(f"[WS] Client connected. Total: {len(self._clients)}")

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            if ws in self._clients:
                self._clients.remove(ws)
        print(f"[WS] Client disconnected. Total: {len(self._clients)}")

    async def broadcast(self, message: dict):
        """Send a message to all connected clients."""
        if not self._clients:
            return
        payload = json.dumps(message)
        dead    = []
        async with self._lock:
            clients = list(self._clients)
        for ws in clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)

    @property
    def client_count(self) -> int:
        return len(self._clients)


manager = ConnectionManager()

# We need an event loop reference to broadcast from sync threads
_event_loop: Optional[asyncio.AbstractEventLoop] = None


def broadcast_from_thread(message: dict):
    """Thread-safe broadcast - called from capture/inference threads."""
    if _event_loop and not _event_loop.is_closed():
        asyncio.run_coroutine_threadsafe(
            manager.broadcast(message), _event_loop
        )


# - Flow Callback - Called When a Flow Completes -

def on_flow_complete(features: dict):
    """
    Called by PacketCapture every time a flow is complete.
    Runs inference and broadcasts result to all WebSocket clients.
    """
    global stats

    try:
        stats["flows_analysed"] += 1

        # Run the 2-stage inference pipeline
        result = engine.predict(features)

        # Always broadcast a stats update
        stats["packets_captured"] = capture.stats()["packets_captured"]

        broadcast_from_thread({
            "type"  : "stats",
            "data"  : {
                "packets_captured"  : stats["packets_captured"],
                "flows_analysed"    : stats["flows_analysed"],
                "anomalies_detected": stats["anomalies_detected"],
                "active_flows"      : capture.stats()["active_flows"],
                "uptime_seconds"    : (
                    int(time.time() - stats["start_time"])
                    if stats["start_time"] else 0
                ),
            }
        })

        # If anomaly detected - broadcast alert
        if result["is_anomaly"]:
            stats["anomalies_detected"] += 1

            alert = {
                "id"          : str(uuid.uuid4()),
                "timestamp"   : datetime.now().isoformat(),
                "attack_type" : result["attack_type"],
                "confidence"  : round(result["confidence"] * 100, 1),
                "description" : result["description"],
                "advice"      : result["advice"],
                "if_score"    : round(result.get("if_score", 0), 4),
                "ae_error"    : round(result.get("ae_error", 0), 6),
                # Flow metadata for display
                "flow_info"   : {
                    "duration_ms"  : round(features.get("Flow Duration", 0) / 1000, 2),
                    "fwd_packets"  : int(features.get("Total Fwd Packets", 0)),
                    "bwd_packets"  : int(features.get("Total Backward Packets", 0)),
                    "bytes_per_s"  : round(features.get("Flow Bytes/s", 0), 2),
                    "protocol"     : "TCP" if features.get("SYN Flag Count", 0) >= 0 else "UDP",
                },
            }

            # Store in history
            alert_history.append(alert)
            if len(alert_history) > MAX_HISTORY:
                alert_history.pop(0)

            # Broadcast alert to all connected browser tabs
            broadcast_from_thread({
                "type" : "alert",
                "data" : alert,
            })

            print(f"[ALERT] {alert['attack_type']} | "
                  f"confidence={alert['confidence']}% | "
                  f"if_score={alert['if_score']}")

    except Exception as e:
        print(f"[Error] on_flow_complete: {e}")


# - Startup / Shutdown -

@app.on_event("startup")
async def startup():
    global engine, _event_loop

    _event_loop = asyncio.get_event_loop()

    print("[IDS] Loading ML models...")
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    engine    = IDSInferenceEngine(model_dir=model_dir)
    print("[IDS] - Models loaded. Server ready.")


@app.on_event("shutdown")
async def shutdown():
    global capture, is_running
    if capture and is_running:
        capture.stop()
        is_running = False


# - REST Endpoints -

@app.get("/api/interfaces")
async def get_interfaces():
    """Return list of available network interfaces for the UI dropdown."""
    try:
        ifaces = list_interfaces()
        return JSONResponse({"interfaces": ifaces})
    except Exception as e:
        return JSONResponse({"interfaces": [], "error": str(e)})


@app.post("/api/start")
async def start_capture(body: dict):
    """
    Start packet capture on the selected interface.
    Body: { "interface": "Ethernet" }
    """
    global capture, is_running, stats

    if is_running:
        return JSONResponse({"ok": False, "message": "Already running."})

    interface = body.get("interface", None)

    try:
        capture = PacketCapture(
            on_flow_complete=on_flow_complete,
            interface=interface,
            bpf_filter="ip",
        )
        capture.start()
        is_running             = True
        stats["status"]        = "running"
        stats["start_time"]    = time.time()
        stats["flows_analysed"]    = 0
        stats["anomalies_detected"] = 0
        stats["packets_captured"]   = 0

        await manager.broadcast({
            "type": "status",
            "data": {"status": "running", "interface": interface}
        })

        return JSONResponse({"ok": True, "message": f"Monitoring started on {interface or 'default interface'}."})

    except Exception as e:
        return JSONResponse({"ok": False, "message": str(e)}, status_code=500)


@app.post("/api/stop")
async def stop_capture():
    """Stop packet capture."""
    global capture, is_running, stats

    if not is_running:
        return JSONResponse({"ok": False, "message": "Not running."})

    capture.stop()
    is_running      = False
    stats["status"] = "idle"

    await manager.broadcast({
        "type": "status",
        "data": {"status": "idle"}
    })

    return JSONResponse({"ok": True, "message": "Monitoring stopped."})


@app.get("/api/status")
async def get_status():
    """Return current system status and stats."""
    return JSONResponse({
        "status"            : stats["status"],
        "packets_captured"  : stats["packets_captured"],
        "flows_analysed"    : stats["flows_analysed"],
        "anomalies_detected": stats["anomalies_detected"],
        "active_flows"      : capture.stats()["active_flows"] if capture else 0,
        "uptime_seconds"    : (
            int(time.time() - stats["start_time"])
            if stats["start_time"] and is_running else 0
        ),
        "ws_clients"        : manager.client_count,
    })


@app.get("/api/alerts")
async def get_alerts():
    """Return alert history (last 100)."""
    return JSONResponse({"alerts": list(reversed(alert_history))})


@app.delete("/api/alerts")
async def clear_alerts():
    """Clear alert history."""
    alert_history.clear()
    return JSONResponse({"ok": True})


# - WebSocket Endpoint -

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)

    # Send current state immediately on connect
    await ws.send_text(json.dumps({
        "type": "status",
        "data": {
            "status"            : stats["status"],
            "packets_captured"  : stats["packets_captured"],
            "flows_analysed"    : stats["flows_analysed"],
            "anomalies_detected": stats["anomalies_detected"],
        }
    }))

    # Send recent alert history
    if alert_history:
        await ws.send_text(json.dumps({
            "type": "history",
            "data": {"alerts": list(reversed(alert_history[-10:]))}
        }))

    try:
        while True:
            # Keep connection alive - client can send "ping"
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        await manager.disconnect(ws)


# - Serve Frontend -

# Serve the React build (frontend/dist) as static files
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")

if os.path.exists(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")

    @app.get("/", response_class=HTMLResponse)
    async def serve_ui():
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        with open(index_path) as f:
            return HTMLResponse(f.read())

    @app.get("/{full_path:path}", response_class=HTMLResponse)
    async def serve_spa(full_path: str):
        """Catch-all - return index.html for all routes (SPA routing)."""
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        with open(index_path) as f:
            return HTMLResponse(f.read())
else:
    @app.get("/")
    async def no_frontend():
        return JSONResponse({
            "message": "IDS Backend running. Frontend not built yet.",
            "docs"   : "http://localhost:8000/docs",
            "ws"     : "ws://localhost:8000/ws",
        })


# - Entry Point -

if __name__ == "__main__":
    print("=" * 55)
    print("  IDS - Network Anomaly Detector")
    print("  Starting server at http://localhost:8000")
    print("  API docs at    http://localhost:8000/docs")
    print("  Run as Administrator for packet capture!")
    print("=" * 55)

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,       # Never use reload=True with packet capture
        log_level="warning",
    )