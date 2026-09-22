"""
streamer.py — Runs a lightweight asyncio WebSocket server on its own thread
and broadcasts JSON telemetry/detection updates to any connected Ground
Control Station clients. main.py calls `push()` from the main (synchronous)
loop; push() hands the message to the streamer's own event loop thread-safely.
"""

import asyncio
import json
import threading

import websockets


class Streamer:
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self._clients = set()
        self._loop = None
        self._thread = None
        self._server = None

    async def _handler(self, websocket):
        self._clients.add(websocket)
        try:
            async for _ in websocket:
                pass  # this channel is broadcast-only; ignore inbound messages
        finally:
            self._clients.discard(websocket)

    async def _broadcast(self, message):
        if not self._clients:
            return
        dead = []
        for ws in self._clients:
            try:
                await ws.send(message)
            except websockets.exceptions.ConnectionClosed:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def _serve():
            self._server = await websockets.serve(self._handler, self.host, self.port)
            await self._server.wait_closed()

        self._loop.run_until_complete(_serve())

    def start(self):
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        return self

    def push(self, data: dict):
        """Thread-safe: call this from the synchronous main loop."""
        if self._loop is None:
            return
        message = json.dumps(data)
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self._loop)

    def stop(self):
        if self._loop is not None and self._server is not None:
            self._loop.call_soon_threadsafe(self._server.close)
