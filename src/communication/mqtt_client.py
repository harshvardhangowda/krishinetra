"""
AI-Powered Precision Pest Management - MQTT Communication Client
File: src/communication/mqtt_client.py

Provides MQTT pub/sub abstraction for Raspberry Pi <-> ESP32 telemetry & commands.
Includes an in-memory mock client for zero-dependency testing and offline development.
"""

from typing import Any, Callable, Dict, List, Optional
import json


class MockMQTTClient:
    """In-memory pub/sub message bus simulating an MQTT broker."""

    def __init__(self, client_id: str = "rpi-edge-publisher"):
        self.client_id = client_id
        self.is_connected = False
        self.subscriptions: Dict[str, List[Callable[[str, str], None]]] = {}
        self.published_messages: List[Dict[str, Any]] = []

    def connect(self, host: str = "localhost", port: int = 1883) -> bool:
        """Simulate broker connection."""
        self.is_connected = True
        return True

    def disconnect(self) -> None:
        """Simulate broker disconnection."""
        self.is_connected = False

    def subscribe(self, topic: str, callback: Callable[[str, str], None]) -> None:
        """Register a callback for a specific topic."""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        self.subscriptions[topic].append(callback)

    def publish(self, topic: str, payload: str) -> bool:
        """Publish payload to subscribers of topic."""
        if not self.is_connected:
            return False

        self.published_messages.append({"topic": topic, "payload": payload})

        # Deliver to direct matches or wildcard subscribers
        if topic in self.subscriptions:
            for cb in self.subscriptions[topic]:
                try:
                    cb(topic, payload)
                except Exception as e:
                    print(f"[MockMQTT] Error in callback for {topic}: {e}")
        return True
