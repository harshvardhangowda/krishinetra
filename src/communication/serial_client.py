"""
AI-Powered Precision Pest Management - Serial Communication Client
File: src/communication/serial_client.py

Provides serial line communication fallback for Raspberry Pi <-> ESP32 wired interface.
Includes an in-memory loopback mock client for testing without physical UART hardware.
"""

from typing import Optional, List


class MockSerialClient:
    """In-memory mock serial interface for wired UART fallback."""

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.is_open = False
        self.tx_buffer: List[bytes] = []
        self.rx_buffer: List[bytes] = []

    def open(self) -> bool:
        self.is_open = True
        return True

    def close(self) -> None:
        self.is_open = False

    def write_line(self, message: str) -> int:
        if not self.is_open:
            raise IOError("Serial port not open")
        data = (message.strip() + "\n").encode("utf-8")
        self.tx_buffer.append(data)
        return len(data)

    def read_line(self) -> Optional[str]:
        if not self.is_open:
            raise IOError("Serial port not open")
        if not self.rx_buffer:
            return None
        data = self.rx_buffer.pop(0)
        return data.decode("utf-8").strip()

    def feed_rx(self, message: str) -> None:
        """Helper to feed incoming serial line for simulation."""
        self.rx_buffer.append((message.strip() + "\n").encode("utf-8"))
