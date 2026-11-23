"""
KrishiNetra Backend - Structured Logging Configuration
File: backend/app/core/logging.py
"""

import logging
import sys

logger = logging.getLogger("krishinetra")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [krishinetra] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
