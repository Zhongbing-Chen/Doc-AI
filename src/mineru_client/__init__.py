"""
MinerU Client Package
A comprehensive client for MinerU document parsing API with bidirectional coordinate mapping.
"""

from .mineru_api import (
    MinerUClient,
    ModelVersion,
    TaskState,
    TaskResult,
    ExtractProgress
)

__version__ = "1.0.0"
__all__ = [
    "MinerUClient",
    "ModelVersion",
    "TaskState",
    "TaskResult",
    "ExtractProgress"
]
