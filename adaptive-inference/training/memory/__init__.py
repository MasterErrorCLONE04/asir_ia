"""
ASIR Memory Management & Checkpointing Subsystem.
Provides activation checkpointing and memory budget utilities.
"""
from training.memory.checkpoint import apply_gradient_checkpointing

__all__ = ["apply_gradient_checkpointing"]
