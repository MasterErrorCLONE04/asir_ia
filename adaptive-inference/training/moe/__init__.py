"""
ASIR MoE Subsystem.
Provides ReferenceMoELayer (Oracle) and SparseMoEDispatcher.
"""
from training.moe.reference import ReferenceMoELayer, ReferenceFFNExpert
from training.moe.dispatcher import SparseMoEDispatcher

__all__ = ["ReferenceMoELayer", "ReferenceFFNExpert", "SparseMoEDispatcher"]
