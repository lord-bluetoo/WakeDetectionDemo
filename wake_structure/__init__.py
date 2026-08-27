"""Weakly supervised wake-structure learning components."""

from .config import StructureConfig, StructureLossConfig
from .geometry import decode_orientation
from .head import StructureHead, split_structure_logits

__all__ = [
    "StructureConfig",
    "StructureHead",
    "StructureLossConfig",
    "decode_orientation",
    "split_structure_logits",
]

