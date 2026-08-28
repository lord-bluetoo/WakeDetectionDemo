"""Weakly supervised wake-structure learning components."""

from .config import StructureConfig, StructureLossConfig
from .geometry import decode_orientation
from .guidance import StructureGuidedExtractor
from .head import StructureHead, split_structure_logits

__all__ = [
    "StructureConfig",
    "StructureHead",
    "StructureGuidedExtractor",
    "StructureLossConfig",
    "decode_orientation",
    "split_structure_logits",
]
