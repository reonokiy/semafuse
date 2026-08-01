"""Adapters for the two public benchmarks used by LazyMem paper v2."""

from .locomo import LoCoMoLicenseWarning, load_locomo
from .longmemeval import load_longmemeval

__all__ = ["LoCoMoLicenseWarning", "load_locomo", "load_longmemeval"]
