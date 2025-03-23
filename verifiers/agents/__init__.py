"""
SmolAgents integration module for the Verifiers framework.

This module implements the integration between Verifiers and SmolAgents,
providing agent classes that work within the Verifiers training system.
"""

from .model_adapter import VerifiersModelAdapter
from .verifiers_agent import VerifiersToolAgent

__all__ = ["VerifiersModelAdapter", "VerifiersToolAgent"]