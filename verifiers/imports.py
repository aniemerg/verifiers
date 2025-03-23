"""Central import handling for platform-specific dependencies and common components."""
import os
import platform
from typing import Any, Dict, List, Optional, Union

# Check if we're on macOS (Darwin)
IS_MACOS = platform.system() == 'Darwin'

# Use mock vLLM on macOS, real vLLM otherwise
if IS_MACOS:
    from .mock_vllm import LLM, SamplingParams
else:
    from vllm import LLM, SamplingParams  # type: ignore

# Dataset imports
try:
    import datasets
    from datasets import load_dataset
except ImportError:
    # Mock load_dataset function for offline mode
    def load_dataset(*args, **kwargs):
        raise ImportError("datasets package not installed. Install with `pip install datasets`.")

# SmolAgents integration imports
try:
    from .agents.model_adapter import VerifiersModelAdapter
    from .agents.verifiers_agent import VerifiersToolAgent
    from .agents.smol_agent_env import SmolAgentEnv
    from .agents.tool_adapter import (
        verifiers_tool_to_smol_tool, 
        create_calculator_tool,
        create_search_tool
    )
    
    HAS_SMOLAGENTS = True
except ImportError:
    HAS_SMOLAGENTS = False


def vllm_llm_factory(model_name: str = None, **kwargs) -> LLM:
    """
    Create a vLLM instance with sensible defaults.
    
    Args:
        model_name: Name of the model to use. If None, uses the model name from
                   the VERIFIERS_MODEL environment variable, or falls back to
                   "gpt2" for testing.
        **kwargs: Additional arguments to pass to the LLM constructor.
        
    Returns:
        A vLLM LLM instance.
    """
    # Get model name from environment variable if not provided
    if model_name is None:
        model_name = os.environ.get("VERIFIERS_MODEL", "gpt2")
    
    # Create and return the LLM
    return LLM(model=model_name, **kwargs)


__all__ = [
    # Core imports
    'LLM', 
    'SamplingParams', 
    'IS_MACOS',
    'load_dataset',
    'vllm_llm_factory',
    
    # SmolAgents integration (if available)
    'HAS_SMOLAGENTS'
]

# Add SmolAgents components to __all__ if available
if HAS_SMOLAGENTS:
    __all__.extend([
        'VerifiersModelAdapter',
        'VerifiersToolAgent',
        'SmolAgentEnv',
        'verifiers_tool_to_smol_tool',
        'create_calculator_tool',
        'create_search_tool'
    ])