"""Central import handling for platform-specific dependencies."""
import platform

# Check if we're on macOS (Darwin)
IS_MACOS = platform.system() == 'Darwin'

# Use mock vLLM on macOS, real vLLM otherwise
if IS_MACOS:
    from .mock_vllm import LLM, SamplingParams, VLLMClient
else:
    from vllm import LLM, SamplingParams  # type: ignore
    from trl.extras.vllm_client import VLLMClient  # type: ignore

__all__ = ['LLM', 'SamplingParams', 'VLLMClient', 'IS_MACOS'] 