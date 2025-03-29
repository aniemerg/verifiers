"""Mock vLLM module for macOS development and testing.
Only provides the minimal interface needed for imports to work.
"""

class SamplingParams:
    """Mock sampling parameters."""
    def __init__(self, **kwargs):
        self.temperature = kwargs.get('temperature', 1.0)
        self.top_p = kwargs.get('top_p', 1.0)
        self.top_k = kwargs.get('top_k', -1)
        self.min_p = kwargs.get('min_p', 0.0)
        self.repetition_penalty = kwargs.get('repetition_penalty', 1.0)
        self.max_tokens = kwargs.get('max_tokens', 100)
        self.n = kwargs.get('n', 1)
        
    def clone(self):
        """Return a copy of the SamplingParams object."""
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_tokens,
            n=self.n
        )

class LLM:
    """Mock LLM that raises NotImplementedError if actually used."""
    def __init__(self, *args, **kwargs):
        self.model = kwargs.get('model', 'mock')
        self.dtype = kwargs.get('dtype', 'float16')
        
    def generate(self, *args, **kwargs):
        raise NotImplementedError(
            "This is a mock vLLM for macOS development. "
            "Install the real vLLM package for actual model inference."
        )
        
    def chat(self, *args, **kwargs):
        raise NotImplementedError(
            "This is a mock vLLM for macOS development. "
            "Install the real vLLM package for actual model inference."
        )

class VLLMClient:
    """Mock VLLMClient for macOS development."""
    def __init__(self, *args, **kwargs):
        pass
        
    def generate(self, prompts, n=1, repetition_penalty=1.0, temperature=1.0, top_p=1.0, 
                top_k=-1, min_p=0.0, max_tokens=16, guided_decoding_regex=None):
        """Mock implementation of generate method."""
        raise NotImplementedError(
            "This is a mock VLLMClient for macOS development. "
            "Install the real vLLM package for actual model inference."
        )
    
    def update_named_param(self, name, weights):
        """Mock implementation of update_named_param method."""
        pass
    
    def reset_prefix_cache(self):
        """Mock implementation of reset_prefix_cache method."""
        pass