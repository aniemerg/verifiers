"""Mock vLLM module for macOS development and testing.
Provides a compatible interface with vLLM that works with OpenAI API.
"""
import time
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Union, Sequence

try:
    import tiktoken
except ImportError:
    raise ImportError(
        "The tiktoken package is required for OpenAI API token handling. "
        "Please install it with: pip install tiktoken"
    )

try:
    from openai import OpenAI
    from openai.types.chat import ChatCompletion
except ImportError:
    raise ImportError(
        "The openai package is required for OpenAI API usage. "
        "Please install it with: pip install openai"
    )


class SamplingParams:
    """Mock sampling parameters compatible with vLLM."""
    
    def __init__(self, **kwargs):
        self.temperature = kwargs.get('temperature', 1.0)
        self.top_p = kwargs.get('top_p', 1.0)
        self.max_tokens = kwargs.get('max_tokens', 100)
        self.skip_special_tokens = kwargs.get('skip_special_tokens', False)
        self.spaces_between_special_tokens = kwargs.get('spaces_between_special_tokens', False)
        self.n = kwargs.get('n', 1)
        self.frequency_penalty = kwargs.get('frequency_penalty', 0.0)
        self.presence_penalty = kwargs.get('presence_penalty', 0.0)
        
        # Store any other parameters
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)
    
    def clone(self):
        """Create a copy of the sampling parameters."""
        return SamplingParams(**self.__dict__)


class MockOutput:
    """Mock output object compatible with vLLM output structure."""
    
    def __init__(self, text: str, token_ids: List[int]):
        self.text = text
        self.token_ids = token_ids


class MockResponse:
    """Mock response object compatible with vLLM response structure."""
    
    def __init__(self, prompt_token_ids: List[int], outputs: List[MockOutput]):
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs


class LLM:
    """Mock LLM that uses OpenAI API to provide vLLM-compatible responses."""
    
    def __init__(self, 
                 openai_client: Optional[OpenAI] = None,
                 model: str = "gpt-3.5-turbo",
                 model_name: Optional[str] = None,  # For compatibility with vLLM
                 api_key: Optional[str] = None,
                 tokenizer_model: Optional[str] = None,
                 max_workers: int = 10,
                 retry_attempts: int = 3,
                 retry_delay: float = 1.0,
                 **kwargs):
        """
        Initialize the mock LLM with OpenAI client.
        
        Args:
            openai_client: An existing OpenAI client instance, or None to create a new one
            model: OpenAI model name to use (e.g., "gpt-3.5-turbo", "gpt-4")
            model_name: Alternative parameter name for model (for vLLM compatibility)
            api_key: OpenAI API key (if not provided via client or environment)
            tokenizer_model: Specific model name for tiktoken, or None to derive from model
            max_workers: Maximum number of concurrent threads for batch processing
            retry_attempts: Number of retry attempts for API calls
            retry_delay: Base delay between retry attempts (with exponential backoff)
        """
        # Handle model name parameter variants (vLLM uses model_name, we support both)
        self.model = model_name or model
        
        # Initialize OpenAI client if not provided
        self.client = openai_client or OpenAI(api_key=api_key)
        
        # Determine tokenizer model
        if tokenizer_model is None:
            # Map OpenAI models to tiktoken encoding names
            if 'gpt-4' in self.model:
                tokenizer_model = 'cl100k_base'  # GPT-4 encoding
            elif 'gpt-3.5-turbo' in self.model:
                tokenizer_model = 'cl100k_base'  # GPT-3.5 encoding
            else:
                # Default to cl100k_base for other models
                tokenizer_model = 'cl100k_base'
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_model)
        except KeyError:
            # Fallback to cl100k_base if the specified encoding is not available
            self.tokenizer = tiktoken.get_encoding('cl100k_base')
        
        # Other settings
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Store kwargs for compatibility
        self.dtype = kwargs.get('dtype', 'float16')
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)
    
    def _encode_messages(self, messages: List[Dict[str, str]]) -> List[int]:
        """
        Encode messages to token IDs using tiktoken.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            List of token IDs
        """
        # Convert messages to the OpenAI chat format string
        # This is a simplified implementation that might not match OpenAI's exact encoding
        token_ids = []
        
        # Encode each message
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            # Add role tokens (approximate)
            role_tokens = self.tokenizer.encode(f"<|{role}|>\n")
            token_ids.extend(role_tokens)
            
            # Add content tokens
            content_tokens = self.tokenizer.encode(content)
            token_ids.extend(content_tokens)
            
            # Add delimiter (approximate)
            token_ids.extend(self.tokenizer.encode("\n"))
        
        return token_ids
    
    def _api_call_with_retry(self, messages: List[Dict[str, str]], params: Dict[str, Any]) -> ChatCompletion:
        """
        Make an API call with retry logic.
        
        Args:
            messages: List of message dictionaries
            params: API parameters
            
        Returns:
            OpenAI API response
        """
        last_error = None
        backoff = self.retry_delay
        
        for attempt in range(self.retry_attempts):
            try:
                # Random delay to avoid rate limiting
                time.sleep(backoff * random.random())
                
                # Make the API call
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **params
                )
                return response
                
            except Exception as e:
                last_error = e
                # Exponential backoff with jitter
                backoff *= 2
                continue
        
        # If we reach here, all attempts failed
        raise last_error or RuntimeError("API call failed after multiple attempts")
    
    def _process_chat_request(self, messages: List[Dict[str, str]], sampling_params: SamplingParams) -> MockResponse:
        """
        Process a single chat request.
        
        Args:
            messages: List of message dictionaries
            sampling_params: Sampling parameters
            
        Returns:
            Mock response object
        """
        # Convert sampling params to OpenAI API parameters
        api_params = {
            'temperature': sampling_params.temperature,
            'top_p': sampling_params.top_p,
            'max_tokens': sampling_params.max_tokens,
            'frequency_penalty': getattr(sampling_params, 'frequency_penalty', 0.0),
            'presence_penalty': getattr(sampling_params, 'presence_penalty', 0.0),
            'n': 1,  # Force to 1 since we process the n parameter ourselves
        }
        
        # Get prompt token IDs
        prompt_token_ids = self._encode_messages(messages)
        
        # Make the API call
        response = self._api_call_with_retry(messages, api_params)
        
        # Get the generated text
        completion_text = response.choices[0].message.content or ""
        
        # Convert completion to token IDs
        completion_token_ids = self.tokenizer.encode(completion_text)
        
        # Create mock outputs
        mock_outputs = [MockOutput(completion_text, completion_token_ids)]
        
        # If n > 1 was requested, make additional API calls
        if sampling_params.n > 1:
            n_additional = sampling_params.n - 1
            additional_outputs = []
            
            for _ in range(n_additional):
                # Add jitter to temperature to encourage diversity
                jittered_params = api_params.copy()
                jittered_params['temperature'] = max(0.1, api_params['temperature'] * (1.0 + random.uniform(-0.05, 0.05)))
                
                # Make additional API call
                add_response = self._api_call_with_retry(messages, jittered_params)
                add_text = add_response.choices[0].message.content or ""
                add_token_ids = self.tokenizer.encode(add_text)
                
                additional_outputs.append(MockOutput(add_text, add_token_ids))
            
            mock_outputs.extend(additional_outputs)
        
        # Create and return mock response
        return MockResponse(prompt_token_ids, mock_outputs)
    
    def chat(self, 
             messages_list: List[List[Dict[str, str]]], 
             sampling_params: SamplingParams,
             use_tqdm: bool = False) -> List[MockResponse]:
        """
        Process a batch of chat requests.
        
        Args:
            messages_list: List of message sequences
            sampling_params: Sampling parameters
            use_tqdm: Whether to use tqdm progress bar (ignored in this implementation)
            
        Returns:
            List of mock response objects
        """
        # For single message list, wrap it
        if not messages_list or not isinstance(messages_list[0], list):
            messages_list = [messages_list]  # type: ignore
        
        # Process requests in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            responses = list(executor.map(
                lambda messages: self._process_chat_request(messages, sampling_params),
                messages_list
            ))
        
        return responses
    
    def generate(self, 
                prompt_token_ids: List[List[int]], 
                sampling_params: Optional[SamplingParams] = None,
                **kwargs):
        """
        Generate completions for token sequences (legacy vLLM method).
        
        Args:
            prompt_token_ids: List of token ID sequences
            sampling_params: Sampling parameters
            
        Raises:
            NotImplementedError: This method is not recommended, use chat() instead
        """
        raise NotImplementedError(
            "The generate() method is not fully implemented in this mock. "
            "Please use the chat() method instead, which is what MultiStepEnv uses."
        )
    
    async def achat(self, 
                   messages_list: List[List[Dict[str, str]]], 
                   sampling_params: SamplingParams,
                   **kwargs):
        """
        Async version of chat (for compatibility with vLLM).
        
        Args:
            messages_list: List of message sequences
            sampling_params: Sampling parameters
            
        Returns:
            List of mock response objects
        """
        # For single message list, wrap it
        if not messages_list or not isinstance(messages_list[0], list):
            messages_list = [messages_list]  # type: ignore
        
        # Process requests in parallel using asyncio
        async def process_one(messages):
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: self._process_chat_request(messages, sampling_params)
            )
        
        # Create tasks for all requests
        tasks = [process_one(messages) for messages in messages_list]
        responses = await asyncio.gather(*tasks)
        
        return responses