"""
Model adapter for integrating Verifiers LLMs with SmolAgents.

This module provides a model adapter that bridges Verifiers' LLM interface
with SmolAgents' model interface, allowing SmolAgents to work with the
dynamic LLM instances used in Verifiers' GRPO training.
"""

from typing import Any, Dict, List, Optional, Union
import json
import logging
import sys
import copy

# Add smolagents to path if needed
sys.path.append('/Users/allanniemerg/dev/verifiers/wip/smolagents')

# Import SmolAgents classes
try:
    from smolagents.types import ChatMessage, ChatMessageToolCall, ChatMessageToolCallDefinition
    from smolagents.models.model import Model
    from smolagents.tools.tool import Tool
except ImportError:
    raise ImportError("SmolAgents not found. Make sure it's installed and in the Python path.")


class VerifiersModelAdapter(Model):
    """
    An adapter that bridges Verifiers' LLM interface with SmolAgents' model interface.
    
    This adapter allows dynamically updating the LLM instance during GRPO training
    while maintaining compatibility with SmolAgents' model calling convention.
    """
    
    def __init__(
        self,
        flatten_messages_as_text: bool = False,
        tool_name_key: str = "name",
        tool_arguments_key: str = "arguments",
        **kwargs
    ):
        """
        Initialize the model adapter.
        
        Args:
            flatten_messages_as_text: Whether to flatten messages as text.
            tool_name_key: The key to use for tool names.
            tool_arguments_key: The key to use for tool arguments.
            **kwargs: Additional arguments to pass to the model.
        """
        super().__init__(
            flatten_messages_as_text=flatten_messages_as_text,
            tool_name_key=tool_name_key,
            tool_arguments_key=tool_arguments_key,
            **kwargs
        )
        self.llm = None
        self.sampling_params = None
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def update_llm(self, llm, sampling_params):
        """
        Update the LLM instance and sampling parameters.
        
        Args:
            llm: The Verifiers LLM instance.
            sampling_params: The sampling parameters to use.
        """
        self.llm = llm
        self.sampling_params = sampling_params
    
    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs
    ) -> ChatMessage:
        """
        Call the LLM with the given messages and return a ChatMessage.
        
        Args:
            messages: List of message dictionaries in the SmolAgents format.
            stop_sequences: Optional list of sequences to stop generation.
            grammar: Optional grammar to constrain generation.
            tools_to_call_from: Optional list of tools to include in the request.
            **kwargs: Additional arguments to pass to the LLM.
            
        Returns:
            A ChatMessage object containing the model's response.
            
        Raises:
            RuntimeError: If the LLM has not been set.
        """
        if self.llm is None:
            raise RuntimeError("LLM not set. Call update_llm() first.")
        
        # Create a copy of sampling params to avoid modifying the original
        sampling_params = copy.deepcopy(self.sampling_params)
        
        # Apply stop sequences if provided
        if stop_sequences:
            sampling_params.stop = stop_sequences
        
        # Note: Unlike OpenAI, we don't need to pass tools in a special format
        # Tool descriptions are included in the system prompt, following Verifiers approach
        
        # Call the LLM
        try:
            # vLLM expects a list of message lists for batch processing
            # For a single request, we wrap our messages in a list
            llm_responses = self.llm.chat([messages], sampling_params=sampling_params, use_tqdm=False)
            
            # Extract the response (first and only element since we're not batching)
            llm_response = llm_responses[0]
            
            # Extract content
            content = llm_response.outputs[0].text
            
            # Track token counts if available
            if hasattr(llm_response, 'prompt_token_ids'):
                self.last_input_token_count = len(llm_response.prompt_token_ids)
            if hasattr(llm_response, 'outputs') and hasattr(llm_response.outputs[0], 'token_ids'):
                self.last_output_token_count = len(llm_response.outputs[0].token_ids)
            
            # Create a ChatMessage without tool calls - we'll parse those from the content
            # The VerifiersToolAgent will handle parsing tool calls from XML tags
            return ChatMessage(
                role="assistant",
                content=content,
                tool_calls=None,  # XML parsing is done by the agent
                raw=llm_response  # Store the raw response
            )
        
        except Exception as e:
            self._logger.error(f"Error calling LLM: {e}")
            # Return a simple error message
            return ChatMessage(
                role="assistant",
                content=f"Error: {str(e)}"
            )
    
    def to_dict(self) -> Dict:
        """
        Convert the model to a dictionary for serialization.
        
        Returns:
            A dictionary representation of the model.
        """
        return {
            "model_type": "VerifiersModelAdapter",
            "flatten_messages_as_text": self.flatten_messages_as_text,
            "tool_name_key": self.tool_name_key,
            "tool_arguments_key": self.tool_arguments_key,
            **self.kwargs
        }
    
    @classmethod
    def from_dict(cls, model_dictionary: Dict[str, Any]) -> "VerifiersModelAdapter":
        """
        Create a model from a dictionary.
        
        Args:
            model_dictionary: A dictionary representation of the model.
            
        Returns:
            A VerifiersModelAdapter instance.
        """
        # Remove model_type from dictionary
        model_dict_copy = model_dictionary.copy()
        model_dict_copy.pop("model_type", None)
        
        return cls(**model_dict_copy)