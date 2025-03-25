"""
Environment for integrating SmolAgents with Verifiers.

This module provides an environment that integrates SmolAgents with Verifiers,
allowing SmolAgents to be used as environments within the Verifiers framework,
particularly for GRPO training.
"""

import sys
import os
import copy
import logging
from typing import Any, Dict, List, Sequence, Optional, Union, Callable, Tuple

# Setup logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add smolagents src directory to path
logger.debug("Adding SmolAgents src directory to path")
sys.path.insert(0, '/Users/allanniemerg/dev/verifiers/wip/smolagents/src')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../wip/smolagents/src')))

# Import SmolAgents classes
try:
    # Import from correct path based on actual structure
    from smolagents.tools import Tool
    logger.debug("Successfully imported Tool class from smolagents.tools")
except ImportError as e:
    error_msg = f"SmolAgents not found. Error: {str(e)}. Python path: {sys.path}"
    logger.error(error_msg)
    raise ImportError(error_msg)

# Import Verifiers classes
from verifiers.envs.multistep_env import MultiStepEnv
from verifiers.parsers.xml_parser import XMLParser
from verifiers.envs.smola_tool_env import SmolaToolEnv

# Import our agent classes
from .verifiers_agent import VerifiersToolAgent
from .model_adapter import VerifiersModelAdapter
from .agent_parser import VerifiersAgentParser


class SmolAgentEnv(MultiStepEnv):
    """
    Environment that wraps SmolAgents for GRPO training.
    
    This environment allows SmolAgents to be used within Verifiers,
    particularly for GRPO training. It handles batched processing
    and multiple parallel agents.
    """
    
    def get_rubric(self, **kwargs: Any) -> List[Any]:
        """
        Get the default reward functions for this environment.
        
        Args:
            **kwargs: Additional arguments.
            
        Returns:
            List of reward functions.
        """
        # Define a simple reward function that checks for answer tags
        def answer_presence(prompts, completions, **kwargs):
            import re
            rewards = []
            
            for completion in completions:
                # Look for answer tags in the completion
                has_answer = False
                for message in completion:
                    if message["role"] == "assistant":
                        if re.search(r"<answer>(.*?)</answer>", message["content"], re.DOTALL):
                            has_answer = True
                            break
                
                # Give reward if an answer is present
                if has_answer:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            
            return rewards
        
        # Return a list of reward functions
        return [answer_presence]
    
    def get_dataset(self, **kwargs: Any) -> Any:
        """
        Get a default dataset for this environment.
        
        This is a required abstract method from MultiStepEnv.
        For actual training, you should provide your own dataset.
        
        Args:
            **kwargs: Additional arguments.
            
        Returns:
            A simple dataset (list of prompts).
        """
        # Try to load the GSM8K dataset if available
        try:
            from datasets import load_dataset
            dataset = load_dataset("gsm8k", "main", split="train")
            # Just return a small subset
            return dataset.select(range(min(100, len(dataset))))
        except Exception as e:
            logger.warning(f"Could not load GSM8K dataset: {e}")
            
            # Return a minimal dataset if loading fails
            return [
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2+2?"}
                ],
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 3*4?"}
                ]
            ]
    
    def __init__(
        self, 
        tools: List[Tool] = None,
        model_adapter_factory: Callable[[], VerifiersModelAdapter] = None,
        agent_factory: Callable[[List[Tool], VerifiersModelAdapter], VerifiersToolAgent] = None,
        max_steps: int = 10,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the SmolAgentEnv.
        
        Args:
            tools: List of SmolAgents Tool instances.
            model_adapter_factory: Factory function to create model adapters.
            agent_factory: Factory function to create agent instances.
            max_steps: Maximum number of steps per interaction.
            system_prompt: Optional system prompt to override default.
            **kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(**kwargs)
        
        self.tools = tools or []
        self.model_adapter_factory = model_adapter_factory or (lambda: VerifiersModelAdapter())
        self.agent_factory = agent_factory or self._default_agent_factory
        self.max_steps = max_steps
        self.system_prompt = system_prompt
        
        # Create a parser for the agent's output
        self.parser = XMLParser(fields=["reasoning", "tool_call", "answer"])
        
        # Store agents by state ID for batched processing
        self.agent_pool = {}
        
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def _default_agent_factory(self, tools: List[Tool], model: VerifiersModelAdapter) -> VerifiersToolAgent:
        """
        Default factory function for creating agent instances.
        
        Args:
            tools: List of tools to provide to the agent.
            model: Model adapter to use for the agent.
            
        Returns:
            A VerifiersToolAgent instance.
        """
        return VerifiersToolAgent(
            tools=tools,
            model=model,
            max_steps=self.max_steps
        )
    
    def _get_or_create_agent(self, state_id: int) -> VerifiersToolAgent:
        """
        Get an existing agent or create a new one.
        
        Args:
            state_id: Unique identifier for the state.
            
        Returns:
            A VerifiersToolAgent instance.
        """
        if state_id not in self.agent_pool:
            # Create a new model adapter
            model_adapter = self.model_adapter_factory()
            
            # Create a new agent
            self.agent_pool[state_id] = self.agent_factory(self.tools, model_adapter)
            
        return self.agent_pool[state_id]
    
    def is_completed(self, messages: List[Dict[str, str]]) -> bool:
        """
        Check if the conversation is completed.
        
        A conversation is completed if the last assistant message contains
        a final answer.
        
        Args:
            messages: List of message dictionaries.
            
        Returns:
            True if the conversation is completed, False otherwise.
        """
        if not messages:
            return False
            
        # Check the last assistant message for a final answer
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        if not assistant_messages:
            return False
            
        last_message = assistant_messages[-1]["content"]
        parsed = self.parser.parse(last_message)
        
        # If the message has an answer tag, it's completed
        return hasattr(parsed, 'answer') and parsed.answer is not None
    
    def env_response(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Generate an environment response for the given messages.
        
        This method processes the last assistant message, extracts any tool calls,
        executes the tools, and returns the result.
        
        Args:
            messages: List of message dictionaries.
            
        Returns:
            A message dictionary with the environment's response.
        """
        if not messages:
            return {"role": "user", "content": "Please start the conversation."}
            
        # Get the last assistant message
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        if not assistant_messages:
            return {"role": "user", "content": "Please respond to my question."}
            
        last_message = assistant_messages[-1]["content"]
        
        # Parse the message to extract tool calls
        parsed = self.parser.parse(last_message)
        
        # Check for a tool call
        if hasattr(parsed, 'tool_call') and parsed.tool_call:
            try:
                # Extract the tool call
                tool_call = parsed.tool_call
                
                # Find the tool by name
                tool_name = tool_call.get('name', '')
                tool = next((t for t in self.tools if t.name == tool_name), None)
                
                if not tool:
                    return {"role": "user", "content": f"Error: Tool '{tool_name}' not found. Available tools: {', '.join(t.name for t in self.tools)}"}
                
                # Extract arguments
                args = tool_call.get('args', {})
                
                # Execute the tool
                result = tool(**args)
                
                # Return the result
                return {"role": "user", "content": f"<observation>\n{result}\n</observation>"}
                
            except Exception as e:
                # If there's an error executing the tool, return an error message
                return {"role": "user", "content": f"<observation>\nError executing tool: {str(e)}\n</observation>"}
        
        # If there's no tool call, check if there's a final answer
        if hasattr(parsed, 'answer') and parsed.answer:
            # The conversation is completed, no need for another response
            return {"role": "user", "content": "Conversation completed."}
        
        # If we can't identify a tool call or answer, prompt the agent to follow the format
        return {"role": "user", "content": """Please follow the expected format:
1. <reasoning>Your reasoning here</reasoning>
2. <tool_call>{"name": "tool_name", "args": {"arg1": "value1"}}</tool_call>
Or provide a final answer:
<answer>Your final answer here</answer>"""}
    
    def step(self, states: List[Dict[str, Any]], llm: Any, sampling_params: Any) -> List[Dict[str, Any]]:
        """
        Process one step for each state.
        
        Args:
            states: List of state dictionaries.
            llm: LLM instance to use for generation.
            sampling_params: Sampling parameters for generation.
            
        Returns:
            Updated list of state dictionaries.
        """
        # Identify live states (not completed)
        live_indices = [i for i, s in enumerate(states) if not s["completed"]]
        
        if not live_indices:
            return states
            
        # Get messages for each live state
        messages_to_step = [states[i]["messages"] for i in live_indices]
        
        # For each live state, update the agent and process the step
        for j, idx in enumerate(live_indices):
            state = states[idx]
            state_id = id(state)
            state_messages = state["messages"]
            
            # Get or create an agent for this state
            agent = self._get_or_create_agent(state_id)
            
            # Update the agent's model with the current LLM
            agent.update_llm(llm, sampling_params)
            
            # Call the LLM to get a response
            llm_responses = llm.chat([state_messages], sampling_params=sampling_params, use_tqdm=False)
            llm_response = llm_responses[0]
            
            # Update state with the response
            state_copy = copy.deepcopy(state)
            
            # Save prompt IDs if this is the first step
            if len(state_copy["prompt_ids"]) == 0:
                state_copy["prompt_ids"] = llm_response.prompt_token_ids
                
            # Add assistant message with the LLM response
            assistant_message = {"role": "assistant", "content": llm_response.outputs[0].text}
            state_copy["messages"].append(assistant_message)
            
            # Track completion IDs and mask
            assistant_token_ids = llm_response.outputs[0].token_ids
            state_copy["completion_ids"].extend(assistant_token_ids)
            state_copy["completion_mask"].extend([1] * len(assistant_token_ids))
            
            # Check if the conversation is completed
            if self.is_completed(state_copy["messages"]):
                state_copy["completed"] = True
            else:
                # Add environment response if not completed
                env_response = self.env_response(state_copy["messages"])
                state_copy["messages"].append(env_response)
                
                # Track completion IDs and mask for environment response (but don't include in training)
                # This should be adjusted based on actual token ID handling in the environment
                state_copy["completion_mask"].extend([0] * 10)  # Placeholder, not used for training
            
            # Update the state
            states[idx] = state_copy
            
        # Clean up completed agents
        for i, state in enumerate(states):
            if state["completed"]:
                state_id = id(state)
                if state_id in self.agent_pool:
                    del self.agent_pool[state_id]
        
        return states
    
    def reset(self):
        """Reset the environment, clearing agent pool."""
        self.agent_pool = {}
    
    def get_agent_metrics(self, states: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """
        Get metrics for each state's agent.
        
        Args:
            states: List of state dictionaries.
            
        Returns:
            List of metric dictionaries, one per state.
        """
        metrics = []
        
        for state in states:
            state_id = id(state)
            
            if state_id in self.agent_pool:
                agent = self.agent_pool[state_id]
                metrics.append(agent.get_metrics_for_grpo())
            else:
                # Default metrics for completed or uninitialized agents
                metrics.append({
                    "tool_success_rate": 0.0,
                    "reasoning_quality": 0.0,
                    "steps_efficiency": 0.0,
                    "answer_found": 1.0 if state.get("completed", False) else 0.0
                })
        
        return metrics