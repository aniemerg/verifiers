"""
Tool-calling agent designed for the Verifiers GRPO training system.

This module implements a tool-calling agent that extends SmolAgents'
ToolCallingAgent class, adding specific functionality for working with
Verifiers' GRPO training system.
"""

import sys
import json
import logging
import time
import copy
from typing import Any, Dict, List, Optional, Union, Callable

# Add smolagents to path if needed
sys.path.append('/Users/allanniemerg/dev/verifiers/wip/smolagents')

# Import SmolAgents classes
try:
    from smolagents.agents import ToolCallingAgent, AgentError, AgentParsingError, AgentExecutionError
    from smolagents.types import ChatMessage, ActionStep, ToolCall
    from smolagents.types import PromptTemplates, LogLevel
    from smolagents.tools.tool import Tool
except ImportError:
    raise ImportError("SmolAgents not found. Make sure it's installed and in the Python path.")

from .agent_parser import VerifiersAgentParser
from .model_adapter import VerifiersModelAdapter


class VerifiersToolAgent(ToolCallingAgent):
    """
    Tool-calling agent designed for the Verifiers GRPO training system.
    
    This agent extends SmolAgents' ToolCallingAgent class, adding specific
    functionality for working with Verifiers' XML-based tool calling format
    and GRPO training system.
    """
    
    def __init__(
        self,
        tools: List[Tool],
        model: Union[VerifiersModelAdapter, Callable[[List[Dict[str, str]]], ChatMessage]],
        prompt_templates: Optional[PromptTemplates] = None,
        max_steps: int = 20,
        **kwargs
    ):
        """
        Initialize the agent with tools and model.
        
        Args:
            tools: List of tools available to the agent.
            model: Model adapter or callable that returns a ChatMessage.
            prompt_templates: Optional custom prompt templates.
            max_steps: Maximum number of steps to take.
            **kwargs: Additional arguments to pass to the parent class.
        """
        # Create Verifiers-specific prompt templates if not provided
        verifiers_templates = prompt_templates or self._create_verifiers_templates()
        
        # Initialize the parent class
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=verifiers_templates,
            max_steps=max_steps,
            **kwargs
        )
        
        # Create a parser for extracting structured content
        self.parser = VerifiersAgentParser(fields=["reasoning", "tool_call", "answer"])
        
        # Initialize metrics tracking
        self.metrics = {
            "tool_calls": [],
            "tool_success_rate": 0.0,
            "reasoning_quality": 0.0,
            "steps_efficiency": 1.0,
            "answer_found": False
        }
    
    def _create_verifiers_templates(self) -> Dict[str, str]:
        """
        Create prompt templates that incorporate Verifiers' XML structure.
        
        Returns:
            Dictionary of template strings.
        """
        # Build up a tool description block for the system prompt
        tool_descriptions = []
        for tool in self.tools.values():
            # Extract parameter info
            params = []
            for name, schema in tool.inputs.items():
                desc = schema.get("description", "")
                required = "required" if schema.get("required", False) else "optional"
                param_type = schema.get("type", "string")
                params.append(f"  - {name}: {desc} ({param_type}, {required})")
            
            param_text = "\n".join(params) if params else "  None"
            
            # Create tool description
            tool_descriptions.append(f"""
Tool: {tool.name}
Description: {tool.description}
Parameters:
{param_text}
Returns: {tool.output_type}
""")
        
        tool_descriptions_text = "\n".join(tool_descriptions)
        
        # Create template with Verifiers XML structure
        templates = {
            "system_prompt": f"""You are an expert assistant who can solve any task using tool calls.

Available Tools:
{tool_descriptions_text}

For each step, you should follow this structure:

1. <reasoning>
   Think through the problem step by step. Explain your thought process in detail.
   </reasoning>

2. <tool_call>
   {{
     "name": "tool_name",
     "args": {{"param1": "value1", "param2": "value2"}}
   }}
   </tool_call>

After each tool call, you will receive a result and can continue with the next step.

When you have the final answer, use:

<answer>Your final answer here</answer>
""",
            "error_prompt": """I encountered an error when executing the tool:
{error}

Please try again, focusing on:
1. Using a valid tool name from the available tools
2. Providing the correct parameters for the tool
3. Formatting your tool call correctly with the <tool_call> tag

Let me try again:

<reasoning>
[Your updated reasoning here]
</reasoning>

<tool_call>
{
  "name": "valid_tool_name",
  "args": {"valid_param": "valid_value"}
}
</tool_call>
""",
            "final_answer_template": """To provide my final answer to the question, I will use:

<answer>
{answer}
</answer>
"""
        }
        
        return templates
    
    def update_llm(self, llm, sampling_params):
        """
        Update the LLM instance used by this agent.
        
        Called during GRPO training steps to update the model.
        
        Args:
            llm: The Verifiers LLM instance.
            sampling_params: Sampling parameters to use.
        """
        if hasattr(self.model, 'update_llm') and isinstance(self.model, VerifiersModelAdapter):
            self.model.update_llm(llm, sampling_params)
    
    def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """
        Process one step with graceful handling of model responses.
        
        Args:
            memory_step: The memory step to process.
            
        Returns:
            The result of the step, or None to continue.
        """
        # Get model input from memory
        memory_messages = self.write_memory_to_messages()
        memory_step.model_input_messages = memory_messages
        
        # Get model output
        model_message = self.model(memory_messages)
        memory_step.model_output_message = model_message
        model_output = model_message.content
        
        # Parse the output, gracefully handling non-compliance with structure
        parsed = self.parser.parse(model_output)
        
        # Extract reasoning if available
        if hasattr(parsed, 'reasoning') and parsed.reasoning:
            memory_step.reasoning = parsed.reasoning
            self.logger.log_markdown(content=parsed.reasoning, title="Reasoning", level=LogLevel.INFO)
            
            # Update reasoning quality metric (simple length-based heuristic for now)
            reasoning_length = len(parsed.reasoning.split())
            self.metrics["reasoning_quality"] = min(1.0, reasoning_length / 50)
        
        # Check for a final answer
        if hasattr(parsed, 'answer') and parsed.answer:
            memory_step.action_output = parsed.answer
            self.metrics["answer_found"] = True
            return parsed.answer
        
        # Check for a tool call
        if hasattr(parsed, 'tool_call') and parsed.tool_call:
            try:
                # Validate the tool call
                tool_call = parsed.tool_call
                
                if not isinstance(tool_call, dict) or 'name' not in tool_call:
                    raise AgentParsingError(
                        f"Invalid tool call format: {tool_call}. Must be a dict with 'name' field.",
                        self.logger
                    )
                
                # Extract tool name and args
                tool_name = tool_call['name']
                tool_args = tool_call.get('args', {})
                
                # Create ToolCall object
                memory_step.tool_calls = [ToolCall(
                    name=tool_name,
                    arguments=tool_args,
                    id=f"call_{len(self.memory.steps)}"
                )]
                
                # Log the tool call
                self.logger.log_markdown(
                    content=f"Tool: {tool_name}\nArgs: {json.dumps(tool_args, indent=2)}",
                    title="Tool Call",
                    level=LogLevel.INFO
                )
                
                # Track tool call in metrics
                self.metrics["tool_calls"].append({
                    "name": tool_name,
                    "step": len(self.memory.steps)
                })
                
                # Execute the tool
                try:
                    result = self.execute_tool_call(tool_name, tool_args)
                    
                    # Update success metrics
                    success_count = sum(1 for tc in self.metrics["tool_calls"] if "error" not in tc)
                    total_calls = len(self.metrics["tool_calls"])
                    self.metrics["tool_success_rate"] = success_count / max(1, total_calls)
                    
                    # Update steps efficiency
                    self.metrics["steps_efficiency"] = 1.0 / max(1, len(self.memory.steps))
                    
                    return result
                except Exception as e:
                    # Track the error in metrics
                    self.metrics["tool_calls"][-1]["error"] = str(e)
                    
                    # Update success metrics
                    success_count = sum(1 for tc in self.metrics["tool_calls"] if "error" not in tc)
                    total_calls = len(self.metrics["tool_calls"])
                    self.metrics["tool_success_rate"] = success_count / max(1, total_calls)
                    
                    # Re-raise the error
                    raise
                
            except json.JSONDecodeError as e:
                # If JSON parsing fails, log but continue
                self.logger.log_markdown(
                    content=f"Failed to parse tool call: {e}\nTool call text: {parsed.tool_call}",
                    title="Parsing Error",
                    level=LogLevel.WARNING
                )
        
        # If we reach here without a tool call or answer, try to detect implicit actions
        detected_tool = self._detect_implicit_tool_call(model_output)
        if detected_tool:
            memory_step.tool_calls = [detected_tool]
            tool_name = detected_tool.name
            tool_args = detected_tool.arguments
            
            # Track in metrics
            self.metrics["tool_calls"].append({
                "name": tool_name,
                "step": len(self.memory.steps),
                "implicit": True
            })
            
            # Log the implicit tool call
            self.logger.log_markdown(
                content=f"Implicit Tool: {tool_name}\nArgs: {json.dumps(tool_args, indent=2)}",
                title="Implicit Tool Call",
                level=LogLevel.INFO
            )
            
            try:
                result = self.execute_tool_call(tool_name, tool_args)
                
                # Update success metrics
                success_count = sum(1 for tc in self.metrics["tool_calls"] if "error" not in tc)
                total_calls = len(self.metrics["tool_calls"])
                self.metrics["tool_success_rate"] = success_count / max(1, total_calls)
                
                return result
            except Exception as e:
                # Track the error in metrics
                self.metrics["tool_calls"][-1]["error"] = str(e)
                
                # Update success metrics
                success_count = sum(1 for tc in self.metrics["tool_calls"] if "error" not in tc)
                total_calls = len(self.metrics["tool_calls"])
                self.metrics["tool_success_rate"] = success_count / max(1, total_calls)
                
                # Re-raise the error
                raise
        
        # If nothing else worked, return None to continue
        return None
    
    def _detect_implicit_tool_call(self, text: str) -> Optional[ToolCall]:
        """
        Try to detect tool calls that don't follow the XML structure.
        
        This is crucial during early training when models don't follow instructions.
        
        Args:
            text: The text to analyze.
            
        Returns:
            A ToolCall object if a tool call was detected, otherwise None.
        """
        # Look for patterns like "I'll use the calculator tool to compute..."
        for tool_name in self.tools.keys():
            # Different patterns that might indicate a tool call
            patterns = [
                rf"(?:use|using|call|calling|execute|executing)\s+(?:the\s+)?(?:tool\s+)?['\"]?{tool_name}['\"]?",
                rf"(?:with|using)\s+(?:the\s+)?{tool_name}(?:\s+tool)?",
                rf"{tool_name}\s+(?:can|could|should|will|would)\s+(?:be used|help)",
                rf"(?:I will|I'll|I can|I should)\s+(?:use|try)\s+{tool_name}"
            ]
            
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    # Try to extract arguments
                    args = {}
                    
                    # Look for parameter values
                    tool = self.tools[tool_name]
                    for param_name in tool.inputs.keys():
                        # Try different patterns for parameter values
                        param_patterns = [
                            rf"{param_name}\s*(?:=|:|is|as|->)\s*(?:\"([^\"]*)\"|'([^']*)'|([^,.\s\)]+))",
                            rf"(?:set|use|with|using)\s+{param_name}\s+(?:as|to|=|:)\s+(?:\"([^\"]*)\"|'([^']*)'|([^,.\s\)]+))",
                            rf"(?:the|for)\s+{param_name}\s+(?:is|of|=|:)\s+(?:\"([^\"]*)\"|'([^']*)'|([^,.\s\)]+))"
                        ]
                        
                        for param_pattern in param_patterns:
                            match = re.search(param_pattern, text, re.IGNORECASE)
                            if match:
                                # Take the first non-None group as the value
                                value = next((g for g in match.groups() if g is not None), "")
                                args[param_name] = value
                                break
                    
                    # Create a ToolCall
                    return ToolCall(
                        name=tool_name,
                        arguments=args,
                        id=f"implicit_call_{len(self.memory.steps)}"
                    )
        
        return None
    
    def clone(self) -> 'VerifiersToolAgent':
        """
        Create a copy of this agent with separate state for batch processing.
        
        Returns:
            A new VerifiersToolAgent instance with the same configuration.
        """
        # Create a new agent with the same tools and model
        clone = VerifiersToolAgent(
            tools=list(self.tools.values()),
            model=self.model,  # The model adapter can be shared
            prompt_templates=copy.deepcopy(self.prompt_templates),
            max_steps=self.max_steps
        )
        
        return clone
    
    def get_metrics_for_grpo(self) -> Dict[str, float]:
        """
        Extract metrics in a format suitable for GRPO reward computation.
        
        Returns:
            A dictionary of metrics for GRPO.
        """
        metrics = {
            "tool_success_rate": self.metrics.get("tool_success_rate", 0.0),
            "reasoning_quality": self.metrics.get("reasoning_quality", 0.0),
            "steps_efficiency": self.metrics.get("steps_efficiency", 0.0),
            "answer_found": 1.0 if self.metrics.get("answer_found", False) else 0.0,
        }
        
        # Add more complex metrics if needed
        
        return metrics