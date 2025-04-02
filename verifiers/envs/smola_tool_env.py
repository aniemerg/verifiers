import json
import inspect
from copy import deepcopy
from typing import List, Dict, Any, Callable, Optional

from datasets import Dataset

from verifiers import RewardFunc
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.parsers.smola_parser import SmolaParser
from verifiers.parsers.xml_parser import XMLParser
from verifiers.rubrics.smola_tool_rubric import SmolaToolRubric

def infer_schema_from_function(func: Callable) -> Dict[str, Any]:
    """Infers a tool schema from a function's signature and docstring."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    # Parse docstring sections
    doc_parts = doc.split("\n\n")
    description = doc_parts[0].strip()
    
    # Extract examples if present
    examples = []
    return_description = ""
    for part in doc_parts:
        if part.startswith("Examples:"):
            examples = [line.strip() for line in part.split("\n")[1:] if line.strip()]
        elif part.startswith("Returns:"):
            return_description = part.split("\n")[1].strip()

    return_type = str(sig.return_annotation.__name__ if sig.return_annotation != inspect.Parameter.empty else "any")
    
    # Build args schema
    args = {}
    for name, param in sig.parameters.items():
        param_doc = ""
        for part in doc_parts:
            if part.strip().startswith("Args:"):
                for line in part.split("\n")[1:]:
                    if line.strip().startswith(f"{name}:"):
                        param_doc = line.strip()[len(name)+1:].strip()
        
        args[name] = {
            "type": str(param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "any"),
            "description": param_doc,
        }
        if param.default != inspect.Parameter.empty:
            args[name]["default"] = param.default
    
    return {
        "name": func.__name__,
        "description": description,
        "args": args,
        "returns": return_description + f" ({return_type})",
        "examples": examples
    }

def format_tool_descriptions(schemas: List[Dict[str, Any]]) -> str:
    """Formats tool schemas into a user-friendly description string."""
    descriptions = []
    for schema in schemas:
        desc = [f"{schema['name']}: {schema['description']}"]
        
        desc.append("\nArguments:")
        for arg_name, arg_info in schema['args'].items():
            default = f" (default: {arg_info['default']})" if 'default' in arg_info else ""
            desc.append(f"  - {arg_name}: {arg_info['description']}{default}")
        
        if schema['examples']:
            desc.append("\nExamples:")
            for example in schema['examples']:
                desc.append(f"  {example}")
        
        if schema['returns']:
            desc.append(f"\nReturns: {schema['returns']}")
        
        descriptions.append("\n".join(desc))
    
    return "\n\n".join(descriptions)

class SmolaToolEnv(MultiTurnEnv):
    """
    Environment for SmolaAgents tools that uses the <tool> tag format.
    
    This environment inherits from MultiTurnEnv and is designed to work with
    SmolaAgents tools, handling tool calls with <tool> tags and JSON-formatted tool calls.
    """
    
    def __init__(self,
                 dataset: Dataset | None = None,
                 eval_dataset: Dataset | None = None,
                 tools: List[Callable] = [],
                 system_prompt: str = "",
                 few_shot: List[Dict[str, str]] = [],
                 sampling_args={
                     "stop": ["</tool>\n", "</answer>\n"],
                     "include_stop_str_in_output": True
                 },
                 mask_env_response: bool = True,
                 max_steps: int = 10, 
                 **kwargs):
        """
        Initialize the SmolaToolEnv with tools and appropriate parsers.
        
        Args:
            dataset: Dataset to use for training
            eval_dataset: Dataset to use for evaluation
            tools: List of tool functions to make available
            system_prompt: System prompt to use
            few_shot: Few-shot examples to include
            sampling_args: Arguments for sampling during generation
            mask_env_response: Whether to mask environment responses
            max_steps: Maximum number of steps to take
            **kwargs: Additional arguments to pass to MultiTurnEnv
        """
        # Infer schemas from tool functions
        self.tool_schemas = [infer_schema_from_function(tool) for tool in tools]
        self.tools = {tool.__name__: tool for tool in tools}
        
        # Format the system prompt with tool descriptions if needed
        if "{tool_descriptions}" in system_prompt:
            tool_descriptions = format_tool_descriptions(self.tool_schemas)
            formatted_prompt = system_prompt.format(tool_descriptions=tool_descriptions)
        else:
            formatted_prompt = system_prompt
        
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=formatted_prompt,
            few_shot=few_shot,
            mask_env_response=mask_env_response,
            max_steps=max_steps,
            sampling_args=sampling_args,
            **kwargs
        )
        
        self.dataset_name = dataset
        self.max_steps = max_steps
        self.rubric = SmolaToolRubric(tools=tools)
        
        # Set up parsers for both LLM output and environment response
        self.llm_parser = SmolaParser(fields=["reasoning", "tool", "answer"])
        self.env_parser = XMLParser(fields=["r"])  # SmolaAgents uses <r> tags for results
    
    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        """Get reward functions from the rubric."""
        return self.rubric.get_reward_funcs()
    
    def get_reward_weights(self, **kwargs: Any) -> List[float]:
        """Get reward weights from the rubric."""
        return self.rubric.get_reward_weights()
    
    def _get_step_count(self, messages: List[Dict[str, str]]) -> int:
        """Count the number of tool uses in the message history, excluding few-shot examples."""
        step_count = 0
        
        # Skip messages that are part of few-shot examples
        # We need to determine where the actual conversation starts
        # System message + few-shot examples + user query = start of actual conversation
        conversation_start = 1  # Start after system message
        if self.few_shot:
            # Account for all few-shot messages
            conversation_start += len(self.few_shot)
        
        # Only count tool uses from the actual conversation
        for message in messages[conversation_start:]:
            if message.get("role") == "assistant":
                step_count += 1
        return step_count
    
    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        """
        Determine if the conversation is completed.
        
        A conversation is completed if:
        1. We've reached the maximum number of steps
        2. The last message contains an answer tag
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional arguments
            
        Returns:
            True if the conversation is completed, False otherwise
        """
        try:
            # Check if we've hit max steps by counting tool uses in the message history
            step_count = self._get_step_count(messages)
            if step_count > self.max_steps:
                return True
            
            # Use the LLM parser to check for an answer field
            parsed = self.llm_parser.parse(messages[-1]["content"])
            
            # Check if we got a valid answer field (not just None from failed parsing)
            return hasattr(parsed, 'answer') and parsed.answer is not None
        except Exception:
            return False
    
    def call_tool(self, tool_json: Dict[str, Any], **kwargs: Any) -> str:
        """
        Call a tool based on a parsed tool call.
        
        Args:
            tool_json: Dictionary containing the "name" and "args" for the tool
            **kwargs: Additional arguments
            
        Returns:
            The result of the tool call as a string
        """
        try:
            if not isinstance(tool_json, dict):
                return "Error: Tool command must be a JSON object with 'name' and 'args' fields"
            
            tool_name = tool_json.get("name")
            if not tool_name:
                return "Error: Tool command must specify 'name'"
            
            if tool_name not in self.tools:
                return f"Error: Unknown tool '{tool_name}'. Available tools: {', '.join(self.tools.keys())}"
            
            tool_func = self.tools[tool_name]
            tool_args = tool_json.get("args", {})
            
            if not isinstance(tool_args, dict):
                tool_schema = next((schema['args'] for schema in self.tool_schemas if schema['name'] == tool_name), None)
                return f"Error: Arguments for {tool_name} must be a JSON object with schema {tool_schema}, not a {type(tool_args).__name__}."
            
            # Call the tool function with arguments
            result = tool_func(**tool_args)
            return str(result)
        except Exception as e:
            return f"Error executing tool: {str(e)}"
    
    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        """
        Generate a response from the environment.
        
        This method:
        1. Parses the last message for tool calls
        2. Executes the tool if a valid call is found
        3. Returns the result formatted appropriately
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional arguments
            
        Returns:
            A message dictionary with the environment's response
        """
        try:
            # Extract and parse the tool call
            tool_call = self.llm_parser.parse_tool(messages[-1]["content"])
            
            if tool_call is not None:
                # Call the tool and get the result
                result = self.call_tool(tool_call)
                
                if len(result.strip()) > 0:
                    # Format with <r> tags as used by SmolaAgents
                    formatted_result = self.llm_parser.format_tool_result(result)
                    return {"role": "user", "content": formatted_result}
                else:
                    return {"role": "user", "content": "<r>\nError: Tool execution returned empty output.\n</r>"}
            
            # No valid tool call found
            return {"role": "user", "content": "<r>\nError: No valid tool call found. Tool calls should use <tool> tags with JSON format: {\"name\": \"tool_name\", \"args\": {...}}\n</r>"}
        
        except Exception as e:
            # Handle any exceptions during tool execution
            return {"role": "user", "content": f"<r>\nError: {str(e)}\n</r>"}