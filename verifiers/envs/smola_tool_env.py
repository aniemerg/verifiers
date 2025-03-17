import json
from typing import List, Dict, Any, Optional

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.multistep_env import MultiStepEnv
from verifiers.parsers.smola_parser import SmolaParser
from verifiers.prompts.smola_templates import SMOLA_TOOL_PROMPT_TEMPLATE
from verifiers.rubrics.smola_tool_rubric import SmolaToolRubric
from verifiers.utils import preprocess_dataset

class SmolaToolEnv(MultiStepEnv):
    """
    Environment for using SmolaAgents tools with Verifiers.
    
    This class provides integration between SmolaAgents Tool objects
    and the Verifiers framework, allowing models to use SmolaAgents tools
    within the Verifiers evaluation and training pipeline.
    """
    
    def __init__(self,
                 dataset: str = "gsm8k",
                 tools: List = [],  # List of SmolaAgents Tool objects
                 system_prompt: str = SMOLA_TOOL_PROMPT_TEMPLATE,
                 few_shot: List[Dict[str, str]] = [],
                 sampling_args={
                     "stop": ["</tool_call>", "</answer>"],
                     "include_stop_str_in_output": True
                 },
                 mask_env_response: bool = True,
                 max_steps: int = 10, **kwargs):
        """
        Initialize a SmolaToolEnv.
        
        Args:
            dataset: Name of the dataset to use
            tools: List of SmolaAgents Tool objects
            system_prompt: System prompt template
            few_shot: List of few-shot examples
            sampling_args: Arguments for sampling from the model
            mask_env_response: Whether to mask the environment response
            max_steps: Maximum number of steps to take
        """
        # Store the tools by name
        self.tools = {tool.name: tool for tool in tools}
        
        # Format the system prompt with tool descriptions
        tool_descriptions = self._format_tool_descriptions(tools)
        formatted_prompt = system_prompt.format(tool_descriptions=tool_descriptions)
        
        super().__init__(
            system_prompt=formatted_prompt,
            few_shot=few_shot,
            mask_env_response=mask_env_response,
            sampling_args=sampling_args,
            **kwargs
        )
        self.dataset_name = dataset
        self.dataset = preprocess_dataset(
            dataset_name=dataset,
            split="train",
            system_prompt=formatted_prompt,
            few_shot=few_shot
        )
        self.eval_dataset = None
        self.max_steps = max_steps
        self.rubric = SmolaToolRubric()
        # Update parser to recognize both tool and tool_call tags
        self.llm_parser = SmolaParser(fields=["reasoning", ("tool", "tool_call"), "answer"])
        self.env_parser = SmolaParser(fields=["result"])
        
        # Debug: Print few-shot examples
        if few_shot:
            print("\n===== FEW-SHOT EXAMPLES =====")
            for i, example in enumerate(few_shot):
                print(f"Example {i}:")
                for msg in example:
                    print(msg)
            print("===== END FEW-SHOT EXAMPLES =====\n")
    
    def _format_tool_descriptions(self, tools) -> str:
        """
        Format tool descriptions for inclusion in the system prompt.
        
        Args:
            tools: List of SmolaAgents Tool objects
            
        Returns:
            Formatted string describing the tools
        """
        descriptions = []
        
        for tool in tools:
            # Start with tool name and description
            desc = [f"{tool.name}: {tool.description}"]
            
            # Add arguments section
            desc.append("\nArguments:")
            for arg_name, arg_info in tool.inputs.items():
                desc.append(f"  - {arg_name}: {arg_info['description']}")
            
            # Add return type
            desc.append(f"\nReturns: {tool.output_type}")
            
            descriptions.append("\n".join(desc))
        
        return "\n\n".join(descriptions)
    
    def get_dataset(self, **kwargs: Any) -> Dataset:
        """Return the training dataset."""
        return self.dataset
    
    def get_eval_dataset(self, n: int = -1, **kwargs: Any) -> Optional[Dataset]:
        """
        Return the evaluation dataset.
        
        Args:
            n: Number of examples to select (-1 for all)
            
        Returns:
            The evaluation dataset
        """
        if self.eval_dataset is None:
            self.eval_dataset = preprocess_dataset(
                dataset_name=self.dataset_name,
                split="test",
                system_prompt=self.system_prompt,
                few_shot=self.few_shot
            )
        if n > 0:
            return self.eval_dataset.shuffle().select(range(n))
        return self.eval_dataset
    
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        """Return the reward functions for this environment."""
        return self.rubric.get_reward_funcs()
    
    def _get_step_count(self, messages: List[Dict[str, str]]) -> int:
        """
        Count the number of tool uses in the message history.
        
        Args:
            messages: List of messages in the conversation
            
        Returns:
            Number of tool uses
        """
        step_count = 0
        
        # Skip messages that are part of few-shot examples
        conversation_start = 1  # Start after system message
        if self.few_shot:
            # Account for all few-shot messages
            conversation_start += len(self.few_shot)
        
        # Only count tool uses from the actual conversation
        for message in messages[conversation_start:]:
            if message.get("role") == "assistant":
                tool_call = self.llm_parser.parse_tool_call(message["content"])
                if tool_call is not None:
                    step_count += 1
        
        return step_count
    
    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        """
        Check if the environment has completed.
        
        Args:
            messages: List of messages in the conversation
            
        Returns:
            Whether the environment has completed
        """
        try:
            # Check if we've hit max steps
            step_count = self._get_step_count(messages)
            if step_count >= self.max_steps:
                return True
            
            # Check if we got an answer
            parsed = self.llm_parser.parse(messages[-1]["content"])
            return hasattr(parsed, 'answer') and parsed.answer is not None
        except Exception:
            return False
    
    def call_tool(self, tool_call_json: str, **kwargs: Any) -> str:
        """
        Call a SmolaAgents tool with the given parameters.
        
        Args:
            tool_call_json: JSON string with tool call information
            
        Returns:
            Result of the tool call as a string
        """
        try:
            # Parse the tool call JSON
            call_data = json.loads(tool_call_json)
            
            # Extract tool name and arguments
            tool_name = call_data.get("name")
            if not tool_name:
                return "Error: Tool command must specify 'name'"
            
            if tool_name not in self.tools:
                return f"Error: Unknown tool '{tool_name}'"
            
            # Get the tool and its arguments
            tool = self.tools[tool_name]
            tool_args = call_data.get("args", {})
            
            # Call the tool with arguments
            result = tool(**tool_args)
            return str(result)
        except json.JSONDecodeError:
            return "Error: Invalid JSON format for tool call"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        """
        Generate an environment response to the model's message.
        
        Args:
            messages: List of messages in the conversation
            
        Returns:
            Environment response message
        """
        print("\n===== DEBUGGING ENV_RESPONSE =====")
        print(f"Last message content: {messages[-1]['content']}")
        
        try:
            # Check for a tool call using SmolaParser
            tool_call = self.llm_parser.parse_tool_call(messages[-1]["content"])
            print(f"Parsed tool_call: {tool_call}")
            
            # Check direct XML parsing for both tags
            parsed = self.llm_parser.parse(messages[-1]["content"])
            print(f"Direct parser results: tool_call={getattr(parsed, 'tool_call', None)}")
            
            # Also check if there's a <tool> tag (backward compatibility)
            if hasattr(parsed, 'tool') and parsed.tool is not None:
                print(f"Found <tool> tag: {parsed.tool}")
                try:
                    # Try to parse it as JSON
                    tool_json = json.loads(parsed.tool)
                    if isinstance(tool_json, dict) and 'name' in tool_json:
                        print(f"Successfully parsed <tool> tag as JSON: {tool_json}")
                        tool_call = tool_json
                except json.JSONDecodeError:
                    print(f"Could not parse <tool> tag as JSON: {parsed.tool}")
            
            if tool_call is not None:
                # Call the tool and format the result
                print(f"Calling tool: {json.dumps(tool_call) if isinstance(tool_call, dict) else tool_call}")
                result = self.call_tool(json.dumps(tool_call) if isinstance(tool_call, dict) else tool_call)
                print(f"Tool result: {result}")
                if len(result.strip()) > 0:
                    return {"role": "user", "content": self.env_parser.format(result=result)}
                else:
                    return {"role": "user", "content": "Error: Tool execution returned empty output."}
        except Exception as e:
            print(f"Exception in env_response: {e}")
            import traceback
            print(traceback.format_exc())
            return {"role": "user", "content": f"Error: {str(e)}"}
        
        print("No valid tool call found")
        print("===== END DEBUGGING ENV_RESPONSE =====\n")
        return {"role": "user", "content": "Error: Tool command not found or invalid format. Please ensure correct formatting."}