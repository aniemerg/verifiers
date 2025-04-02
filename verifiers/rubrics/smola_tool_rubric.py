import json
import re
from typing import List, Dict, Any, Callable

from verifiers import RewardFunc
from verifiers.parsers.smola_parser import SmolaParser
from verifiers.rubrics.rubric import Rubric

class SmolaToolRubric(Rubric):
    """
    Rubric for evaluating SmolaAgents tool usage.
    
    This rubric implements reward functions specific to the SmolaAgents tool format,
    focusing on <tool> tags and proper JSON structure for tool calls.
    """
    
    def __init__(self, tools: List[Callable] = []):
        """
        Initialize the SmolaToolRubric with the available tools.
        
        Args:
            tools: List of callable tool functions to evaluate usage of
        """
        self.tools = {tool.__name__: tool for tool in tools}
        self.parser = SmolaParser(fields=["reasoning", "tool", "answer"])
        
        # Set up the reward functions and their weights
        self._reward_funcs = [
            self.format_reward_func,
            self.tool_execution_reward_func,
            self.tool_selection_reward_func,
            self.reasoning_quality_reward_func
        ]
        
        self._reward_weights = [0.25, 0.3, 0.25, 0.2]  # Weights should sum to 1
    
    def get_reward_funcs(self) -> List[RewardFunc]:
        """Return the list of reward functions."""
        return self._reward_funcs
    
    def get_reward_weights(self) -> List[float]:
        """Return the weights for each reward function."""
        return self._reward_weights
    
    def format_reward_func(self, completions, **kwargs) -> List[float]:
        """
        Reward function that evaluates whether the model uses the correct format.
        
        This function specifically checks for proper <tool> tag usage and JSON structure.
        
        Args:
            completions: List of completion trajectories
            **kwargs: Additional arguments
            
        Returns:
            List of reward scores, one for each completion
        """
        def evaluate_format(trajectory):
            # Get assistant messages
            assistant_msgs = [msg for msg in trajectory if msg['role'] == 'assistant']
            if not assistant_msgs:
                return 0.0
            
            format_scores = []
            for msg in assistant_msgs:
                content = msg['content']
                score = 0.0
                
                # Check for proper tag usage
                tool_tags_present = "<tool>" in content and "</tool>" in content
                answer_tags_present = "<answer>" in content and "</answer>" in content
                
                # Give credit for using the right tags
                if tool_tags_present or answer_tags_present:
                    score += 0.5
                
                # Check tool format if tool tags are present
                if tool_tags_present:
                    tool_call = self.parser.parse_tool(content)
                    if tool_call and isinstance(tool_call, dict):
                        # Check for proper name and args fields
                        has_name = "name" in tool_call
                        has_args = "args" in tool_call
                        args_is_dict = isinstance(tool_call.get("args", {}), dict)
                        
                        if has_name and has_args and args_is_dict:
                            score += 0.5
                        elif has_name or has_args:
                            score += 0.25  # Partial credit for partial structure
                
                # Check answer format
                if answer_tags_present:
                    parsed = self.parser.parse(content)
                    if hasattr(parsed, 'answer') and parsed.answer is not None:
                        score += 0.5
                
                format_scores.append(min(1.0, score))  # Cap at 1.0
            
            # Average the scores across all messages
            return sum(format_scores) / len(format_scores) if format_scores else 0.0
        
        return [evaluate_format(comp) for comp in completions]
    
    def tool_execution_reward_func(self, completions, **kwargs) -> List[float]:
        """
        Reward function that evaluates whether tools are executed correctly.
        
        This rewards proper tool usage including valid arguments and handling of the result.
        
        Args:
            completions: List of completion trajectories
            **kwargs: Additional arguments
            
        Returns:
            List of reward scores, one for each completion
        """
        def evaluate_tool_execution(trajectory):
            # Get assistant messages
            assistant_msgs = [msg for msg in trajectory if msg['role'] == 'assistant']
            if not assistant_msgs:
                return 0.0
            
            execution_scores = []
            for i, msg in enumerate(assistant_msgs):
                content = msg['content']
                score = 0.0
                
                # Check if tool tags are present
                if "<tool>" in content and "</tool>" in content:
                    tool_call = self.parser.parse_tool(content)
                    
                    if tool_call is not None:
                        # Validate tool name
                        has_valid_tool = tool_call.get("name") in self.tools
                        score += 0.5 if has_valid_tool else 0.0
                        
                        # Validate arguments
                        args = tool_call.get("args", {})
                        if has_valid_tool and isinstance(args, dict):
                            # Check if all required arguments are provided
                            tool_name = tool_call.get("name")
                            tool_func = self.tools[tool_name]
                            
                            try:
                                # Try to get the signature
                                import inspect
                                sig = inspect.signature(tool_func)
                                required_args = {
                                    name for name, param in sig.parameters.items()
                                    if param.default == inspect.Parameter.empty
                                }
                                
                                # Check if all required args are present
                                all_required_present = all(arg in args for arg in required_args)
                                score += 0.5 if all_required_present else 0.25
                            except Exception:
                                # If we can't get the signature, give partial credit
                                score += 0.25
                    
                # Check for proper handling of the result (looking for the next message)
                if i < len(assistant_msgs) - 1 and i + 1 < len(trajectory):
                    # Get the next user message (environment response)
                    next_msg = trajectory[trajectory.index(msg) + 1]
                    if next_msg['role'] == 'user' and '<r>' in next_msg['content']:
                        # Check if the assistant uses the result in its next message
                        if i + 1 < len(assistant_msgs):
                            next_assistant_msg = assistant_msgs[i + 1]
                            if "reasoning" in next_assistant_msg['content'].lower():
                                score += 0.2  # Bonus for reasoning based on the result
                
                execution_scores.append(min(1.0, score))  # Cap at 1.0
            
            # Average the scores across all messages
            return sum(execution_scores) / len(execution_scores) if execution_scores else 0.0
        
        return [evaluate_tool_execution(comp) for comp in completions]
    
    def tool_selection_reward_func(self, completions, **kwargs) -> List[float]:
        """
        Reward function that evaluates whether the model selects appropriate tools.
        
        This rewards selecting the appropriate tool for the task and using it at the right time.
        
        Args:
            completions: List of completion trajectories
            **kwargs: Additional arguments
            
        Returns:
            List of reward scores, one for each completion
        """
        def evaluate_tool_selection(trajectory):
            # Extract task information if available
            task = kwargs.get('task', [None] * len(completions))[0] if 'task' in kwargs else None
            assistant_msgs = [msg for msg in trajectory if msg['role'] == 'assistant']
            
            if not assistant_msgs:
                return 0.0
            
            # If no specific task is given, just evaluate based on tool variety and completion
            if task is None:
                # Count unique tools used
                used_tools = set()
                for msg in assistant_msgs:
                    tool_call = self.parser.parse_tool(msg['content'])
                    if tool_call and 'name' in tool_call:
                        used_tools.add(tool_call['name'])
                
                # Simple heuristic: using at least one tool (25%), variety of tools (25%), 
                # and a final answer (50%)
                used_at_least_one = len(used_tools) > 0
                used_variety = len(used_tools) > 1
                has_final_answer = any("<answer>" in msg['content'] for msg in assistant_msgs)
                
                score = (0.25 if used_at_least_one else 0.0) + \
                        (0.25 if used_variety else 0.0) + \
                        (0.5 if has_final_answer else 0.0)
                
                return score
            
            # For specific tasks, we would need task-specific evaluation
            # This is a simplistic approach that can be extended
            return 0.5  # Default score for task-based evaluation
        
        return [evaluate_tool_selection(comp) for comp in completions]
    
    def reasoning_quality_reward_func(self, completions, **kwargs) -> List[float]:
        """
        Reward function that evaluates the quality of reasoning.
        
        This rewards clear reasoning about the problem, tool usage, and conclusions.
        
        Args:
            completions: List of completion trajectories
            **kwargs: Additional arguments
            
        Returns:
            List of reward scores, one for each completion
        """
        def evaluate_reasoning(trajectory):
            assistant_msgs = [msg for msg in trajectory if msg['role'] == 'assistant']
            if not assistant_msgs:
                return 0.0
            
            reasoning_scores = []
            for msg in assistant_msgs:
                content = msg['content']
                score = 0.0
                
                # Check for reasoning sections
                parsed = self.parser.parse(content)
                has_reasoning = hasattr(parsed, 'reasoning') and parsed.reasoning is not None
                
                if has_reasoning:
                    reasoning = parsed.reasoning
                    
                    # Length-based heuristic (longer reasoning is usually better)
                    # This is a simplistic approach that can be enhanced
                    words = len(reasoning.split())
                    if words > 50:  # Substantial reasoning
                        score += 0.7
                    elif words > 20:  # Moderate reasoning
                        score += 0.4
                    elif words > 5:   # Brief reasoning
                        score += 0.2
                    
                    # Check for problem understanding (keywords like "first", "next", "then")
                    step_words = ["first", "second", "third", "next", "then", "finally"]
                    has_step_words = any(word in reasoning.lower() for word in step_words)
                    if has_step_words:
                        score += 0.3
                
                reasoning_scores.append(min(1.0, score))  # Cap at 1.0
            
            # Average the scores across all messages
            return sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else 0.0
        
        return [evaluate_reasoning(comp) for comp in completions]