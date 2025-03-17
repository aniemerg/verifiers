import json
from typing import List, Dict, Any

from verifiers.parsers.smola_parser import SmolaParser
from verifiers.rubrics.rubric import Rubric

class SmolaToolRubric(Rubric):
    """
    Rubric for evaluating SmolaAgents tools usage.
    
    This rubric evaluates:
    1. Whether the model provided an exact answer
    2. Whether tools were executed successfully
    3. Whether the format follows XML guidelines
    4. Whether XML tags are used properly
    """
    
    def __init__(self,
                 parser: SmolaParser = SmolaParser(fields=["reasoning", "tool_call", "answer"]),
                 env_parser: SmolaParser = SmolaParser(fields=["result"])):
        self.parser = parser
        self.env_parser = env_parser
        self.reward_funcs = [
            self.exact_answer_reward_func,
            self.tool_execution_reward_func,
            self.parser.get_format_reward_func(),
            self.parser.get_xml_reward_func(),
        ]

    def exact_answer_reward_func(self, completions: List[List[Dict[str, str]]], references: List[Dict[str, Any]], **kwargs) -> List[float]:
        """
        Reward function for providing the exact reference answer.
        
        Args:
            completions: List of completion trajectories (lists of messages)
            references: List of reference data with correct answers
            
        Returns:
            List of float rewards (1.0 for correct answer, 0.0 otherwise)
        """
        rewards = []
        
        for completion, reference in zip(completions, references):
            model_answer = None
            for message in completion:
                if message['role'] == 'assistant':
                    try:
                        parsed = self.parser.parse(message['content'])
                        if hasattr(parsed, 'answer') and parsed.answer is not None:
                            model_answer = parsed.answer
                    except Exception:
                        pass
            
            # Check if the model's answer matches the reference answer
            correct_answer = reference.get('answer', '')
            if model_answer and correct_answer:
                # Simple exact match for now - could be expanded
                reward = 1.0 if model_answer.strip() == correct_answer.strip() else 0.0
            else:
                reward = 0.0
            
            rewards.append(reward)
        
        return rewards

    def tool_execution_reward_func(self, completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """
        Reward function that checks tool execution success for SmolaAgents tools.
        
        Args:
            completions: List of completion trajectories (lists of messages)
            
        Returns:
            List of float rewards (0.0-0.2) based on tool execution success rate
        """
        def check_execution(trajectory):
            tool_attempts = 0
            successful_executions = 0
            
            # Find assistant messages with tools and their responses
            for i, msg in enumerate(trajectory):
                if msg['role'] == 'assistant':
                    # Parse the tool call
                    tool_call = self.parser.parse_tool_call(msg['content'])
                    if tool_call is not None:
                        # Found a properly formatted tool call
                        if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'user':
                            tool_attempts += 1
                            # Check response with env_parser
                            parsed_response = self.env_parser.parse(trajectory[i + 1]['content'])
                            if hasattr(parsed_response, 'result') and parsed_response.result is not None and not parsed_response.result.startswith("Error:"):
                                successful_executions += 1
            
            # Calculate reward
            if tool_attempts == 0:
                return 0.0
            return 0.2 * (successful_executions / tool_attempts)
        
        return [check_execution(c) for c in completions]