"""
Tests for SmolaAgents tools integration.
"""

import json
import unittest
from unittest.mock import MagicMock

import pytest

from verifiers.parsers import SmolaParser
from verifiers.rubrics import SmolaToolRubric

try:
    from smolagents.default_tools import PythonInterpreterTool
    from verifiers.envs import SmolaToolEnv
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False


@pytest.mark.skipif(not SMOLAGENTS_AVAILABLE, reason="SmolaAgents not available")
class TestSmolaParser(unittest.TestCase):
    """Test the SmolaParser class."""

    def setUp(self):
        self.parser = SmolaParser(fields=["reasoning", "tool_call", "answer"])
        self.env_parser = SmolaParser(fields=["result"])

    def test_parse_tool_call(self):
        """Test parsing a tool call from XML."""
        # Create a message with a tool call
        tool_call_data = {"name": "python_interpreter", "args": {"code": "2 + 2"}}
        message = f"<reasoning>Let me calculate</reasoning>\n<tool_call>\n{json.dumps(tool_call_data)}\n</tool_call>"
        
        # Parse the tool call
        parsed = self.parser.parse_tool_call(message)
        
        # Check that the tool call was parsed correctly
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["name"], "python_interpreter")
        self.assertEqual(parsed["args"]["code"], "2 + 2")

    def test_format_tool_call(self):
        """Test formatting a tool call as XML."""
        # Format a tool call
        name = "python_interpreter"
        args = {"code": "2 + 2"}
        formatted = self.parser.format_tool_call(name, args)
        
        # Check that the tool call was formatted correctly
        self.assertIn("<tool_call>", formatted)
        self.assertIn("</tool_call>", formatted)
        self.assertIn('"name": "python_interpreter"', formatted)
        self.assertIn('"code": "2 + 2"', formatted)


@pytest.mark.skipif(not SMOLAGENTS_AVAILABLE, reason="SmolaAgents not available")
class TestSmolaToolRubric(unittest.TestCase):
    """Test the SmolaToolRubric class."""

    def setUp(self):
        self.rubric = SmolaToolRubric()

    def test_tool_execution_reward(self):
        """Test the tool execution reward function."""
        # Create a completion with a successful tool execution
        completion = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": '<reasoning>I need to calculate 2+2</reasoning>\n<tool_call>\n{"name": "python_interpreter", "args": {"code": "2 + 2"}}\n</tool_call>'},
            {"role": "user", "content": '<result>4</result>'},
            {"role": "assistant", "content": '<reasoning>The answer is 4</reasoning>\n<answer>4</answer>'}
        ]
        
        # Calculate the reward
        rewards = self.rubric.tool_execution_reward_func([completion])
        
        # Check that the reward is positive
        self.assertEqual(len(rewards), 1)
        self.assertGreater(rewards[0], 0)

    def test_exact_answer_reward(self):
        """Test the exact answer reward function."""
        # Create a completion with a correct answer
        completion = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": '<reasoning>The answer is 4</reasoning>\n<answer>4</answer>'}
        ]
        
        # Define the reference
        reference = {"answer": "4"}
        
        # Calculate the reward
        rewards = self.rubric.exact_answer_reward_func([completion], [reference])
        
        # Check that the reward is 1.0 (correct answer)
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)


@pytest.mark.skipif(not SMOLAGENTS_AVAILABLE, reason="SmolaAgents not available")
class TestSmolaToolEnv(unittest.TestCase):
    """Test the SmolaToolEnv class."""

    def setUp(self):
        # Create a mock Python interpreter tool
        self.python_tool = MagicMock()
        self.python_tool.name = "python_interpreter"
        self.python_tool.description = "Evaluates Python code"
        self.python_tool.inputs = {"code": {"type": "string", "description": "Python code to evaluate"}}
        self.python_tool.output_type = "string"
        self.python_tool.return_value = "4"
        
        # Create the environment
        self.env = SmolaToolEnv(
            dataset="gsm8k",
            tools=[self.python_tool],
            few_shot=[],  # No few-shot examples for testing
            max_steps=5
        )

    def test_format_tool_descriptions(self):
        """Test formatting tool descriptions."""
        # Format tool descriptions
        tools = [self.python_tool]
        descriptions = self.env._format_tool_descriptions(tools)
        
        # Check that the description contains the expected information
        self.assertIn("python_interpreter", descriptions)
        self.assertIn("Evaluates Python code", descriptions)
        self.assertIn("code", descriptions)
        self.assertIn("Python code to evaluate", descriptions)

    def test_call_tool(self):
        """Test calling a tool."""
        # Create a tool call
        tool_call = json.dumps({
            "name": "python_interpreter",
            "args": {"code": "2 + 2"}
        })
        
        # Call the tool
        result = self.env.call_tool(tool_call)
        
        # Check that the tool was called
        self.python_tool.assert_called_once_with(code="2 + 2")
        
        # Check that the result is what we expect
        self.assertEqual(result, "4")

    def test_is_completed(self):
        """Test completion detection."""
        # Create a message history with an answer
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": '<reasoning>The answer is 4</reasoning>\n<answer>4</answer>'}
        ]
        
        # Check that the environment is completed
        self.assertTrue(self.env.is_completed(messages))
        
        # Create a message history without an answer
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": '<reasoning>I need to calculate</reasoning>'}
        ]
        
        # Check that the environment is not completed
        self.assertFalse(self.env.is_completed(messages))


if __name__ == "__main__":
    unittest.main()