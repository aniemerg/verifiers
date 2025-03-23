"""Test for SmolAgents integration."""

import sys
import unittest
from unittest.mock import MagicMock, patch
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import SmolAgents dependencies
try:
    from verifiers.agents.model_adapter import VerifiersModelAdapter
    from verifiers.agents.verifiers_agent import VerifiersToolAgent
    from verifiers.agents.smol_agent_env import SmolAgentEnv
    from verifiers.agents.tool_adapter import create_calculator_tool
    from verifiers.agents.agent_parser import VerifiersAgentParser
    HAS_SMOLAGENTS = True
except ImportError:
    HAS_SMOLAGENTS = False


@unittest.skipIf(not HAS_SMOLAGENTS, "SmolAgents integration not available")
class TestSmolAgentIntegration(unittest.TestCase):
    """Test the SmolAgents integration."""
    
    def test_model_adapter_init(self):
        """Test that the model adapter can be initialized."""
        adapter = VerifiersModelAdapter()
        self.assertIsNotNone(adapter)
    
    def test_agent_init(self):
        """Test that the agent can be initialized."""
        # Create a mock model adapter
        model_adapter = MagicMock()
        
        # Create a mock tool
        calculator_tool = create_calculator_tool()
        
        # Create the agent
        agent = VerifiersToolAgent(
            tools=[calculator_tool],
            model=model_adapter,
            max_steps=10
        )
        
        self.assertIsNotNone(agent)
        self.assertEqual(len(agent.tools), 1)
    
    def test_parser(self):
        """Test the agent parser."""
        parser = VerifiersAgentParser()
        
        # Test with a valid tool call
        test_text = """
        <reasoning>
        I need to calculate 2 + 2.
        </reasoning>
        
        <tool_call>
        {"name": "calculator", "args": {"expression": "2 + 2"}}
        </tool_call>
        """
        
        parsed = parser.parse(test_text)
        
        self.assertTrue(hasattr(parsed, 'reasoning'))
        self.assertTrue(hasattr(parsed, 'tool_call'))
        self.assertEqual(parsed.tool_call['name'], 'calculator')
        self.assertEqual(parsed.tool_call['args']['expression'], '2 + 2')
    
    def test_env_init(self):
        """Test that the environment can be initialized."""
        # Create a mock tool
        calculator_tool = create_calculator_tool()
        
        # Create the environment
        env = SmolAgentEnv(
            tools=[calculator_tool],
            max_steps=10
        )
        
        self.assertIsNotNone(env)
        self.assertEqual(len(env.tools), 1)


if __name__ == '__main__':
    unittest.main()