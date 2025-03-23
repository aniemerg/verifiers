"""
Test script for SmolAgents integration.

This script provides a simple way to test the SmolAgents integration
with a single example, showing the full interaction.
"""

import os
import sys
import logging
import argparse
from typing import List, Dict, Any

# Add smolagents to path if needed
sys.path.append('/Users/allanniemerg/dev/verifiers/wip/smolagents')

# Import necessary components
from verifiers.imports import vllm_llm_factory
from verifiers.agents.model_adapter import VerifiersModelAdapter
from verifiers.agents.verifiers_agent import VerifiersToolAgent
from verifiers.agents.tool_adapter import create_calculator_tool, create_search_tool


def test_agent(problem, tool_names=None):
    """
    Test an agent on a single problem.
    
    Args:
        problem: The problem to solve.
        tool_names: List of tool names to use.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create tools
    available_tools = {
        "calculator": create_calculator_tool(),
        "search": create_search_tool()
    }
    
    # Select tools
    if tool_names:
        tools = [available_tools[name] for name in tool_names if name in available_tools]
    else:
        tools = list(available_tools.values())
    
    # Create LLM
    llm = vllm_llm_factory()
    
    # Create model adapter
    from vllm.sampling_params import SamplingParams
    sampling_params = SamplingParams(temperature=0.2, max_tokens=1024)
    
    model_adapter = VerifiersModelAdapter()
    model_adapter.update_llm(llm, sampling_params)
    
    # Create agent
    agent = VerifiersToolAgent(
        tools=tools,
        model=model_adapter,
        max_steps=10
    )
    
    # Prepare messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": problem}
    ]
    
    # Run agent
    logger.info(f"Testing agent on problem: {problem}")
    logger.info(f"Available tools: {', '.join(available_tools.keys())}")
    
    # Use the agent's run method
    from smolagents.monitoring import ConsoleLogger
    
    result = agent.run(
        task=problem,
        stream=True,
        logger=ConsoleLogger(logging_level="INFO")  # Show detailed logs
    )
    
    # Print the final result
    print("\n=== Final Result ===")
    print(result)


def main():
    parser = argparse.ArgumentParser(description="Test an agent on a single problem")
    parser.add_argument("problem", type=str, help="The problem to solve")
    parser.add_argument("--tools", type=str, nargs="+", help="Tools to use (calculator, search)")
    args = parser.parse_args()
    
    # Run the test
    test_agent(args.problem, args.tools)


if __name__ == "__main__":
    main()