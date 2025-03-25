"""
GSM8K example using SmolAgents integration.

This example demonstrates the integration between Verifiers and SmolAgents,
using a calculator tool to solve GSM8K math problems.
"""

import os
import sys
import logging
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add smolagents src directory to path
logger.debug("Adding SmolAgents src directory to path")
sys.path.insert(0, '/Users/allanniemerg/dev/verifiers/wip/smolagents/src')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../wip/smolagents/src')))

# Import Verifiers components
from verifiers.imports import load_dataset, vllm_llm_factory

# Import our agent components
from verifiers.agents.model_adapter import VerifiersModelAdapter
from verifiers.agents.verifiers_agent import VerifiersToolAgent
from verifiers.agents.smol_agent_env import SmolAgentEnv
from verifiers.agents.tool_adapter import create_calculator_tool


def load_gsm8k_examples(split="train", max_examples=100):
    """
    Load examples from the GSM8K dataset.
    
    Args:
        split: Dataset split to use (train or test).
        max_examples: Maximum number of examples to load.
        
    Returns:
        List of examples with question and answer.
    """
    # Specify the 'main' config for GSM8K
    dataset = load_dataset("gsm8k", "main", split=split)
    
    examples = []
    for i, example in enumerate(dataset):
        if i >= max_examples:
            break
            
        # Extract the question and answer
        question = example["question"]
        answer = example["answer"]
        
        # Extract just the final numerical answer
        import re
        answer_match = re.search(r"The answer is (\d+)", answer)
        if answer_match:
            final_answer = answer_match.group(1)
        else:
            final_answer = answer.split("####")[-1].strip()
        
        examples.append({
            "question": question,
            "answer": final_answer
        })
    
    return examples


def prepare_message(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Prepare a message for the agent from an example.
    
    Args:
        example: Example dictionary with question and answer.
        
    Returns:
        List of message dictionaries.
    """
    return [
        {"role": "system", "content": "You are an expert math problem solver."},
        {"role": "user", "content": f"Solve this math problem step-by-step: {example['question']}"}
    ]


def create_env(tools=None):
    """
    Create a SmolAgentEnv for solving math problems.
    
    Args:
        tools: List of tools to provide to the agent.
        
    Returns:
        A SmolAgentEnv instance.
    """
    # Create tools if not provided
    if tools is None:
        tools = [create_calculator_tool()]
    
    # Create the environment
    env = SmolAgentEnv(
        tools=tools,
        max_steps=10
    )
    
    return env


def run_example():
    """Run the GSM8K example with SmolAgents integration."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Load examples
    examples = load_gsm8k_examples(split="test", max_examples=5)
    
    # Create prompts from examples
    prompts = [prepare_message(example) for example in examples]
    
    # Create LLM
    llm = vllm_llm_factory()
    
    # Create environment
    env = create_env()
    
    # Run the examples
    from vllm.sampling_params import SamplingParams
    sampling_params = SamplingParams(temperature=0.2, max_tokens=1024)
    
    results = env.generate(
        prompts=prompts,
        llm=llm,
        sampling_params=sampling_params
    )
    
    # Extract completions
    completions = results["messages"]
    
    # Compare with ground truth
    for i, (example, completion) in enumerate(zip(examples, completions)):
        print(f"\n=== Example {i+1} ===")
        print(f"Question: {example['question']}")
        print(f"Expected answer: {example['answer']}")
        
        # Extract the final answer from the completion
        final_answer = None
        for message in completion:
            if message["role"] == "assistant":
                # Look for an answer tag
                import re
                answer_match = re.search(r"<answer>(.*?)</answer>", message["content"], re.DOTALL)
                if answer_match:
                    final_answer = answer_match.group(1).strip()
                    break
        
        print(f"Agent answer: {final_answer}")
        print(f"Correct: {final_answer == example['answer']}")
        
        # Print the full conversation
        print("\nConversation:")
        for j, message in enumerate(completion):
            role = message["role"]
            content = message["content"]
            print(f"{role.upper()}: {content[:100]}{'...' if len(content) > 100 else ''}")


if __name__ == "__main__":
    run_example()