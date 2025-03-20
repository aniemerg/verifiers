"""
Demo showing how to use the mock LLM with OpenAI API in MultiStepEnv.
This demonstrates how to run the MultiStepEnv with OpenAI models instead of vLLM.

Usage:
    OPENAI_API_KEY=<your-api-key> python -m verifiers.examples.openai_multistep_demo

The mock implementation allows running the verifiers package on macOS or other
environments where vLLM might not be available.
"""
import os
from typing import List, Dict, Any

from openai import OpenAI
from verifiers.imports import LLM, SamplingParams
from verifiers.envs.tool_env import ToolEnv
from verifiers.tools.calculator import calculator
from verifiers.prompts.few_shots import CALCULATOR_FEW_SHOT


def main():
    """Run a simple demonstration of the mock LLM with MultiStepEnv."""
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set in environment")
        print("Usage: OPENAI_API_KEY=<your-api-key> python -m verifiers.examples.openai_multistep_demo")
        return
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Initialize mock LLM
    llm = LLM(
        openai_client=client,
        model="gpt-3.5-turbo",  # Can be changed to any OpenAI model
        max_workers=5,          # Limit concurrent requests
        retry_attempts=3        # Retry failed requests
    )
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=200
    )
    
    # Initialize ToolEnv with calculator tool
    env = ToolEnv(
        tools=[calculator],
        few_shot=CALCULATOR_FEW_SHOT[0],
        max_steps=3
    )
    
    # Create a test prompt with a math problem
    prompt = [
        {"role": "system", "content": env.system_prompt},
        {"role": "user", "content": "If I have 5 apples and eat 2, then buy 3 more, how many do I have?"}
    ]
    
    print("Running demo with OpenAI API and ToolEnv...")
    print("Sending prompt:", prompt[1]["content"])
    print()
    
    # Run generation
    result = env.generate([prompt], llm, sampling_params)
    
    # Display results
    print("Generation complete!")
    print("Generated messages:")
    for i, msg in enumerate(result["messages"][0]):
        print(f"Message {i+1}: {msg['role']} - {msg['content']}")
    
    print("\nToken IDs length:", len(result["ids"][0]))
    print("Mask length:", len(result["mask"][0]))
    
    # Demo the API-based evaluation
    print("\nRunning API-based evaluation demo...")
    
    # Create a simple dataset for testing
    from datasets import Dataset
    
    test_data = {
        "prompt": [
            [
                {"role": "system", "content": env.system_prompt},
                {"role": "user", "content": "What is 25 * 17?"}
            ],
            [
                {"role": "system", "content": env.system_prompt},
                {"role": "user", "content": "If I have 120 dollars and spend 45, then earn 20 more, how much do I have?"}
            ]
        ],
        "answer": ["425", "95"]
    }
    
    env.eval_dataset = Dataset.from_dict(test_data)
    
    # Run API-based evaluation
    try:
        rewards = env.eval_api(
            client=client,
            model="gpt-3.5-turbo",
            max_concurrent=2,
            timeout=30
        )
        print("Evaluation complete!")
        print("Rewards:", rewards)
    except Exception as e:
        print(f"Evaluation error: {str(e)}")


if __name__ == "__main__":
    main()