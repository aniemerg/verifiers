"""
Demo showing how to use the mock LLM with OpenAI API in MultiStepEnv.
This demonstrates how to run the MultiStepEnv with OpenAI models instead of vLLM.

Usage:
    OPENAI_API_KEY=<your-api-key> python examples/openai_multistep_demo.py

The mock implementation allows running the verifiers package on macOS or other
environments where vLLM might not be available.
"""
import os
import sys
from typing import List, Dict, Any

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai import OpenAI
from verifiers.imports import LLM, SamplingParams
from verifiers.envs.simple_env import SimpleEnv


def main():
    """Run a simple demonstration of the mock LLM with MultiStepEnv."""
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set in environment")
        print("Usage: OPENAI_API_KEY=<your-api-key> python examples/openai_multistep_demo.py")
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
    
    # Initialize simple environment (from verifiers.envs.simple_env)
    env = SimpleEnv(system_prompt="You are a helpful assistant.")
    
    # Create a test prompt
    prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the concept of reinforcement learning in 3 sentences."}
    ]
    
    print("Running demo with OpenAI API and MultiStepEnv...")
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
    
    # Create a simple dataset
    from datasets import Dataset
    
    test_data = {
        "prompt": [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is machine learning?"}
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain neural networks simply."}
            ]
        ],
        "answer": ["A type of AI", "Simplified computing"]
    }
    
    env.eval_dataset = Dataset.from_dict(test_data)
    
    # Run API-based evaluation (requires rubric implementation)
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