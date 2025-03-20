"""
Demo showing how to use the mock GRPOEnvTrainer with OpenAI API.
This demonstrates how to test environments without requiring GPU or TRL.

Usage:
    OPENAI_API_KEY=<your-api-key> python -m verifiers.examples.mock_trainer_demo

The mock implementation allows running the verifiers package on macOS or other
environments where vLLM and torch CUDA might not be available.
"""
import os
import time

import verifiers as vf
from verifiers.tools import calculator
from verifiers.prompts import CALCULATOR_FEW_SHOT


def main():
    """Run a demonstration of the mock trainer with ToolEnv."""
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set in environment")
        print("Usage: OPENAI_API_KEY=<your-api-key> python -m verifiers.examples.mock_trainer_demo")
        return
    
    print("Initializing the ToolEnv with calculator tool...")
    
    # Initialize ToolEnv with calculator tool
    env = vf.ToolEnv(
        dataset="gsm8k",
        few_shot=CALCULATOR_FEW_SHOT[0],
        tools=[calculator],
        max_steps=3
    )
    
    # Get datasets
    print("Loading datasets (small subset for demonstration)...")
    try:
        # Use full dataset but limit to a small number of examples
        dataset = env.get_dataset()
        small_dataset = dataset.select(range(10))  # Just 10 examples
        eval_dataset = env.get_eval_dataset(n=5)  # Just 5 examples
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        return
    
    print("Creating mock training configuration...")
    # Create a mock training config
    args = vf.get_default_grpo_config(
        run_name="mock_gsm8k_test",
        num_gpus=0,  # No GPUs needed
        for_mock=True,  # Force using mock config
        max_steps=3  # Very short run for demo
    )
    
    # Override some settings for the demo
    args.per_device_train_batch_size = 2
    args.per_device_eval_batch_size = 2
    args.num_generations = 1
    args.logging_steps = 1
    args.eval_steps = 2
    
    print("Creating mock trainer...")
    # Create mock trainer (no need for real model or tokenizer)
    trainer = vf.get_mock_grpo_env_trainer(
        model="dummy",  # Not used
        env=env,
        reward_funcs=env.get_rubric(),
        args=args,
        train_dataset=small_dataset,
        eval_dataset=eval_dataset,
        openai_model="gpt-3.5-turbo",  # Specify which OpenAI model to use
        temperature=0.7,
        max_tokens=200
    )
    
    print("\nStarting training process...\n")
    start_time = time.time()
    
    # Run the mock training process
    metrics = trainer.train()
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f} seconds")
    print("Average reward:", sum(metrics.get("reward", [0])) / len(metrics.get("reward", [1])))


if __name__ == "__main__":
    main()