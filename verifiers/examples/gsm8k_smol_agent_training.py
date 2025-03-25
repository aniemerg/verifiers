"""
GRPO training example using SmolAgents integration.

This example demonstrates how to train an agent using the GRPO trainer
with the SmolAgents integration.
"""

import os
import sys
import logging
import argparse
from typing import Dict, List, Any, Callable

# Add smolagents to path if needed
sys.path.append('/Users/allanniemerg/dev/verifiers/wip/smolagents')

# Import Verifiers components
from verifiers.imports import load_dataset, vllm_llm_factory
from verifiers.trainers.grpo_env_trainer import GRPOEnvTrainer
from verifiers.utils.config_utils import load_config

# Import agent components
from verifiers.agents.smol_agent_env import SmolAgentEnv
from verifiers.agents.tool_adapter import create_calculator_tool

# Import example utilities
from verifiers.examples.gsm8k_smol_agent import load_gsm8k_examples, prepare_message


def create_agent_env(config):
    """
    Create a SmolAgentEnv for training.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        A SmolAgentEnv instance.
    """
    # Create tools
    tools = [create_calculator_tool()]
    
    # Create the environment
    env = SmolAgentEnv(
        tools=tools,
        max_steps=config.get("max_steps", 10)
    )
    
    return env


def numeric_answer_reward(prompts, completions, **kwargs):
    """
    Compute rewards based on numerical answer correctness.
    
    Args:
        prompts: List of prompt message lists.
        completions: List of completion message lists.
        **kwargs: Additional arguments.
        
    Returns:
        List of reward values.
    """
    import re
    
    # Get the ground truth answers
    examples = kwargs.get("examples", [])
    if not examples:
        return [0.0] * len(prompts)
    
    rewards = []
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        # Get the expected answer
        example = examples[i % len(examples)]
        expected_answer = example["answer"]
        
        # Extract the final answer from the completion
        final_answer = None
        for message in completion:
            if message["role"] == "assistant":
                # Look for an answer tag
                answer_match = re.search(r"<answer>(.*?)</answer>", message["content"], re.DOTALL)
                if answer_match:
                    answer_text = answer_match.group(1).strip()
                    
                    # Try to extract a numerical answer
                    number_match = re.search(r"\b(\d+(?:\.\d+)?)\b", answer_text)
                    if number_match:
                        final_answer = number_match.group(1)
                    else:
                        final_answer = answer_text
                    break
        
        # Compute reward
        if final_answer is not None and final_answer == expected_answer:
            rewards.append(1.0)  # Correct answer
        else:
            rewards.append(0.0)  # Incorrect or no answer
    
    return rewards


def step_efficiency_reward(prompts, completions, **kwargs):
    """
    Compute rewards based on step efficiency.
    
    Args:
        prompts: List of prompt message lists.
        completions: List of completion message lists.
        **kwargs: Additional arguments.
        
    Returns:
        List of reward values.
    """
    # Get the environment instance
    env = kwargs.get("env")
    if not env:
        return [0.0] * len(prompts)
    
    # Get metrics for agent states
    all_states = kwargs.get("states", [])
    if not all_states:
        return [0.0] * len(prompts)
    
    # Extract metrics from the agent pool
    metrics = env.get_agent_metrics(all_states)
    
    # Extract step efficiency from metrics
    rewards = [m.get("steps_efficiency", 0.0) for m in metrics]
    
    return rewards


def tool_usage_reward(prompts, completions, **kwargs):
    """
    Compute rewards based on tool usage quality.
    
    Args:
        prompts: List of prompt message lists.
        completions: List of completion message lists.
        **kwargs: Additional arguments.
        
    Returns:
        List of reward values.
    """
    # Get the environment instance
    env = kwargs.get("env")
    if not env:
        return [0.0] * len(prompts)
    
    # Get metrics for agent states
    all_states = kwargs.get("states", [])
    if not all_states:
        return [0.0] * len(prompts)
    
    # Extract metrics from the agent pool
    metrics = env.get_agent_metrics(all_states)
    
    # Extract tool success rate from metrics
    rewards = [m.get("tool_success_rate", 0.0) for m in metrics]
    
    return rewards


def train(config):
    """
    Train an agent using GRPO.
    
    Args:
        config: Configuration dictionary.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Load examples
    examples = load_gsm8k_examples(
        split=config.get("dataset_split", "train"),
        max_examples=config.get("max_examples", 1000)
    )
    
    # Create prompts from examples
    prompts = [prepare_message(example) for example in examples]
    
    # Create environment
    env = create_agent_env(config)
    
    # Create reward functions
    reward_funcs = [
        numeric_answer_reward,  # 40% weight on correct answer
        step_efficiency_reward,  # 30% weight on step efficiency
        tool_usage_reward       # 30% weight on tool usage quality
    ]
    
    # Create reward weights
    reward_weights = [0.4, 0.3, 0.3]
    
    # Create LLM with specified model
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    llm = vllm_llm_factory(model_name=model_name)
    
    # Create trainer
    trainer = GRPOEnvTrainer(
        llm=llm,
        env=env,
        n_epochs=config.get("n_epochs", 3),
        batch_size=config.get("batch_size", 8),
        lr=config.get("lr", 5e-6),
        reward_fn=reward_funcs,
        reward_weights=reward_weights,
        kl_coef=config.get("kl_coef", 0.1),
        checkpoint_dir=config.get("checkpoint_dir", "./checkpoints"),
        save_best=True
    )
    
    # Train the model
    trainer.train(
        prompts=prompts,
        reward_fn_kwargs={"examples": examples, "env": env}
    )


def main():
    parser = argparse.ArgumentParser(description="Train an agent using GRPO")
    parser.add_argument("--config", type=str, default="configs/zero3.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run training
    train(config)


if __name__ == "__main__":
    main()