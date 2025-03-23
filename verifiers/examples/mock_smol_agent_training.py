"""
Mock training demo for the VerifiersToolAgent with SmolAgents integration.

This script demonstrates how to use the mock GRPO trainer with SmolAgentEnv,
allowing testing of the SmolAgents integration without requiring a GPU or TRL.

Usage:
    OPENAI_API_KEY=<your-api-key> python -m verifiers.examples.mock_smol_agent_training
"""
import os
import sys
import time
import logging

# Add smolagents to path if needed
sys.path.append('/Users/allanniemerg/dev/verifiers/wip/smolagents')

# Import Verifiers components
import verifiers as vf

# Try to import SmolAgents components
try:
    from verifiers.agents.tool_adapter import create_calculator_tool, create_search_tool
    from verifiers.agents.smol_agent_env import SmolAgentEnv
    from verifiers.examples.gsm8k_smol_agent import load_gsm8k_examples, prepare_message
    HAS_SMOLAGENTS = True
except ImportError:
    HAS_SMOLAGENTS = False
    print("SmolAgents integration not available. Please check if SmolAgents is installed.")


def create_mock_reward_funcs():
    """Create reward functions for the mock training."""
    
    def answer_correctness(prompts, completions, **kwargs):
        """
        Reward function for answer correctness.
        
        Args:
            prompts: List of prompt messages
            completions: List of completion messages
            **kwargs: Additional arguments
            
        Returns:
            List of rewards
        """
        import re
        examples = kwargs.get("examples", [])
        if not examples:
            return [0.0] * len(prompts)
            
        rewards = []
        for i, completion in enumerate(completions):
            # For mock training, just check if there's an answer tag
            example = examples[i % len(examples)]
            expected_answer = example["answer"]
            
            has_answer = False
            correct_answer = False
            
            for message in completion:
                if message["role"] == "assistant":
                    # Check for answer tag
                    answer_match = re.search(r"<answer>(.*?)</answer>", message["content"], re.DOTALL)
                    if answer_match:
                        has_answer = True
                        answer_text = answer_match.group(1).strip()
                        
                        # Extract number from answer
                        number_match = re.search(r"\b(\d+(?:\.\d+)?)\b", answer_text)
                        if number_match and number_match.group(1) == expected_answer:
                            correct_answer = True
            
            # Assign reward
            if correct_answer:
                rewards.append(1.0)  # Correct answer
            elif has_answer:
                rewards.append(0.2)  # At least has an answer tag
            else:
                rewards.append(0.0)  # No answer
        
        return rewards
    
    def tool_usage(prompts, completions, **kwargs):
        """
        Reward function for tool usage.
        
        Args:
            prompts: List of prompt messages
            completions: List of completion messages
            **kwargs: Additional arguments
            
        Returns:
            List of rewards
        """
        import re
        rewards = []
        
        for completion in completions:
            # Count tool calls in the completion
            tool_call_count = 0
            for message in completion:
                if message["role"] == "assistant":
                    # Count tool call tags
                    tool_calls = re.findall(r"<tool_call>", message["content"])
                    tool_call_count += len(tool_calls)
            
            # Assign reward based on tool usage
            if tool_call_count > 2:
                rewards.append(0.8)  # Multiple tool usage is good
            elif tool_call_count > 0:
                rewards.append(0.5)  # At least used a tool
            else:
                rewards.append(0.0)  # No tool usage
        
        return rewards
    
    return [answer_correctness, tool_usage]


def main():
    """Run a mock training demo with SmolAgents."""
    # Check if SmolAgents integration is available
    if not HAS_SMOLAGENTS:
        print("SmolAgents integration not available. Please check installation.")
        return
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set in environment")
        print("Usage: OPENAI_API_KEY=<your-api-key> python -m verifiers.examples.mock_smol_agent_training")
        return
    
    print("Initializing the SmolAgentEnv with calculator tool...")
    
    # Create tools
    calculator_tool = create_calculator_tool()
    
    # Create SmolAgentEnv
    env = SmolAgentEnv(
        tools=[calculator_tool],
        max_steps=3  # Limit steps for demo
    )
    
    # Load example data
    print("Loading GSM8K examples (small subset for demonstration)...")
    examples = load_gsm8k_examples(split="test", max_examples=10)
    
    # Create prompts
    prompts = [prepare_message(example) for example in examples]
    small_prompts = prompts[:5]  # Just use 5 examples for the demo
    
    print("Creating mock training configuration...")
    # Create a mock training config
    args = vf.get_default_grpo_config(
        run_name="mock_smolA_gsm8k_test",
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
    
    print("Creating mock reward functions...")
    reward_funcs = create_mock_reward_funcs()
    
    print("Creating mock trainer...")
    # Create mock trainer (no need for real model or tokenizer)
    trainer = vf.get_mock_grpo_env_trainer(
        model="dummy",  # Not used
        env=env,
        reward_funcs=reward_funcs,
        reward_weights=[0.7, 0.3],  # More weight on answer correctness
        args=args,
        train_dataset=small_prompts,  # Just use the prompts directly
        openai_model="gpt-3.5-turbo",  # Specify which OpenAI model to use
        temperature=0.7,
        max_tokens=300
    )
    
    print("\nStarting mock training process...\n")
    start_time = time.time()
    
    # Run the mock training process
    metrics = trainer.train(
        reward_fn_kwargs={"examples": examples}
    )
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f} seconds")
    print("Average reward:", sum(metrics.get("reward", [0])) / len(metrics.get("reward", [1])))
    
    print("\nThis was a mock training demo using OpenAI API instead of local models.")
    print("Note that no actual model weights were updated in this demo.")


if __name__ == "__main__":
    main()