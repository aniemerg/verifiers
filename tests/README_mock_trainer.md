# Mock GRPOEnvTrainer for macOS Development

This document describes how to use the mock GRPOEnvTrainer implementation in the `verifiers` package. This mock allows you to test Environment classes without requiring GPU or the full TRL stack, which is particularly useful on macOS or other systems where these dependencies may not be available.

## Overview

The `MockGRPOEnvTrainer` is a lightweight implementation that:
- Works with our existing OpenAI-based mock LLM
- Removes all GPU and distributed training dependencies
- Maintains the same interface as the real GRPOEnvTrainer
- Allows testing environments, reward functions, and generation pipelines

## Prerequisites

The mock implementation requires these additional packages:

```bash
pip install openai tiktoken
```

You will also need to set an OpenAI API key in your environment:

```bash
export OPENAI_API_KEY=your_key_here
```

## Basic Usage

Here's how to use the mock trainer:

```python
import verifiers as vf
from verifiers.tools import calculator
from verifiers.prompts import CALCULATOR_FEW_SHOT

# Initialize environment
env = vf.ToolEnv(
    dataset="gsm8k",
    few_shot=CALCULATOR_FEW_SHOT[0],
    tools=[calculator],
    max_steps=3
)

# Get a small subset of data for testing
dataset = env.get_dataset()
small_dataset = dataset.select(range(10))  # Just 10 examples
eval_dataset = env.get_eval_dataset(n=5)  # Just 5 examples

# Create mock training args
args = vf.get_default_grpo_config(
    run_name="mock_run",
    num_gpus=0,  # No GPUs needed
    for_mock=True,  # Force using mock config
    max_steps=5  # Short run for testing
)

# Create mock trainer
trainer = vf.get_mock_grpo_env_trainer(
    model="dummy",  # Model is not used
    env=env,
    reward_funcs=env.get_rubric(),
    args=args,
    train_dataset=small_dataset,
    eval_dataset=eval_dataset,
    openai_model="gpt-3.5-turbo"  # Specify OpenAI model
)

# Run the environment
results = trainer.train()
```

## Configuration Options

The mock trainer accepts these additional parameters:

```python
vf.get_mock_grpo_env_trainer(
    # Standard GRPOEnvTrainer parameters
    model,
    env,
    reward_funcs,
    args,
    train_dataset,
    eval_dataset,
    
    # Mock-specific parameters
    openai_model="gpt-3.5-turbo",  # OpenAI model to use
    temperature=0.7,               # Sampling temperature 
    top_p=0.9,                     # Top-p sampling parameter
    max_tokens=200                 # Max tokens to generate
)
```

## Mock Training Args

When using the mock trainer, create training args with:

```python
args = vf.get_default_grpo_config(
    run_name="mock_run",
    num_gpus=0,           # No GPUs needed for mock
    for_mock=True,        # Force using mock config
    max_steps=10          # Usually fewer steps for testing
)
```

## Example Scripts

We provide two example scripts that demonstrate the mock trainer:

### 1. Basic ToolEnv Example

See the example script in `verifiers/examples/mock_trainer_demo.py` for a demonstration with ToolEnv and calculator.

Run it with:

```bash
OPENAI_API_KEY=your_key_here python -m verifiers.examples.mock_trainer_demo
```

### 2. SmolaToolEnv Example

See the example script in `verifiers/examples/mock_smola_tool_demo.py` for a demonstration with SmolaToolEnv and a Python interpreter tool.

Run it with:

```bash
OPENAI_API_KEY=your_key_here python -m verifiers.examples.mock_smola_tool_demo
```

## Limitations

The mock trainer has these limitations:

1. No actual model training (parameters are not updated)
2. No gradient computation or optimization
3. No multi-GPU or distributed capabilities
4. No checkpointing or model saving

It's designed purely for environment testing, not for actual model training.

## Integration with Other Mocks

This mock trainer works seamlessly with the mock LLM implementation described in `README_mock_vllm.md`. Together, they provide a complete solution for testing environments on macOS without GPU dependencies.