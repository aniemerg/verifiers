# Mock vLLM Implementation for OpenAI API

This document describes how to use the mock vLLM implementation that works with the OpenAI API in the `verifiers` package. This mock allows you to use the `MultiStepEnv` environment with OpenAI models instead of vLLM, which is particularly useful on macOS or other systems where vLLM may not be installable.

## Prerequisites

The mock implementation requires these additional packages:

```bash
pip install openai tiktoken
```

## Basic Usage

Here's how to use the mock LLM with any existing `MultiStepEnv` environment:

```python
import os
from openai import OpenAI
from verifiers.imports import LLM, SamplingParams

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize mock LLM
llm = LLM(
    openai_client=client,
    model="gpt-3.5-turbo",  # Can be changed to any OpenAI model
    max_workers=5           # Limit concurrent requests
)

# Create sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=200
)

# Use with ToolEnv which extends MultiStepEnv
from verifiers.envs.tool_env import ToolEnv
from verifiers.tools.calculator import calculator
from verifiers.prompts.few_shots import CALCULATOR_FEW_SHOT

# Initialize ToolEnv with calculator tool
env = ToolEnv(
    tools=[calculator],
    few_shot=CALCULATOR_FEW_SHOT[0],
    max_steps=3
)

# Create a prompt with a math problem
prompt = [
    {"role": "system", "content": env.system_prompt},
    {"role": "user", "content": "What is 25 * 17?"}
]

# Generate completions
result = env.generate([prompt], llm, sampling_params)
```

## Features

The mock LLM implementation provides:

1. **Compatible Interface**: Implements the same interface as vLLM's LLM class
2. **OpenAI API Integration**: Uses the OpenAI API for generating responses
3. **Tokenization**: Handles token counting with tiktoken
4. **Parameter Mapping**: Maps vLLM SamplingParams to OpenAI API parameters
5. **Batch Processing**: Supports parallel processing of multiple requests
6. **Retry Logic**: Implements exponential backoff for API failures
7. **Error Handling**: Robust error handling for API issues

## Advanced Configuration

The LLM constructor accepts these parameters:

```python
LLM(
    openai_client=None,          # Existing OpenAI client or None to create new one
    model="gpt-3.5-turbo",       # OpenAI model name
    model_name=None,             # Alternative parameter name for compatibility
    api_key=None,                # OpenAI API key if not using client
    tokenizer_model=None,        # Tiktoken model name
    max_workers=10,              # Maximum concurrent threads
    retry_attempts=3,            # Number of retry attempts
    retry_delay=1.0              # Base delay between retries
)
```

## Testing

You can run the test script to verify the implementation works correctly:

```bash
OPENAI_API_KEY=your_key_here python tests/test_mock_vllm.py
```

## Example Script

See the example script in `verifiers/examples/openai_multistep_demo.py` for a complete demonstration.

Run it with:

```bash
OPENAI_API_KEY=your_key_here python -m verifiers.examples.openai_multistep_demo
```

## API-Based Evaluation

Even without using the mock LLM, all environments extending `MultiStepEnv` support direct API-based evaluation:

```python
from openai import OpenAI
from verifiers.envs.simple_env import SimpleEnv

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
env = SimpleEnv()

# Load your evaluation dataset
env.get_eval_dataset()

# Run evaluation
rewards = env.eval_api(
    client=client,
    model="gpt-3.5-turbo",
    max_concurrent=5,
    timeout=60
)
```