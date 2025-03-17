"""
GSM8K example using SmolaAgents PythonInterpreterTool.
"""

import verifiers as vf
from verifiers.prompts import SMOLA_CALCULATOR_FEW_SHOT

# Note: You need to have SmolaAgents installed with:
# $ uv pip install -e ./wip/smolagents

try:
    from smolagents.default_tools import PythonInterpreterTool
except ImportError:
    raise ImportError(
        "SmolaAgents is required for this example. "
        "Please install it with: uv pip install -e ./wip/smolagents"
    )

# Model configuration
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

# Create a SmolaAgents calculator tool
calculator_tool = PythonInterpreterTool(
    authorized_imports=["math"],  # Customize available imports
)

# Initialize SmolaToolEnv
vf_env = vf.SmolaToolEnv(
    dataset="gsm8k",
    few_shot=SMOLA_CALCULATOR_FEW_SHOT[0],
    tools=[calculator_tool],
    max_steps=5
)
dataset = vf_env.get_dataset()
eval_dataset = vf_env.get_eval_dataset(n=100)
rubric = vf_env.get_rubric()

# Training configuration - similar to gsm8k_calculator.py
run_name = "gsm8k-smola-calc_" + model_name.split("/")[-1].lower()
training_args = vf.get_default_grpo_config(
    run_name=run_name,
    num_gpus=1
)
# rollouts per prompt
training_args.num_generations = 2
# minibatch size per GPU (bs 6 * 7 gpus / 7 rollouts -> 6 prompts per batch)
training_args.per_device_train_batch_size = 2
# batches to accumulate (6 prompts * 4 -> 32 prompts per global batch)
training_args.gradient_accumulation_steps = 4
# steps per global batch (1 on-policy, 1 off-policy)
training_args.num_iterations = 2
# no ref model
training_args.beta = 0.04

# Create trainer
trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric,
    env=vf_env,
    args=training_args,
    train_dataset=dataset,
    #eval_dataset=eval_dataset,
)

# Start training
if __name__ == "__main__":
    trainer.train()
