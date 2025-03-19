import verifiers as vf
from verifiers.tools import calculator
from verifiers.prompts import CALCULATOR_FEW_SHOT
from verifiers.utils.logging_utils import setup_logging
import os
import logging

# Setup enhanced logging
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "gsm8k_calculator.log")
setup_logging(level="DEBUG", log_file=log_file)
logger = logging.getLogger("verifiers.examples.gsm8k_calculator")
logger.info("Starting GSM8K Calculator training script")

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
logger.info(f"Loading model: {model_name}")
model, tokenizer = vf.get_model_and_tokenizer(model_name)
logger.info("Model loading complete")

# Initialize tool environment for GSM8K
logger.info("Initializing GSM8K environment")
vf_env = vf.ToolEnv(
    dataset="gsm8k",
    few_shot=CALCULATOR_FEW_SHOT[0],
    tools=[calculator],
    max_steps=5
)
dataset = vf_env.get_dataset()
eval_dataset = vf_env.get_eval_dataset(n=100)
rubric = vf_env.get_rubric()
logger.info(f"Environment initialization complete. Dataset size: {len(dataset)}")

# notable defaults: lr = 1e-6, max_grad_norm = 0.01, constant lr 10 warmup steps, 1024 tokens in+out
run_name = "gsm8k-calc_" + model_name.split("/")[-1].lower()
logger.info(f"Using run name: {run_name}")
training_args = vf.get_default_grpo_config(
    run_name=run_name,
    num_gpus=8
)
# rollouts per prompt
training_args.num_generations = 7
# minibatch size per GPU ( bs 6 * 7 gpus / 7 rollouts -> 6 prompts per batch)
training_args.per_device_train_batch_size = 6
# batches to accumulate (6 prompts * 4 -> 32 prompts per global batch)
training_args.gradient_accumulation_steps = 4
# steps per global batch (1 on-policy, 1 off-policy)
training_args.num_iterations = 2
# no ref model
training_args.beta = 0.04
logger.debug(f"Training config: num_generations={training_args.num_generations}, "
             f"batch_size={training_args.per_device_train_batch_size}, "
             f"grad_accum_steps={training_args.gradient_accumulation_steps}, "
             f"num_iterations={training_args.num_iterations}")

# evals
#training_args.eval_strategy = "steps"
##training_args.eval_on_start = True
#training_args.eval_steps = 100
# training_args.per_device_eval_batch_size = 8
# training_args.eval_accumulation_steps = 1

logger.info("Creating GRPOEnvTrainer...")
try:
    trainer = vf.GRPOEnvTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=rubric,
        env=vf_env,
        args=training_args,
        train_dataset=dataset,
        #eval_dataset=eval_dataset,
    )
    logger.info("GRPOEnvTrainer initialization successful")
except Exception as e:
    logger.error(f"Error initializing GRPOEnvTrainer: {str(e)}", exc_info=True)
    raise

logger.info("Starting training...")
try:
    trainer.train()
    logger.info("Training completed successfully")
except Exception as e:
    logger.error(f"Error during training: {str(e)}", exc_info=True)
    raise 