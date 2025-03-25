from typing import List, Optional, Dict, Any
import os
import platform
import yaml

# Check if we're on macOS (Darwin)
IS_MACOS = platform.system() == 'Darwin'

try:
    from trl import GRPOConfig as TRLGRPOConfig
    REAL_CONFIG_AVAILABLE = True
except ImportError:
    REAL_CONFIG_AVAILABLE = False

# Create a simple mock config class for macOS systems
class MockGRPOConfig:
    """
    Simple class to mimic GRPOConfig from trl library when it's not available.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        # Ensure these are always set for compatibility
        for attr in ["per_device_train_batch_size", "max_steps", "logging_steps", 
                    "num_generations", "beta", "use_vllm", "run_name"]:
            if not hasattr(self, attr):
                setattr(self, attr, None)

def get_default_grpo_config(run_name: str,
                            num_gpus: int = 1,
                            reward_weights: Optional[List[float]] = None,
                            max_steps: int = 5000,
                            for_mock: bool = False) -> "GRPOConfig":
    """
    Create a default GRPO training configuration with common settings.
    
    Args:
        run_name: Name for the training run
        num_gpus: Number of GPUs to use
        reward_weights: Weights for multiple reward functions
        max_steps: Maximum training steps
        for_mock: Force using the mock config even when real one is available
        
    Returns:
        GRPOConfig: A configuration object for GRPO training
    """
    # On macOS or when mocking is requested, use our mock config
    if IS_MACOS or for_mock or not REAL_CONFIG_AVAILABLE:
        return MockGRPOConfig(
            output_dir=f"outputs/{run_name}",
            run_name=run_name,
            learning_rate=1e-6,
            warmup_steps=20,
            max_grad_norm=0.1,
            num_iterations=1, 
            beta=0.04,
            max_tokens=1024,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_generations=(2 * num_gpus - 2 if num_gpus > 1 else 2),
            gradient_accumulation_steps=int(16 / num_gpus) if num_gpus > 0 else 1,
            save_strategy="no",
            logging_steps=1,
            log_completions=True,
            report_to=["none"],
            reward_weights=reward_weights,
            use_vllm=True,
            max_steps=max_steps
        )
    else:
        # Use the real GRPOConfig from trl
        return TRLGRPOConfig(
            output_dir=f"outputs/{run_name}",
            run_name=run_name,
            learning_rate=1e-6,
            lr_scheduler_type="constant_with_warmup",
            warmup_steps=20,
            num_train_epochs=1,
            bf16=True,
            adam_beta1=0.9,
            adam_beta2=0.99,
            max_grad_norm=0.1,
            num_iterations=1,
            beta=0.04,
            max_prompt_length=1024,
            max_completion_length=1024,
            per_device_train_batch_size=2,
            num_generations=(2 * num_gpus - 2 if num_gpus > 1 else 2),
            gradient_accumulation_steps=int(16 / num_gpus),
            gradient_checkpointing=True,
            save_strategy="steps",
            save_steps=100,
            save_only_model=True,
            use_vllm=True,
            vllm_device=f"cuda:{num_gpus-1}",
            vllm_gpu_memory_utilization=0.7 if num_gpus > 1 else 0.3,
            logging_steps=1,
            log_on_each_node=False,
            log_completions=True,
            report_to="wandb",
            reward_weights=reward_weights,
            max_steps=max_steps
        )

# Use MockGRPOConfig as the GRPOConfig type on macOS
GRPOConfig = MockGRPOConfig if IS_MACOS or not REAL_CONFIG_AVAILABLE else TRLGRPOConfig


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        A dictionary containing the configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config

