"""
Mock implementation of GRPOEnvTrainer that works without GPU dependencies.
Designed for environment testing only, not actual training.
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Union, Sequence

import numpy as np
from datasets import Dataset, IterableDataset
from openai import OpenAI

from verifiers.imports import LLM, SamplingParams
from verifiers.envs.environment import Environment


class MockGRPOEnvTrainer:
    """
    A mock implementation of GRPOEnvTrainer that works without GPU dependencies.
    Designed for environment testing only, not actual training.
    
    This class mimics the interface of GRPOEnvTrainer but uses our OpenAI mock LLM
    for generation instead of requiring GPU-based models.
    """
    
    def __init__(
            self,
            model: Union[str, Any],  # We won't use the actual model
            env: Environment,
            reward_funcs: Union[Callable, List[Callable]],
            args: Optional[Any] = None,  # We'll ignore most args
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            processing_class: Optional[Any] = None,  # Tokenizer, but we won't use it 
            **kwargs
    ):
        """
        Initialize the mock trainer.
        
        Args:
            model: Model identifier or object (not used in mock)
            env: Environment instance 
            reward_funcs: Reward function(s) to use
            args: Training arguments (only a few are used)
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            processing_class: Tokenizer (not used in mock)
            **kwargs: Additional arguments
        """
        # Check for OpenAI API key
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable must be set to use MockGRPOEnvTrainer"
            )
        
        # Store the environment and reward functions
        self.env = env
        self.reward_funcs = [reward_funcs] if callable(reward_funcs) and not isinstance(reward_funcs, list) else reward_funcs
        
        # Create OpenAI client and mock LLM
        self.client = OpenAI(api_key=self.api_key)
        self.llm = LLM(
            openai_client=self.client,
            model=kwargs.get("openai_model", "gpt-3.5-turbo")  # Allow model override
        )
        
        # Create sampling parameters
        self.sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            max_tokens=kwargs.get("max_tokens", 200)
        )
        
        # Store datasets for testing
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Tracking
        self.epoch = 0
        self.global_step = 0
        
        # Configuration options (simplified from args)
        if args:
            self.batch_size = getattr(args, "per_device_train_batch_size", 4)
            self.num_generations = getattr(args, "num_generations", 1)
            self.eval_batch_size = getattr(args, "per_device_eval_batch_size", self.batch_size)
            self.logging_steps = getattr(args, "logging_steps", 1)
            self.eval_steps = getattr(args, "eval_steps", 10)
            self.max_steps = getattr(args, "max_steps", 100)
            self.gradient_accumulation_steps = getattr(args, "gradient_accumulation_steps", 1)
            self.max_grad_norm = getattr(args, "max_grad_norm", 1.0)
            self.learning_rate = getattr(args, "learning_rate", 1e-6)
            self.run_name = getattr(args, "run_name", "mock_run")
        else:
            self.batch_size = 4
            self.num_generations = 1
            self.eval_batch_size = 4
            self.logging_steps = 1
            self.eval_steps = 10
            self.max_steps = 100
            self.gradient_accumulation_steps = 1
            self.max_grad_norm = 1.0
            self.learning_rate = 1e-6
            self.run_name = "mock_run"
        
        # For print formatting
        try:
            from rich.console import Console
            from rich.table import Table
            self._rich_available = True
            self.console = Console()
            self.Table = Table  # Store Table class reference
        except ImportError:
            self._rich_available = False
    
    def _get_batch(self, dataset, batch_size, start_idx=0):
        """Get a batch of examples from a dataset."""
        if dataset is None:
            return None
        
        # Ensure we're working with valid indexes
        dataset_size = len(dataset)
        start_idx = min(start_idx, dataset_size - 1)
        
        # Calculate end index, ensuring we don't exceed dataset size
        end_idx = min(start_idx + batch_size, dataset_size)
        
        # Handle wrap-around if necessary
        if start_idx >= dataset_size - 1 or start_idx >= end_idx:
            start_idx = 0
            end_idx = min(batch_size, dataset_size)
        
        # Select the batch using the correct range
        indexes = list(range(start_idx, end_idx))
        batch = dataset.select(indexes)
        
        # Extract prompts and additional fields
        prompts = [example["prompt"] for example in batch]
        
        additional_fields = {}
        for key in batch.features.keys():
            if key != "prompt":
                additional_fields[key] = [example[key] for example in batch]
        
        return {
            "prompts": prompts,
            "additional_fields": additional_fields,
            "next_idx": end_idx
        }
    
    def _print_results(self, step, batch_rewards, avg_reward):
        """Print training results in a formatted way."""
        if self._rich_available:
            table = self.Table(title=f"Results at Step {step}")
            table.add_column("Reward Function", justify="right", style="cyan")
            table.add_column("Average Reward", justify="center", style="green")
            
            # Add overall average first
            table.add_row("Overall", f"{avg_reward:.4f}")
            
            # Add individual reward functions
            for i, func in enumerate(self.reward_funcs):
                name = getattr(func, "__name__", f"reward_func_{i}")
                avg = batch_rewards[i]
                table.add_row(name, f"{avg:.4f}")
            
            self.console.print(table)
        else:
            print(f"Step {step} - Average Reward: {avg_reward:.4f}")
            for i, func in enumerate(self.reward_funcs):
                name = getattr(func, "__name__", f"reward_func_{i}")
                avg = batch_rewards[i]
                print(f"  {name}: {avg:.4f}")
    
    def _print_completion_sample(self, prompt, completion, reward):
        """Print a sample prompt and completion."""
        if self._rich_available:
            try:
                # Import rich components at function level to avoid issues
                from rich.panel import Panel as RichPanel
                from rich.text import Text as RichText
                # Format prompt content for display - only show the user's query, not previous context
                
                if isinstance(prompt, list) and len(prompt) > 0:
                    # Find the last user message that isn't part of a few-shot example
                    prompt_text = prompt[-1].get("content", "")
                elif isinstance(prompt, str):
                    prompt_text = prompt
                else:
                    prompt_text = str(prompt)
                
                # Format completion for display
                completion_text = ""
                if isinstance(completion, list):
                    for msg in completion:
                        if isinstance(msg, dict):
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            completion_text += f"{role}: {content}\n"
                elif isinstance(completion, str):
                    completion_text = completion
                else:
                    completion_text = str(completion)
                
                # Create panels
                prompt_panel = RichPanel(
                    RichText(prompt_text, style="blue"), 
                    title="Prompt",
                    expand=False
                )
                completion_panel = RichPanel(
                    RichText(completion_text, style="green"), 
                    title="Completion",
                    expand=False
                )
                
                # Print
                self.console.print(prompt_panel)
                self.console.print(completion_panel)
                self.console.print(f"Reward: {reward:.4f}")
            except Exception as e:
                # Fallback to simple printing if rich formatting fails
                print(f"(Rich formatting error: {str(e)})")
                print("Sample:")
                print("Prompt:", prompt)
                print("Completion:", completion)
                print("Reward:", reward)
        else:
            print("Sample:")
            print("Prompt:", prompt)
            print("Completion:", completion)
            print("Reward:", reward)
    
    def train(self, reward_fn_kwargs=None):
        """
        Mock training method that runs batches through the environment.
        
        Args:
            reward_fn_kwargs: Optional dictionary of additional arguments to pass to reward functions.
                This matches the real GRPOTrainer interface.
                
        Returns:
            Dict with training metrics
        """
        if not self.train_dataset:
            print("No dataset provided, nothing to train on")
            return {}
        
        # Get reward_fn_kwargs or initialize as empty dict if not provided
        reward_fn_kwargs = reward_fn_kwargs or {}
        
        print(f"\n=== Starting mock training with {len(self.train_dataset)} examples ===")
        print(f"Batch size: {self.batch_size}, Num generations: {self.num_generations}")
        print(f"Running for {self.max_steps} steps\n")
        
        start_time = time.time()
        self.global_step = 0
        batch_idx = 0
        
        metrics = {
            "train_runtime": 0,
            "train_steps_per_second": 0,
            "train_loss": 0,
            "epoch": 0,
            "reward": []
        }
        
        try:
            # Main training loop
            for step in range(self.max_steps):
                self.global_step = step + 1
                
                # Get batch of examples
                batch = self._get_batch(self.train_dataset, self.batch_size, batch_idx)
                batch_idx = batch["next_idx"]
                prompts = batch["prompts"]
                
                # Run generation through the environment
                results = self.env.generate(prompts, self.llm, self.sampling_params)
                
                # Calculate rewards
                all_rewards = []
                avg_per_func = []
                
                for reward_func in self.reward_funcs:
                    # Combine batch additional fields with reward_fn_kwargs
                    kwargs = batch["additional_fields"].copy()
                    kwargs.update(reward_fn_kwargs)
                    
                    # Calculate rewards with combined kwargs
                    rewards = reward_func(prompts=prompts, completions=results["messages"], **kwargs)
                    all_rewards.append(rewards)
                    avg_per_func.append(sum(rewards) / len(rewards))
                
                # Calculate average reward
                avg_reward = sum(sum(rewards) for rewards in all_rewards) / len(all_rewards) / len(prompts)
                metrics["reward"].append(avg_reward)
                
                # Log progress
                if step % self.logging_steps == 0 or step == self.max_steps - 1:
                    self._print_results(step + 1, avg_per_func, avg_reward)
                    
                    # Print a sample
                    if len(prompts) > 0:
                        idx = 0  # First example in batch
                        self._print_completion_sample(
                            prompts[idx], 
                            results["messages"][idx],
                            all_rewards[0][idx]  # Use first reward func
                        )
                
                # Run evaluation occasionally
                if step % self.eval_steps == 0 and step > 0:
                    eval_results = self.evaluate(reward_fn_kwargs=reward_fn_kwargs)
                
                # Simulate a brief pause for realism
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nTraining interrupted.")
        
        # Finalize metrics
        train_time = time.time() - start_time
        metrics["train_runtime"] = train_time
        metrics["train_steps_per_second"] = self.global_step / train_time
        metrics["epoch"] = 1.0
        
        print(f"\n=== Training completed: {self.global_step} steps in {train_time:.2f} seconds ===")
        print(f"Final average reward: {metrics['reward'][-1] if metrics['reward'] else 0:.4f}")
        
        return metrics
    
    def evaluate(self, reward_fn_kwargs=None):
        """
        Run evaluation on the eval dataset.
        
        Args:
            reward_fn_kwargs: Optional dictionary of additional arguments to pass to reward functions.
                This matches the real GRPOTrainer interface.
                
        Returns:
            Dict with evaluation metrics
        """
        if not self.eval_dataset:
            print("No evaluation dataset provided, skipping evaluation")
            return {}
        
        print(f"\n--- Running evaluation with {len(self.eval_dataset)} examples ---")
        
        metrics = {"eval_reward": 0.0}  # Initialize with a float instead of list
        batch_idx = 0
        
        try:
            # Get batch of examples
            batch = self._get_batch(self.eval_dataset, self.eval_batch_size, batch_idx)
            if not batch:
                print("No evaluation batch could be created")
                return metrics
                
            prompts = batch["prompts"]

            # Run generation through the environment
            results = self.env.generate(prompts, self.llm, self.sampling_params)
            
            # Calculate rewards
            all_rewards = []
            avg_per_func = []
            
            for reward_func in self.reward_funcs:
                # Combine batch additional fields with reward_fn_kwargs
                kwargs = batch["additional_fields"].copy()
                if reward_fn_kwargs:
                    kwargs.update(reward_fn_kwargs)
                
                rewards = reward_func(prompts=prompts, completions=results["messages"], **kwargs)
                all_rewards.append(rewards)
                avg_per_func.append(sum(rewards) / len(rewards) if rewards else 0.0)
            
            # Calculate average reward
            if all_rewards and all(rewards for rewards in all_rewards):
                avg_reward = sum(sum(rewards) for rewards in all_rewards) / len(all_rewards) / len(prompts)
                metrics["eval_reward"] = avg_reward
            else:
                avg_reward = 0.0
                metrics["eval_reward"] = 0.0
            
            # Print results
            if self._rich_available:
                table = self.Table(title="Evaluation Results")
                table.add_column("Reward Function", justify="right", style="cyan")
                table.add_column("Average Reward", justify="center", style="green")
                
                # Add overall average
                table.add_row("Overall", f"{avg_reward:.4f}")
                
                # Add individual reward functions
                for i, func in enumerate(self.reward_funcs):
                    name = getattr(func, "__name__", f"reward_func_{i}")
                    avg = avg_per_func[i]
                    table.add_row(name, f"{avg:.4f}")
                
                self.console.print(table)
            else:
                print(f"Evaluation Results:")
                for i, func in enumerate(self.reward_funcs):
                    name = getattr(func, "__name__", f"reward_func_{i}")
                    avg = avg_per_func[i]
                    print(f"  {name}: {avg:.4f}")
                print(f"Overall: {avg_reward:.4f}")
            
            # Print a sample from evaluation
            if len(prompts) > 0:
                print("\nEvaluation Sample:")
                self._print_completion_sample(
                    prompts[0], 
                    results["messages"][0],
                    all_rewards[0][0] if all_rewards and all_rewards[0] else 0.0
                )
                
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            
        return metrics
    
    def create_optimizer(self):
        """Mock method for compatibility."""
        pass
    
    def save_model(self, output_dir=None):
        """Mock method for compatibility."""
        pass
    
    def log(self, logs):
        """Mock method for compatibility."""
        pass