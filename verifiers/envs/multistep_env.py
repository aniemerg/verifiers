from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
import random
import time
from typing import List, Dict, Sequence, Any, Union, Tuple

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc
from trl.data_utils import maybe_apply_chat_template
from ..imports import LLM, SamplingParams, VLLMClient  # type: ignore

from verifiers.envs.environment import Environment


class MultiStepEnv(Environment):
    def __init__(self,
                 system_prompt: str = "",
                 few_shot: List[Dict[str, str]] = [],
                 sampling_args: Dict[str, Any] = {},
                 mask_env_response: bool = True,
                 max_workers: int = 10,
                 max_steps: int = 10,
                 sleep_time: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = system_prompt
        self.few_shot = few_shot
        self.sampling_args = {
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "n": 1
        }
        self.sampling_args.update(sampling_args)
        self.env_mask = 0 if mask_env_response else 1
        self.max_workers = max_workers
        self.sleep_time = sleep_time
        self.max_steps = max_steps
    def get_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    def get_eval_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    @abstractmethod
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        pass

    @abstractmethod
    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        pass

    @abstractmethod
    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        pass

    def step(self,
             states: List[Dict[str, Any]],
             vllm_client: Union[VLLMClient, LLM],
             temperature: float = 1.0,
             top_p: float = 1.0,
             top_k: int = -1,
             min_p: float = 0.0,
             repetition_penalty: float = 1.0,
             max_tokens: int = 100,
             n: int = 1,
             **kwargs: Any) -> List[Dict[str, Any]]:
        
        live_indices = [i for i, s in enumerate(states) if not s["completed"]]
        messages_to_step = [states[i]["messages"] for i in live_indices]
        
        # Get the tokenizer from kwargs
        tokenizer = kwargs.get('tokenizer', None)
        if tokenizer is None:
            raise ValueError("Tokenizer is required for step method")
        
        # Check if we're using VLLMClient or the old LLM interface
        if isinstance(vllm_client, VLLMClient):
            # Convert messages to formatted prompt text for VLLMClient
            prompt_texts = []
            for messages in messages_to_step:
                formatted = {"prompt": messages}
                text = maybe_apply_chat_template(formatted, tokenizer)["prompt"]
                prompt_texts.append(text)
            
            # Generate using VLLMClient.generate()
            try:
                completion_ids_list = vllm_client.generate(
                    prompts=prompt_texts,
                    n=n,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                )
                
                def update_state(j, prompt_text, completion_ids):
                    # sleep for 0-1 seconds to avoid rate limiting
                    time.sleep(self.sleep_time * random.random())
                    
                    state = states[j].copy()
                    
                    # Get or create the prompt token IDs
                    if len(state["prompt_ids"]) == 0:
                        # Tokenize the prompt text to get token IDs
                        state["prompt_ids"] = tokenizer.encode(prompt_text)
                    
                    # Decode the completion token IDs to text
                    completion_text = tokenizer.decode(completion_ids)
                    
                    # Add the assistant message
                    state["messages"].append({"role": "assistant", "content": completion_text})
                    
                    # Calculate token lengths
                    total_prev_len = len(state["prompt_ids"]) + len(state["completion_ids"])
                    
                    # Check if we're done or need to add environment response
                    if self.is_completed(state["messages"]) or len(completion_ids) > max_tokens:
                        state["completed"] = True
                        
                        # Update completion IDs and mask
                        state["completion_ids"] = completion_ids[:max_tokens]
                        state["completion_mask"] = [1] * len(state["completion_ids"])
                    else:
                        # Add environment response
                        env_response = self.env_response(state["messages"])
                        state["messages"].append(env_response)
                        
                        # Tokenize the env response to get its length
                        env_response_ids = tokenizer.encode(env_response["content"])
                        
                        # Update completion IDs and mask
                        state["completion_ids"] = completion_ids
                        state["completion_mask"] = [1] * len(completion_ids)
                        state["completion_mask"].extend([self.env_mask] * len(env_response_ids))
                        state["completion_ids"].extend(env_response_ids)
                    
                    # Validate that mask and IDs have the same length
                    if len(state["completion_mask"]) != len(state["completion_ids"]):
                        print(state["messages"])
                        print(state["completion_mask"])
                        print(state["completion_ids"])
                        raise ValueError(f"Completion mask and completion ids are not the same length for state {j}")
                    
                    return j, state
                
                # Process results in parallel
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    results = list(executor.map(
                        lambda args: update_state(*args),
                        [(j, prompt_texts[i], completion_ids_list[i]) for i, j in enumerate(live_indices)]
                    ))
                
            except Exception as e:
                # Log error but don't add complex recovery logic
                print(f"Error during VLLMClient.generate: {e}")
                raise
        else:
            # Legacy LLM.chat() interface
            # Create a SamplingParams object
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
                n=n
            )
            
            # Use the old chat interface
            llm_responses = vllm_client.chat(messages_to_step, sampling_params=sampling_params, use_tqdm=False)
            
            def update_state(j, llm_response):
                # sleep for 0-1 seconds to avoid rate limiting
                time.sleep(self.sleep_time * random.random())
                
                state = states[j].copy()
                if len(state["prompt_ids"]) == 0:
                    state["prompt_ids"] = llm_response.prompt_token_ids
                state["messages"].append({"role": "assistant", "content": llm_response.outputs[0].text})
            
                # get token lengths of env response and new completion
                total_prev_len = len(state["prompt_ids"]) + len(state["completion_ids"])
                env_response_len = len(list(llm_response.prompt_token_ids)) - total_prev_len
                new_completion_len = len(llm_response.outputs[0].token_ids)
                
                # update completion masks
                state["completion_mask"].extend([self.env_mask] * env_response_len)
                state["completion_mask"].extend([1] * new_completion_len)
                
                # update completion ids
                state["completion_ids"] = list(llm_response.prompt_token_ids)
                state["completion_ids"].extend(list(llm_response.outputs[0].token_ids))
                state["completion_ids"] = state["completion_ids"][len(state["prompt_ids"]):]
                
                if self.is_completed(state["messages"]) or len(state["completion_ids"]) > max_tokens:
                    state["completed"] = True
                    state["completion_ids"] = state["completion_ids"][:max_tokens]
                    state["completion_mask"] = state["completion_mask"][:len(state["completion_ids"])]
                else:
                    state["messages"].append(self.env_response(state["messages"]))
                
                if not len(state["completion_mask"]) == len(state["completion_ids"]):
                    print(state["messages"])
                    print(state["completion_mask"])
                    print(state["completion_ids"])
                    raise ValueError(f"Completion mask and completion ids are not the same length for state {j}")
                
                return j, state
            
            # Process results in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(
                    lambda args: update_state(*args),
                    [(j, llm_responses[i]) for i, j in enumerate(live_indices)]
                ))
        
        # Update states with results
        for j, state in results:
            states[j] = state
        
        return states

    def generate(self, 
                 prompts: List[List[Dict[str, Any]]],
                 vllm_client: Union[VLLMClient, LLM],
                 sampling_params: SamplingParams = None,
                 **kwargs: Any) -> Dict[str, List[Sequence[int]] | List[str] | List[List[Dict[str, Any]]]]:
        
        # Extract parameters from SamplingParams or use defaults
        if sampling_params:
            temperature = getattr(sampling_params, 'temperature', 1.0)
            top_p = getattr(sampling_params, 'top_p', 1.0)
            top_k = getattr(sampling_params, 'top_k', -1)
            min_p = getattr(sampling_params, 'min_p', 0.0)
            repetition_penalty = getattr(sampling_params, 'repetition_penalty', 1.0)
            max_tokens = getattr(sampling_params, 'max_tokens', 100)
            n = getattr(sampling_params, 'n', 1)
        else:
            temperature = kwargs.get('temperature', 1.0)
            top_p = kwargs.get('top_p', 1.0)
            top_k = kwargs.get('top_k', -1)
            min_p = kwargs.get('min_p', 0.0)
            repetition_penalty = kwargs.get('repetition_penalty', 1.0)
            max_tokens = kwargs.get('max_tokens', 100)
            n = kwargs.get('n', 1)
        
        # Apply any custom sampling args
        for k, v in self.sampling_args.items():
            if k == 'temperature': temperature = v
            elif k == 'top_p': top_p = v
            elif k == 'top_k': top_k = v
            elif k == 'min_p': min_p = v
            elif k == 'repetition_penalty': repetition_penalty = v
            elif k == 'max_tokens': max_tokens = v
            elif k == 'n': n = v
        
        # Initialize state variables
        all_completed = False
        states = [{
            "messages": m,
            "prompt_messages": len(m),
            "prompt_ids": [],
            "completed": False,
            "completion_ids": [],
            "completion_mask": []
        } for m in prompts]
        
        # Main loop
        while not all_completed:
            states = self.step(
                states, 
                vllm_client, 
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
                n=n,
                **kwargs
            )
            all_completed = all(state["completed"] for state in states)
        
        # Return the completions in the expected format
        completion_messages = [s["messages"][s["prompt_messages"]:] for s in states]
        completion_ids = [s["completion_ids"] for s in states]
        completion_mask = [s["completion_mask"] for s in states]
        
        return {
            "ids": completion_ids,
            "messages": completion_messages,
            "mask": completion_mask
        }

    def step_api(self, 
             client: Any,
             model: str,
             messages: List[Dict[str, str]],
             **kwargs: Any) -> Tuple[List[Dict[str, str]], bool]:
        """
        Execute a single step using OpenAI API, including environment response if needed.
        
        Args:
            client: OpenAI client instance
            messages: Conversation history
            model: Model name to use
            **kwargs: Additional arguments for the chat completion API
        
        Returns:
            Updated messages list with assistant response and possibly environment response
        """
        messages_copy = messages.copy()
        
        try:            
            # Get assistant response
            response = client.chat.completions.create(
                model=model,
                messages=messages_copy,
            )
            
            # Add assistant response to messages
            assistant_msg = {
                "role": "assistant", 
                "content": response.choices[0].message.content
            }
            messages_copy.append(assistant_msg)
            
            # Check if we're done
            if self.is_completed(messages_copy):
                rollout_is_completed = True
            else:
                rollout_is_completed = False
                # If not done, get and add environment response
                env_msg = self.env_response(messages_copy)
                messages_copy.append(env_msg)
            
            return messages_copy, rollout_is_completed
            
        except Exception as e:
            # Handle errors by adding error message and returning
            error_msg = {"role": "assistant", "content": f"Error in API call: {str(e)}"}
            messages_copy.append(error_msg)
            return messages_copy, True
    
    def eval_api(self, 
                client: Any,
                model: str,
                max_concurrent: int = 32,
                timeout: int = 60,
                sampling_args: Dict[str, Any] = {},
                **kwargs: Any):
        """
        Evaluate model using OpenAI API with proper concurrency.
        
        Args:
            client: OpenAI client instance
            model: Model name as string
            max_concurrent: Maximum number of concurrent API calls
            timeout: Maximum seconds to wait for each example
            sampling_args: Arguments specific to sampling (separate from env sampling_args)
            **kwargs: Additional arguments for evaluation
        
        Returns:
            Tuple of (eval_dataset, rewards)
        """
        def run_evaluation():
            # Import libraries here to avoid requiring them for normal operation
            import asyncio
            from asyncio import Semaphore
            # Get the evaluation dataset
            if self.eval_dataset is None:
                self.eval_dataset = self.get_eval_dataset(**kwargs)
                
            if self.eval_dataset is None:
                raise ValueError("Failed to load evaluation dataset")
            
            eval_dataset = self.eval_dataset
            
            async def process_example(example, semaphore):
                async with semaphore:
                    # Initialize conversation with system prompt and few-shot examples
                    prompt = example["prompt"]
                    messages = example["prompt"].copy()
                    answer = example["answer"]
                    
                    # Save the length of initial messages to extract just the interaction part later
                    initial_length = len(messages)

                    # Run the conversation loop until completion or max steps
                    for _ in range(self.max_steps):  # Safety limit on conversation turns
                        try:
                            # Run step_api to get model and environment response
                            # Note: step_api now returns a tuple (messages, is_completed)
                            step_result = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: self.step_api(
                                    client=client,
                                    model=model,
                                    messages=messages,
                                    **sampling_args
                                )
                            )
                            
                            # Unpack the step_api result
                            messages, is_completed = step_result
                            
                            # If the rollout is completed, break the loop
                            if is_completed:
                                break
                            
                        except Exception as e:
                            print(f"Error processing example {example.get('id', 'unknown')}: {str(e)}")
                            break
                    
                    # Extract only the interaction part (not system/few-shot)
                    completions = messages[initial_length:]
                    
                    return {
                        "prompt": prompt,
                        "completions": completions,
                        "answer": answer
                    }
            
            async def run_all_examples():
                # Create semaphore for concurrency control
                from tqdm.asyncio import tqdm_asyncio

                semaphore = Semaphore(max_concurrent)
                
                # Process all examples concurrently
                tasks = [process_example(example, semaphore) for example in eval_dataset]
                results = await tqdm_asyncio.gather(
                    *tasks,
                    total=len(eval_dataset),
                    desc=f"Evaluating {len(eval_dataset)} examples"
                )
                
                return results
            
            # Run the async evaluation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(run_all_examples())
            finally:
                loop.close()
            
            # Calculate rewards
            results_prompt = [result["prompt"] for result in results]
            results_answer = [result["answer"] for result in results]
            results_completions = [result["completions"] for result in results]
            results = {"prompt": results_prompt, "answer": results_answer, "completions": results_completions}
            
            reward_funcs = self.get_rubric()
            rewards = {}
            
            for reward_func in reward_funcs:
                func_rewards = reward_func(**results) # type: ignore
                func_reward_avg = sum(func_rewards) / len(func_rewards)
                func_name = reward_func.__name__ # type: ignore
                print(f"{func_name}: {func_reward_avg}")
                rewards[func_name] = func_reward_avg
            
            return rewards
            
        # Run the evaluation function
        return run_evaluation()
    

    