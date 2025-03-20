"""Test the mock vLLM implementation with OpenAI API.
"""
import sys
import os
from typing import List, Dict, Any

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from verifiers.imports import LLM, SamplingParams
from verifiers.envs.multistep_env import MultiStepEnv
from openai import OpenAI


def test_mock_llm_creation():
    """Test creating a mock LLM instance."""
    # Create a mock LLM without OpenAI client (will use environment API key)
    llm = LLM(model="gpt-3.5-turbo")
    
    # Verify attributes
    assert llm.model == "gpt-3.5-turbo"
    assert hasattr(llm, "tokenizer")
    assert hasattr(llm, "client")
    
    print("✓ Mock LLM creation test passed")
    return llm


def test_encode_messages():
    """Test encoding of messages to tokens."""
    llm = LLM(model="gpt-3.5-turbo")
    
    # Test message encoding
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    token_ids = llm._encode_messages(messages)
    
    # Verify we got some tokens
    assert isinstance(token_ids, list)
    assert len(token_ids) > 0
    assert all(isinstance(t, int) for t in token_ids)
    
    print(f"✓ Message encoding test passed, got {len(token_ids)} tokens")
    return token_ids


def test_chat_method():
    """Test the chat method of the mock LLM."""
    # You'll need a valid OpenAI API key for this test
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠ Skipping chat test: OPENAI_API_KEY not set in environment")
        return None
    
    client = OpenAI(api_key=api_key)
    llm = LLM(openai_client=client, model="gpt-3.5-turbo")
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=50
    )
    
    # Create a test message
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    # Call chat method
    responses = llm.chat([messages], sampling_params)
    
    # Verify response
    assert len(responses) == 1
    assert hasattr(responses[0], "prompt_token_ids")
    assert hasattr(responses[0], "outputs")
    assert len(responses[0].outputs) > 0
    assert hasattr(responses[0].outputs[0], "text")
    assert hasattr(responses[0].outputs[0], "token_ids")
    
    # Print response
    print(f"✓ Chat test passed, got response: {responses[0].outputs[0].text[:50]}...")
    return responses[0]


class SimpleTestEnv(MultiStepEnv):
    """A simple test environment for MultiStepEnv."""
    
    def __init__(self, **kwargs):
        super().__init__(system_prompt="You are a helpful assistant.", **kwargs)
    
    def get_rubric(self, **kwargs):
        """Return a dummy reward function."""
        def dummy_reward(**kwargs):
            return [1.0]  # Always return 1.0
        
        return [dummy_reward]
    
    def is_completed(self, messages, **kwargs):
        """Check if conversation is complete."""
        # Complete after 1 assistant message
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        return len(assistant_messages) >= 1
    
    def env_response(self, messages, **kwargs):
        """Generate environment response."""
        # Simple acknowledgment response
        return {"role": "user", "content": "Thank you for your response."}


def test_multistep_env():
    """Test the MultiStepEnv with our mock LLM."""
    # You'll need a valid OpenAI API key for this test
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠ Skipping MultiStepEnv test: OPENAI_API_KEY not set in environment")
        return None
    
    # Create test environment
    test_env = SimpleTestEnv(max_steps=2)
    
    # Create LLM
    client = OpenAI(api_key=api_key)
    llm = LLM(openai_client=client, model="gpt-3.5-turbo")
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100
    )
    
    # Create test prompt
    prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    # Call generate method
    result = test_env.generate([prompt], llm, sampling_params)
    
    # Verify result
    assert "messages" in result
    assert "ids" in result
    assert "mask" in result
    assert len(result["messages"]) == 1  # One conversation
    assert len(result["ids"]) == 1  # One set of token IDs
    assert len(result["mask"]) == 1  # One set of masks
    
    # Print result
    print("✓ MultiStepEnv test passed")
    print(f"Generated {len(result['messages'][0])} messages")
    for i, msg in enumerate(result["messages"][0]):
        print(f"Message {i+1}: {msg['role']} - {msg['content'][:50]}...")
    
    return result


if __name__ == "__main__":
    # Run tests
    print("Testing mock vLLM implementation...")
    test_mock_llm_creation()
    test_encode_messages()
    try:
        test_chat_method()
        test_multistep_env()
    except Exception as e:
        print(f"Error in tests requiring API key: {str(e)}")
        print("Make sure you have the OPENAI_API_KEY environment variable set.")
    
    print("Tests completed.")