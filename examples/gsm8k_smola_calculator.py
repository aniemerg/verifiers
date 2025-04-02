"""
Example showing the use of the SmolaToolEnv with a calculator tool on GSM8K-style problems.

This example demonstrates:
1. Setting up the SmolaToolEnv with the calculator tool
2. Using the SmolaAgents <tool> tag format for tool calls
3. Evaluating model performance with the SmolaToolRubric
"""

import json
from typing import Dict, Any

from verifiers.envs import SmolaToolEnv
from verifiers.tools import calculator
from verifiers.prompts import (
    SMOLA_CALCULATOR_PROMPT,
    get_calculator_few_shots
)

def gsm8k_demo():
    """
    Run a demonstration of the SmolaToolEnv with calculator tool using a GSM8K-style math problem.
    """
    # Set up the tool to use
    calculator_tool = calculator.calculate

    # Create the environment with the calculator tool
    env = SmolaToolEnv(
        tools=[calculator_tool],
        system_prompt=SMOLA_CALCULATOR_PROMPT,
        few_shot=get_calculator_few_shots(),
        max_steps=10
    )

    # Example math problem from GSM8K
    math_problem = "John has 5 boxes of pencils with 12 pencils in each box. " \
                  "He gives 3 pencils to each of his 8 friends. " \
                  "How many pencils does John have left?"

    # Create prompt messages
    messages = [
        {"role": "system", "content": env.system_prompt},
        *env.few_shot,
        {"role": "user", "content": math_problem}
    ]
    
    # This would normally use the actual LLM, but for testing we can mock it
    # by manually creating a sample conversation flow
    import os
    from verifiers.mock_vllm import MockLLM
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Use the mock LLM to show the flow without actual API calls
    mock_llm = MockLLM(model_path=f"{parent_dir}/configs/zero3.yaml")
    
    # Generate sample output to demonstrate the format
    print(f"Problem: {math_problem}\n")
    print("Starting environment simulation with SmolaAgents tools...\n")
    
    sample_response = {
        "reasoning": "To solve this problem, I need to find how many pencils John has left after giving some to his friends.\n\nFirst, let's calculate the total number of pencils John had initially.",
        "tool": json.dumps({"name": "calculator", "args": {"expression": "5 * 12"}})
    }
    print(f"Step 1 - Model output:")
    print(f"<reasoning>\n{sample_response['reasoning']}\n</reasoning>")
    print(f"<tool>\n{sample_response['tool']}\n</tool>\n")
    
    print("Tool result:")
    print("<r>\n60\n</r>\n")
    
    sample_response2 = {
        "reasoning": "John started with 60 pencils. Now I need to calculate how many pencils he gave to his friends. He gave 3 pencils to each of his 8 friends.",
        "tool": json.dumps({"name": "calculator", "args": {"expression": "3 * 8"}})
    }
    print(f"Step 2 - Model output:")
    print(f"<reasoning>\n{sample_response2['reasoning']}\n</reasoning>")
    print(f"<tool>\n{sample_response2['tool']}\n</tool>\n")
    
    print("Tool result:")
    print("<r>\n24\n</r>\n")
    
    sample_response3 = {
        "reasoning": "John gave away a total of 24 pencils to his friends. Now I need to calculate how many pencils he has left by subtracting the pencils he gave away from his initial total.",
        "tool": json.dumps({"name": "calculator", "args": {"expression": "60 - 24"}})
    }
    print(f"Step 3 - Model output:")
    print(f"<reasoning>\n{sample_response3['reasoning']}\n</reasoning>")
    print(f"<tool>\n{sample_response3['tool']}\n</tool>\n")
    
    print("Tool result:")
    print("<r>\n36\n</r>\n")
    
    sample_final = {
        "reasoning": "I've calculated that John started with 60 pencils (5 boxes × 12 pencils per box). He gave away 24 pencils (3 pencils × 8 friends). Therefore, John has 60 - 24 = 36 pencils left.",
        "answer": "John has 36 pencils left."
    }
    print(f"Final response - Model output:")
    print(f"<reasoning>\n{sample_final['reasoning']}\n</reasoning>")
    print(f"<answer>\n{sample_final['answer']}\n</answer>")
    
    print("\nNote: This is a simulated example to demonstrate the format. In a real application, the model would generate these responses.")
    

if __name__ == "__main__":
    gsm8k_demo()