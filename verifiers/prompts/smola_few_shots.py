"""
Few-shot examples for SmolaAgents tools integration, focusing on the <tool> tag format.

The examples demonstrate the proper format for using tools with SmolaAgents:
1. Clear reasoning using <reasoning> tags
2. Tool calls with <tool> tags and proper JSON structure
3. Final answers with <answer> tags
"""

from typing import List, Dict

from verifiers.parsers.smola_parser import SmolaParser

# Initialize parser for formatting examples
_parser = SmolaParser(fields=["reasoning", "tool", "answer"])

def get_calculator_few_shots() -> List[Dict[str, str]]:
    """
    Get few-shot examples for calculator tool usage with SmolaAgents format.
    
    These examples demonstrate:
    - Proper reasoning about the calculation needed
    - Appropriate formatting of calculator tool calls with <tool> tags
    - Handling of tool results
    - Providing a final answer with <answer> tags
    
    Returns:
        List of chat messages in few-shot format
    """
    example1 = {
        "role": "user", 
        "content": "What is 25 squared plus 13?"
    }
    
    example1_response = {
        "role": "assistant", 
        "content": _parser.format(
            reasoning="To solve this problem, I need to calculate 25 squared (25²) and then add 13 to the result.\n\nFirst, I'll calculate 25²:",
            tool="""{"name": "calculator", "args": {"expression": "25^2"}}"""
        )
    }
    
    example1_result = {
        "role": "user", 
        "content": "<r>\n625\n</r>"
    }
    
    example1_final = {
        "role": "assistant", 
        "content": _parser.format(
            reasoning="Now that I have 25² = 625, I need to add 13 to get the final result:\n\n625 + 13 = 638",
            answer="The result of 25 squared plus 13 is 638."
        )
    }
    
    example2 = {
        "role": "user", 
        "content": "If a triangle has sides of length 3, 4, and 5, what is its area?"
    }
    
    example2_response = {
        "role": "assistant", 
        "content": _parser.format(
            reasoning="To find the area of a triangle with sides 3, 4, and 5, I can use Heron's formula, which states that:\n\nArea = √(s(s-a)(s-b)(s-c))\n\nwhere s is the semi-perimeter: s = (a+b+c)/2\n\nLet me calculate s first:",
            tool="""{"name": "calculator", "args": {"expression": "(3+4+5)/2"}}"""
        )
    }
    
    example2_result = {
        "role": "user", 
        "content": "<r>\n6.0\n</r>"
    }
    
    example2_step2 = {
        "role": "assistant", 
        "content": _parser.format(
            reasoning="Now I have s = 6. Using Heron's formula, I need to calculate √(s(s-a)(s-b)(s-c)) = √(6(6-3)(6-4)(6-5)) = √(6·3·2·1) = √(36)",
            tool="""{"name": "calculator", "args": {"expression": "sqrt(6*3*2*1)"}}"""
        )
    }
    
    example2_result2 = {
        "role": "user", 
        "content": "<r>\n6.0\n</r>"
    }
    
    example2_final = {
        "role": "assistant", 
        "content": _parser.format(
            reasoning="I've calculated the area using Heron's formula. The area of the triangle is √(6·3·2·1) = √36 = 6 square units.",
            answer="The area of the triangle with sides 3, 4, and 5 is 6 square units."
        )
    }
    
    return [example1, example1_response, example1_result, example1_final, 
            example2, example2_response, example2_result, example2_step2, example2_result2, example2_final]

def get_python_few_shots() -> List[Dict[str, str]]:
    """
    Get few-shot examples for Python code execution tool usage with SmolaAgents format.
    
    These examples demonstrate:
    - Reasoning about how to solve a problem with Python
    - Proper formatting of Python code execution tool calls
    - Interpretation of results
    - Providing a final answer
    
    Returns:
        List of chat messages in few-shot format
    """
    example1 = {
        "role": "user", 
        "content": "Find the first 5 prime numbers."
    }
    
    example1_response = {
        "role": "assistant", 
        "content": _parser.format(
            reasoning="To find the first 5 prime numbers, I'll write a Python program that generates prime numbers and stops after finding 5 of them.",
            tool="""{"name": "python", "args": {"code": "def is_prime(n):\\n    if n <= 1:\\n        return False\\n    if n <= 3:\\n        return True\\n    if n % 2 == 0 or n % 3 == 0:\\n        return False\\n    i = 5\\n    while i * i <= n:\\n        if n % i == 0 or n % (i + 2) == 0:\\n            return False\\n        i += 6\\n    return True\\n\\nprimes = []\\nnum = 2\\nwhile len(primes) < 5:\\n    if is_prime(num):\\n        primes.append(num)\\n    num += 1\\n\\nprint(primes)"}}"""
        )
    }
    
    example1_result = {
        "role": "user", 
        "content": "<r>\n[2, 3, 5, 7, 11]\n</r>"
    }
    
    example1_final = {
        "role": "assistant", 
        "content": _parser.format(
            reasoning="The program found the first 5 prime numbers by testing each number sequentially and checking if it's prime. The first 5 prime numbers are 2, 3, 5, 7, and 11.",
            answer="The first 5 prime numbers are 2, 3, 5, 7, and 11."
        )
    }
    
    example2 = {
        "role": "user", 
        "content": "Calculate the factorial of 5."
    }
    
    example2_response = {
        "role": "assistant", 
        "content": _parser.format(
            reasoning="To calculate the factorial of 5 (5!), I'll use Python. The factorial of a number n is the product of all positive integers less than or equal to n. For example, 5! = 5 × 4 × 3 × 2 × 1.",
            tool="""{"name": "python", "args": {"code": "def factorial(n):\\n    if n == 0 or n == 1:\\n        return 1\\n    else:\\n        result = 1\\n        for i in range(2, n+1):\\n            result *= i\\n        return result\\n\\nprint(f\\"The factorial of 5 is: {factorial(5)}\\")"}}"""
        )
    }
    
    example2_result = {
        "role": "user", 
        "content": "<r>\nThe factorial of 5 is: 120\n</r>"
    }
    
    example2_final = {
        "role": "assistant", 
        "content": _parser.format(
            reasoning="I calculated the factorial of 5 using Python. The factorial of 5 (5!) is equal to 5 × 4 × 3 × 2 × 1 = 120.",
            answer="The factorial of 5 is 120."
        )
    }
    
    return [example1, example1_response, example1_result, example1_final,
            example2, example2_response, example2_result, example2_final]

def get_search_few_shots() -> List[Dict[str, str]]:
    """
    Get few-shot examples for search tool usage with SmolaAgents format.
    
    These examples demonstrate:
    - Reasoning about how to find information using search
    - Proper formatting of search tool calls
    - Interpretation of search results
    - Providing a final answer based on search findings
    
    Returns:
        List of chat messages in few-shot format
    """
    example1 = {
        "role": "user", 
        "content": "Who is the current Secretary-General of the United Nations?"
    }
    
    example1_response = {
        "role": "assistant", 
        "content": _parser.format(
            reasoning="To answer this question, I need to find out who the current Secretary-General of the United Nations is. I'll use a search to find this information.",
            tool="""{"name": "search", "args": {"query": "current Secretary-General of the United Nations"}}"""
        )
    }
    
    example1_result = {
        "role": "user", 
        "content": "<r>\nAntónio Guterres is the current Secretary-General of the United Nations. He took office on January 1, 2017, and was re-elected for a second term that began on January 1, 2022, and will end on December 31, 2026.\n</r>"
    }
    
    example1_final = {
        "role": "assistant", 
        "content": _parser.format(
            reasoning="According to the search results, António Guterres is the current Secretary-General of the United Nations. He began his first term on January 1, 2017, and is currently serving his second term, which began on January 1, 2022, and will end on December 31, 2026.",
            answer="The current Secretary-General of the United Nations is António Guterres. He has been serving since January 1, 2017, and is currently in his second term, which runs until December 31, 2026."
        )
    }
    
    example2 = {
        "role": "user", 
        "content": "What is the capital of Australia?"
    }
    
    example2_response = {
        "role": "assistant", 
        "content": _parser.format(
            reasoning="To answer this question, I need to find out the capital city of Australia. I'll search for this information.",
            tool="""{"name": "search", "args": {"query": "capital of Australia"}}"""
        )
    }
    
    example2_result = {
        "role": "user", 
        "content": "<r>\nCanberra is the capital city of Australia. It is located in the Australian Capital Territory (ACT) and was founded in 1913 as a planned city designed by American architect Walter Burley Griffin. Many people mistakenly believe Sydney or Melbourne is the capital, but Canberra was chosen as a compromise between these two rival cities.\n</r>"
    }
    
    example2_final = {
        "role": "assistant", 
        "content": _parser.format(
            reasoning="According to the search results, Canberra is the capital city of Australia. It's located in the Australian Capital Territory (ACT) and was founded in 1913 as a planned city. Interestingly, it was designed by American architect Walter Burley Griffin and was chosen as a compromise between the rival cities of Sydney and Melbourne.",
            answer="The capital of Australia is Canberra, located in the Australian Capital Territory (ACT). It was established in 1913 as a planned city and serves as the seat of the Australian government."
        )
    }
    
    return [example1, example1_response, example1_result, example1_final,
            example2, example2_response, example2_result, example2_final]

def get_mixed_tools_few_shots() -> List[Dict[str, str]]:
    """
    Get few-shot examples for mixed tools usage with SmolaAgents format.
    
    These examples demonstrate:
    - Using multiple tools in sequence
    - Reasoning between tool calls
    - Proper formatting of different tool calls
    - Providing a final answer based on multiple tool results
    
    Returns:
        List of chat messages in few-shot format
    """
    example = {
        "role": "user", 
        "content": "If I have a rectangular garden that is 5m × 3m, how many square meters of soil do I need if the soil should be 0.15m deep?"
    }
    
    example_response1 = {
        "role": "assistant", 
        "content": _parser.format(
            reasoning="To solve this problem, I need to find the volume of soil needed for the garden. The formula for volume is:\n\nVolume = Area × Depth\n\nFirst, I'll calculate the area of the rectangular garden:",
            tool="""{"name": "calculator", "args": {"expression": "5 * 3"}}"""
        )
    }
    
    example_result1 = {
        "role": "user", 
        "content": "<r>\n15\n</r>"
    }
    
    example_response2 = {
        "role": "assistant", 
        "content": _parser.format(
            reasoning="Now I know the area of the garden is 15 square meters. To find the volume of soil needed, I multiply the area by the depth (0.15m):",
            tool="""{"name": "calculator", "args": {"expression": "15 * 0.15"}}"""
        )
    }
    
    example_result2 = {
        "role": "user", 
        "content": "<r>\n2.25\n</r>"
    }
    
    example_final = {
        "role": "assistant", 
        "content": _parser.format(
            reasoning="I've calculated that for a garden with an area of 15 square meters (5m × 3m) and a soil depth of 0.15m, the volume of soil needed is:\n\nVolume = Area × Depth = 15 m² × 0.15m = 2.25 m³\n\nTherefore, 2.25 cubic meters of soil are needed for the garden.",
            answer="For a rectangular garden that is 5m × 3m with soil 0.15m deep, you need 2.25 cubic meters (m³) of soil."
        )
    }
    
    return [example, example_response1, example_result1, example_response2, example_result2, example_final]