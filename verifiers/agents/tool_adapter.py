"""
Tool adapter for integrating Verifiers tools with SmolAgents.

This module provides adapters for converting between Verifiers tools
and SmolAgents tools, enabling either type to be used with both systems.
"""

import sys
import inspect
from typing import Any, Dict, List, Callable, Optional, Type, Union

# Add smolagents to path if needed
sys.path.append('/Users/allanniemerg/dev/verifiers/wip/smolagents')

# Import SmolAgents classes
try:
    from smolagents.tools.tool import Tool as SmolTool
except ImportError:
    raise ImportError("SmolAgents not found. Make sure it's installed and in the Python path.")


def verifiers_tool_to_smol_tool(
    name: str,
    func: Callable,
    description: str = None,
    **kwargs
) -> SmolTool:
    """
    Convert a Verifiers tool function to a SmolAgents Tool.
    
    Args:
        name: Name of the tool.
        func: Verifiers tool function.
        description: Optional description of the tool.
        **kwargs: Additional arguments to pass to the SmolAgents Tool.
        
    Returns:
        A SmolAgents Tool instance.
    """
    # Get function signature
    sig = inspect.signature(func)
    
    # Extract parameter information
    inputs = {}
    for param_name, param in sig.parameters.items():
        # Skip 'self' parameter for class methods
        if param_name == 'self':
            continue
            
        # Determine parameter type
        param_type = "string"  # Default type
        if param.annotation != inspect.Parameter.empty:
            if param.annotation == str:
                param_type = "string"
            elif param.annotation == int:
                param_type = "integer"
            elif param.annotation == float:
                param_type = "number"
            elif param.annotation == bool:
                param_type = "boolean"
        
        # Determine if parameter is required
        required = param.default == inspect.Parameter.empty
        
        # Create parameter schema
        inputs[param_name] = {
            "type": param_type,
            "description": f"Parameter {param_name}",
            "required": required
        }
    
    # Extract return type information
    return_type = "string"  # Default return type
    if sig.return_annotation != inspect.Signature.empty:
        if sig.return_annotation == str:
            return_type = "string"
        elif sig.return_annotation == int:
            return_type = "integer"
        elif sig.return_annotation == float:
            return_type = "number"
        elif sig.return_annotation == bool:
            return_type = "boolean"
        elif sig.return_annotation == Dict:
            return_type = "object"
        elif sig.return_annotation == List:
            return_type = "array"
    
    # Create the SmolAgents Tool
    class VerifiersToolAdapter(SmolTool):
        def setup(self):
            self.name = name
            self.description = description or f"Tool {name}"
            self.inputs = inputs
            self.output_type = return_type
        
        def forward(self, **kwargs):
            # Call the original Verifiers tool function
            return func(**kwargs)
    
    return VerifiersToolAdapter()


def create_calculator_tool() -> SmolTool:
    """
    Create a calculator tool for SmolAgents.
    
    Returns:
        A SmolAgents Tool for mathematical calculations.
    """
    class CalculatorTool(SmolTool):
        def setup(self):
            self.name = "calculator"
            self.description = "A tool for evaluating mathematical expressions"
            self.inputs = {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression to evaluate (e.g., '2 + 2', '3 * 4')",
                    "required": True
                }
            }
            self.output_type = "number"
        
        def forward(self, expression: str) -> Union[int, float, str]:
            """
            Evaluate a mathematical expression.
            
            Args:
                expression: A string representing a mathematical expression.
                
            Returns:
                The result of the evaluation.
            """
            try:
                # Use a restricted subset of Python's eval for safety
                allowed_names = {
                    "abs": abs, "round": round,
                    "max": max, "min": min,
                    "pow": pow, "sum": sum
                }
                
                # Add common math functions
                import math
                for name in ["sin", "cos", "tan", "exp", "log", "sqrt", "ceil", "floor", "pi"]:
                    if hasattr(math, name):
                        allowed_names[name] = getattr(math, name)
                
                # Evaluate the expression with the restricted namespace
                code = compile(expression, "<string>", "eval")
                for name in code.co_names:
                    if name not in allowed_names:
                        raise NameError(f"The use of '{name}' is not allowed")
                
                result = eval(code, {"__builtins__": {}}, allowed_names)
                return result
            except Exception as e:
                return f"Error: {str(e)}"
    
    return CalculatorTool()


def create_search_tool() -> SmolTool:
    """
    Create a search tool for SmolAgents.
    
    Returns:
        A SmolAgents Tool for searching.
    """
    class SearchTool(SmolTool):
        def setup(self):
            self.name = "search"
            self.description = "A tool for searching the web for information"
            self.inputs = {
                "query": {
                    "type": "string",
                    "description": "The search query",
                    "required": True
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "required": False
                }
            }
            self.output_type = "string"
        
        def forward(self, query: str, max_results: int = 3) -> str:
            """
            Search for information.
            
            This is a mock implementation that would be replaced with
            an actual search API in production.
            
            Args:
                query: The search query.
                max_results: Maximum number of results to return.
                
            Returns:
                Search results as a string.
            """
            # In a real implementation, this would call a search API
            # For now, we'll return a mock response
            return f"Mock search results for '{query}' (limited to {max_results} results):\n" + \
                   "\n".join([f"Result {i+1}: This is a mock search result" for i in range(min(max_results, 5))])
    
    return SearchTool()