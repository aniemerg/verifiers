from .system_prompts import (
    SIMPLE_PROMPT, CODE_PROMPT,
    DEFAULT_TOOL_PROMPT_TEMPLATE
)
from .few_shots import (
    MATH_FEW_SHOT, DOUBLECHECK_FEW_SHOT, CODE_FEW_SHOT, 
    TOOL_FEW_SHOT, COMMONSENSE_FEW_SHOT, SEARCH_FEW_SHOT,
    CALCULATOR_FEW_SHOT
)
from .smola_templates import (
    SMOLA_SYSTEM_PROMPT, SMOLA_CALCULATOR_PROMPT,
    SMOLA_PYTHON_PROMPT, SMOLA_SEARCH_PROMPT
)
from .smola_few_shots import (
    get_calculator_few_shots, get_python_few_shots,
    get_search_few_shots, get_mixed_tools_few_shots
)

__all__ = [
    "SIMPLE_PROMPT", "MATH_FEW_SHOT", "DOUBLECHECK_FEW_SHOT", 
    "CODE_FEW_SHOT", "CODE_PROMPT", "TOOL_FEW_SHOT",
    "COMMONSENSE_FEW_SHOT", "DEFAULT_TOOL_PROMPT_TEMPLATE",
    "SEARCH_FEW_SHOT", "CALCULATOR_FEW_SHOT",
    "SMOLA_SYSTEM_PROMPT", "SMOLA_CALCULATOR_PROMPT",
    "SMOLA_PYTHON_PROMPT", "SMOLA_SEARCH_PROMPT",
    "get_calculator_few_shots", "get_python_few_shots",
    "get_search_few_shots", "get_mixed_tools_few_shots"
]