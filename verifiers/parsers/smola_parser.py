import json
import re
from typing import Any, Dict, Optional

from verifiers.parsers.xml_parser import XMLParser

class SmolaParser(XMLParser):
    """
    A parser specifically designed for SmolaAgents tool format, focusing on the <tool> tag format.
    
    This parser extends the XMLParser to handle SmolaAgents tools format and provides methods
    for extracting and validating tool calls.
    """
    
    def __init__(self, fields=["reasoning", "tool", "answer"]):
        """
        Initialize the SmolaParser with appropriate fields for tool-based conversations.
        
        Args:
            fields: List of XML tag fields to parse for. Defaults to ["reasoning", "tool", "answer"]
        """
        super().__init__(fields=fields)
    
    def parse_tool(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Parse tool calls using only <tool> tags.
        
        This method extracts the content of <tool> tags and parses it as JSON.
        SmolaAgents uses a JSON format within <tool> tags for tool calls.
        
        Args:
            content: The content to parse for tool calls
            
        Returns:
            A dictionary with the parsed tool call if successful, None otherwise
        """
        # First, attempt to parse with the regular XML parser
        parsed = self.parse(content)
        
        # Check if we have a successfully parsed tool field
        if hasattr(parsed, 'tool') and parsed.tool is not None:
            try:
                # Try to parse the tool field as JSON
                return json.loads(parsed.tool)
            except json.JSONDecodeError:
                # If not valid JSON, return None
                return None
        
        # If we don't have a tool field, use regex as a fallback to look for tool tags
        # This helps handle cases where the model might use slightly different formatting
        tool_pattern = r"<tool>\s*(.*?)\s*</tool>"
        match = re.search(tool_pattern, content, re.DOTALL)
        
        if match:
            try:
                # Try to parse the matched content as JSON
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                # If not valid JSON, return None
                return None
                
        # No valid tool call found
        return None
    
    def format_tool_result(self, result: str) -> str:
        """
        Format a tool execution result in the <r> tags used by SmolaAgents.
        
        Args:
            result: The result string to format
            
        Returns:
            The formatted result string
        """
        # Format result with <r> tags as used in SmolaAgents
        return f"<r>\n{result}\n</r>"
    
    def get_smola_format_reward_func(self):
        """
        Return a reward function that checks for proper SmolaAgents XML tag usage.
        
        The returned function evaluates if messages in trajectories properly use
        the SmolaAgents expected XML tags, particularly focusing on <tool> tags.
        """
        def smola_format_reward_func(completions, **kwargs):
            """Reward function specifically for SmolaAgents format."""
            def check_format(trajectory):
                # Get assistant messages
                model_messages = [msg for msg in trajectory if msg['role'] == 'assistant']
                if not model_messages:
                    return 0.0
                
                # Calculate format scores for each message
                format_scores = []
                for msg in model_messages:
                    content = msg['content']
                    score = 0.0
                    
                    # Check for proper <tool> tags
                    tool_tags = content.count("<tool>")
                    tool_close_tags = content.count("</tool>")
                    
                    # If tool tags are present, check JSON validity
                    if tool_tags > 0 and tool_close_tags > 0:
                        # Extract tool content
                        tool_pattern = r"<tool>\s*(.*?)\s*</tool>"
                        match = re.search(tool_pattern, content, re.DOTALL)
                        
                        if match:
                            try:
                                # Check if it's valid JSON
                                tool_json = json.loads(match.group(1))
                                # Check if it has name and args
                                if "name" in tool_json and "args" in tool_json:
                                    score += 1.0
                                else:
                                    score += 0.5  # Partial credit for valid JSON but missing fields
                            except json.JSONDecodeError:
                                score += 0.2  # Small credit for using tags but invalid JSON
                        else:
                            score += 0.1  # Minimal credit for tags without parseable content
                    elif "<answer>" in content and "</answer>" in content:
                        # Give credit for final answer format
                        score += 1.0
                    
                    format_scores.append(score)
                
                # Return average format score
                if not format_scores:
                    return 0.0
                return sum(format_scores) / len(format_scores)
            
            return [check_format(c) for c in completions]
        
        return smola_format_reward_func