import json
import re
from typing import List, Dict, Any, Union, Tuple, Optional, Callable
from types import SimpleNamespace

from verifiers.parsers.xml_parser import XMLParser

class SmolaParser(XMLParser):
    """
    Parser for handling SmolaAgents tool format within Verifiers.
    
    Extends the XMLParser to provide compatibility with SmolaAgents tool format
    while maintaining the XML structure used by Verifiers.
    """
    
    def __init__(self, fields: List[Union[str, Tuple[str, ...]]]):
        super().__init__(fields)
    
    def format_tool_call(self, name: str, args: Dict[str, Any]) -> str:
        """
        Format a tool call in SmolaAgents-compatible format using XML.
        
        Args:
            name: The name of the tool to call
            args: Dictionary of arguments to pass to the tool
            
        Returns:
            Formatted XML string for the tool call
        """
        # Format the tool call as JSON
        tool_json = json.dumps({"name": name, "args": args}, indent=2)
        return f"<tool_call>\n{tool_json}\n</tool_call>"
    
    def parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse a tool call from the given text.
        
        Args:
            text: The text containing the tool call
            
        Returns:
            Dict with 'name' and 'args' if parsed successfully, None otherwise
        """
        # First use the standard parser to extract the tool_call field
        parsed = self.parse(text)
        
        # If tool_call field exists, parse it as JSON
        if hasattr(parsed, 'tool_call') and parsed.tool_call is not None:
            try:
                return json.loads(parsed.tool_call)
            except json.JSONDecodeError:
                return None
        return None