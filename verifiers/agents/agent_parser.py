"""
Parser for extracting tool calls and XML tags from agent outputs.

This module provides a parser for extracting tool calls and other structured
information from agent outputs, with a focus on XML-based formats used by
both Verifiers and SmolAgents.
"""

import re
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from types import SimpleNamespace
import logging


class VerifiersAgentParser:
    """
    Parser for extracting tool calls and XML tags from agent outputs.
    
    This parser is designed to handle the XML format used by Verifiers,
    with specific support for reasoning, tool calls, and final answers.
    It also provides graceful fallback mechanisms for handling imperfect
    model outputs, which is essential during early training.
    """
    
    def __init__(self, fields: List[str] = None):
        """
        Initialize the parser with the fields to extract.
        
        Args:
            fields: List of field names to extract from XML tags.
                   Defaults to ["reasoning", "tool_call", "answer"].
        """
        self.fields = fields or ["reasoning", "tool_call", "answer"]
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def _extract_tag_content(self, text: str, tag: str) -> Optional[str]:
        """
        Extract content between XML tags.
        
        Args:
            text: The text to parse.
            tag: The tag name to extract.
            
        Returns:
            The content between the tags, or None if not found.
        """
        if not text:
            return None
            
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract a tool call from text, with fallback mechanisms for imperfect formats.
        
        Args:
            text: The text to parse.
            
        Returns:
            A dictionary with the tool call (name and args), or None if not found.
        """
        # First, try the standard XML tag format
        tool_call_text = self._extract_tag_content(text, "tool_call")
        
        if tool_call_text:
            try:
                # Try to parse as JSON
                return json.loads(tool_call_text.strip())
            except json.JSONDecodeError:
                self._logger.warning(f"Failed to parse tool call as JSON: {tool_call_text}")
                
                # Fallback: Try to extract name and args with regex
                name_match = re.search(r'"name"\s*:\s*"([^"]+)"', tool_call_text)
                if name_match:
                    name = name_match.group(1)
                    # Try to extract args
                    args_match = re.search(r'"args"\s*:\s*(\{.*\})', tool_call_text, re.DOTALL)
                    if args_match:
                        try:
                            args = json.loads(args_match.group(1))
                            return {"name": name, "args": args}
                        except json.JSONDecodeError:
                            return {"name": name, "args": {}}
                    return {"name": name, "args": {}}
        
        # More fallbacks for other formats
        # Try to find a tool call using different patterns
        
        # Look for "I'm using tool X" patterns
        tool_intention_match = re.search(r"(use|using|call|calling|execute|executing)\s+(?:the\s+)?(?:tool\s+)?['\"]?([a-zA-Z0-9_]+)['\"]?(?:\s+to|\s+with|\s+tool)?", text, re.IGNORECASE)
        if tool_intention_match:
            tool_name = tool_intention_match.group(2)
            
            # Try to find arguments in the vicinity
            args_dict = {}
            
            # Look for patterns like "arg=value" or "arg: value"
            arg_patterns = re.finditer(r"([a-zA-Z0-9_]+)\s*(?:=|:)\s*(?:\"([^\"]*)\"|'([^']*)'|([^,\s\}]*))", text)
            for match in arg_patterns:
                arg_name = match.group(1)
                # Take the first non-None group as the value
                arg_value = next((g for g in match.groups()[1:] if g is not None), "")
                args_dict[arg_name] = arg_value
            
            return {"name": tool_name, "args": args_dict}
        
        return None
    
    def _detect_final_answer(self, text: str) -> Optional[str]:
        """
        Detect if the text contains a final answer pattern, even without proper tags.
        
        Args:
            text: The text to analyze.
            
        Returns:
            The extracted answer if found, otherwise None.
        """
        # First, try the standard tag
        answer = self._extract_tag_content(text, "answer")
        if answer:
            return answer
        
        # Look for phrases that might indicate an answer
        answer_phrases = [
            r"(?:the|my) (?:final )?answer is[:\s]+(.+)",
            r"(?:therefore|thus|hence|in conclusion|to summarize)[,\s]+(.+)",
            r"(?:the|my) (?:final )?result is[:\s]+(.+)"
        ]
        
        for pattern in answer_phrases:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return None
    
    def parse(self, text: str) -> SimpleNamespace:
        """
        Parse text into a SimpleNamespace with extracted fields.
        
        Args:
            text: The text to parse.
            
        Returns:
            A SimpleNamespace containing the extracted fields.
        """
        parsed = SimpleNamespace()
        
        # Extract standard fields
        for field in self.fields:
            if field == "tool_call":
                # Special handling for tool calls
                tool_call = self._extract_tool_call(text)
                if tool_call:
                    setattr(parsed, field, tool_call)
            else:
                value = self._extract_tag_content(text, field)
                if value is not None:
                    setattr(parsed, field, value)
        
        # Try to detect a final answer even if not explicitly tagged
        if not hasattr(parsed, "answer"):
            answer = self._detect_final_answer(text)
            if answer:
                parsed.answer = answer
        
        return parsed
    
    def format_tool_call(self, name: str, args: Dict[str, Any]) -> str:
        """
        Format a tool call as XML with JSON content.
        
        Args:
            name: The name of the tool.
            args: The arguments to pass to the tool.
            
        Returns:
            A formatted tool call string.
        """
        tool_json = json.dumps({"name": name, "args": args}, indent=2)
        return f"<tool_call>\n{tool_json}\n</tool_call>"
    
    def format_response(self, reasoning: Optional[str] = None, tool_call: Optional[Dict[str, Any]] = None, 
                       answer: Optional[str] = None) -> str:
        """
        Format a complete response with reasoning, tool call, and/or answer.
        
        Args:
            reasoning: Optional reasoning text.
            tool_call: Optional tool call dictionary (name and args).
            answer: Optional final answer text.
            
        Returns:
            A formatted response string.
        """
        parts = []
        
        if reasoning:
            parts.append(f"<reasoning>\n{reasoning}\n</reasoning>")
        
        if tool_call:
            tool_call_str = self.format_tool_call(tool_call["name"], tool_call.get("args", {}))
            parts.append(tool_call_str)
        
        if answer:
            parts.append(f"<answer>\n{answer}\n</answer>")
        
        return "\n\n".join(parts)