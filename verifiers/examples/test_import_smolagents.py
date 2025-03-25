"""
Simple script to test if SmolAgents can be imported.

This script attempts to import the SmolAgents module and prints its location.
"""

import os
import sys
import logging

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Print out the Python path
logger.info("Python path:")
for path in sys.path:
    logger.info(f"  {path}")

# Add all possible paths for smolagents
logger.info("Adding potential SmolAgents paths...")

# Absolute path to src directory
src_path = '/Users/allanniemerg/dev/verifiers/wip/smolagents/src'
sys.path.insert(0, src_path)
logger.info(f"Added {src_path} to path")

# Relative path from script location
script_dir = os.path.dirname(os.path.abspath(__file__))
wip_smolagents_src_path = os.path.abspath(os.path.join(script_dir, '../../../wip/smolagents/src'))
sys.path.insert(0, wip_smolagents_src_path)
logger.info(f"Added {wip_smolagents_src_path} to path")

# Try importing
try:
    logger.info("Attempting to import smolagents module...")
    import smolagents
    logger.info(f"Successfully imported smolagents from {smolagents.__file__}")
    
    # Try importing specific modules with correct paths
    logger.info("Attempting to import specific modules...")
    from smolagents.models import ChatMessage
    logger.info("Imported ChatMessage from models")
    
    from smolagents.models import Model
    logger.info("Imported Model from models")
    
    from smolagents.tools import Tool
    logger.info("Imported Tool from tools")
    
    from smolagents.agents import ToolCallingAgent
    logger.info("Imported ToolCallingAgent from agents")
    
    from smolagents.memory import ActionStep, ToolCall
    logger.info("Imported ActionStep and ToolCall from memory")
    
    from smolagents.monitoring import LogLevel
    logger.info("Imported LogLevel from monitoring")
    
    # Print package information
    logger.info(f"SmolAgents version: {getattr(smolagents, '__version__', 'unknown')}")
    
    print("SUCCESS: All imports worked correctly!")
    
except ImportError as e:
    logger.error(f"Failed to import smolagents: {str(e)}")
    
    # List the contents of the directory to verify structure
    for potential_path in ['/Users/allanniemerg/dev/verifiers/wip/smolagents/src', wip_smolagents_src_path]:
        if os.path.exists(potential_path):
            logger.info(f"Contents of {potential_path}:")
            try:
                files = os.listdir(potential_path)
                for f in files:
                    logger.info(f"  {f}")
                
                # Check for __init__.py
                if '__init__.py' in files:
                    logger.info("  Found __init__.py file")
                else:
                    logger.warning("  No __init__.py file found!")
                    
            except Exception as e:
                logger.error(f"Error listing directory: {str(e)}")
        else:
            logger.error(f"Path does not exist: {potential_path}")
    
    print(f"ERROR: Failed to import smolagents: {str(e)}")

if __name__ == "__main__":
    pass  # All code is executed on import