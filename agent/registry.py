"""
Tool Registry module for managing and accessing tool instances.

This module provides a centralized registry for all tool model instances,
allowing easy access, management, and memory optimization of AI tool models.
"""

from typing import Dict, Any, Optional, Type
import logging
from tools.models.BaseModel import BaseModel

class ToolRegistry:
    """
    A registry for managing tool model instances.
    
    This class implements the Singleton pattern to ensure only one registry
    exists throughout the application.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance
    
    def register(self, tool_name: str, tool_instance: BaseModel) -> None:
        """
        Register a tool instance with the registry.
        
        Args:
            tool_name: A unique name for the tool
            tool_instance: The tool model instance
        """
        if tool_name in self._tools:
            logging.warning(f"Tool '{tool_name}' already registered. Overwriting.")
        self._tools[tool_name] = tool_instance
        logging.info(f"Tool '{tool_name}' registered successfully.")
    
    def get(self, tool_name: str) -> Optional[BaseModel]:
        """
        Get a tool instance by name.
        
        Args:
            tool_name: The name of the tool to retrieve
            
        Returns:
            The tool instance, or None if not found
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            logging.warning(f"Tool '{tool_name}' not found in registry.")
        return tool
    
    def register_lazy(self, tool_name: str, tool_class: Type[BaseModel]) -> None:
        """
        Register a tool class for lazy initialization.
        
        The tool will be instantiated only when first requested.
        
        Args:
            tool_name: A unique name for the tool
            tool_class: The tool model class (not instance)
        """
        self._tools[tool_name] = LazyToolLoader(tool_class)
        logging.info(f"Tool '{tool_name}' registered for lazy loading.")
    
    def list_tools(self) -> Dict[str, Any]:
        """
        List all registered tools.
        
        Returns:
            Dictionary with tool names as keys and tool instances as values
        """
        return {k: v for k, v in self._tools.items()}
    
    def clear(self, tool_name: Optional[str] = None) -> None:
        """
        Clear tool instances from the registry.
        
        Args:
            tool_name: The name of the tool to clear, or None to clear all
        """
        if tool_name is None:
            # Clear all tools
            for name, tool in list(self._tools.items()):
                if hasattr(tool, 'discord') and callable(tool.discord):
                    tool.discord()
            self._tools.clear()
            logging.info("All tools cleared from registry.")
        elif tool_name in self._tools:
            # Clear specific tool
            tool = self._tools[tool_name]
            if hasattr(tool, 'discord') and callable(tool.discord):
                tool.discord()
            del self._tools[tool_name]
            logging.info(f"Tool '{tool_name}' cleared from registry.")
        else:
            logging.warning(f"Tool '{tool_name}' not found in registry.")


class LazyToolLoader:
    """
    Helper class for lazy-loading tool models.
    
    Only instantiates the tool when it's actually needed.
    """
    def __init__(self, tool_class: Type[BaseModel]):
        self.tool_class = tool_class
        self.instance = None
    
    def __getattr__(self, name):
        if self.instance is None:
            self.instance = self.tool_class()
        return getattr(self.instance, name)


# Global singleton registry instance
tool_registry = ToolRegistry()

def get_tool(tool_name: str) -> Optional[BaseModel]:
    """
    Get a tool instance by name from the global registry.
    
    Args:
        tool_name: The name of the tool to retrieve
        
    Returns:
        The tool instance, or None if not found
    """
    return tool_registry.get(tool_name)

def register_tool(tool_name: str, tool_instance: BaseModel) -> None:
    """
    Register a tool with the global registry.
    
    Args:
        tool_name: A unique name for the tool
        tool_instance: The tool model instance
    """
    tool_registry.register(tool_name, tool_instance)

def register_lazy_tool(tool_name: str, tool_class: Type[BaseModel]) -> None:
    """
    Register a tool class for lazy initialization with the global registry.
    
    Args:
        tool_name: A unique name for the tool
        tool_class: The tool model class (not instance)
    """
    tool_registry.register_lazy(tool_name, tool_class)
