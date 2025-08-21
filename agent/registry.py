"""
Tool Registry module for managing and accessing tool instances.

This module provides a centralized registry for all tool model instances,
allowing easy access, management, and memory optimization of AI tool models.
"""
from typing import Dict, Any, Optional, Type
import logging
from tools.models.BaseModel import BaseModel
from enum import Enum
from tools.model_map import MODEL_MAP

class ModelWeightState(Enum):
    DISK = 0
    CPU = 1
    GPU = 2

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
            cls._instance._state= {}
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
        self._state[tool_name] = ModelWeightState.DISK
        logging.info(f"Tool '{tool_name}' registered successfully.")

    def register_model_map(self, model_map: Dict[str, BaseModel]) -> None:
        """
        Register a map of tool instances with the registry.

        Args:
            model_map: A dictionary mapping tool names to their instances
        """
        for tool_name, tool_instance in model_map.items():
            self.register(tool_name, tool_instance)
        logging.info(f"Registered {len(model_map)} tools from model map.")

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

    def preload(self, tool_name: Optional[str] = None) -> None:
        """
        Preload tool instances into memory.
        Args:
            tool_name: The name of the tool to preload, or None to preload all
        """
        if tool_name is None:
            for name, tool in self._tools.items():
                if hasattr(tool, 'preload') and callable(tool.preload) and self._state[name] == ModelWeightState.DISK:
                    tool.preload()
                    self._state[name] = ModelWeightState.CPU
                    logging.info(f"Tool '{name}' preloaded into memory.")
        else:
            if tool_name not in self._tools:
                self.register(tool_name, MODEL_MAP[tool_name]())

            tool = self._tools[tool_name]
            if hasattr(tool, 'preload') and callable(tool.preload):
                tool.preload()
                self._state[tool_name] = ModelWeightState.CPU
                logging.info(f"Tool '{tool_name}' preloaded into memory.")
            else:
                logging.error(f"Tool '{tool_name}' no implement preload method.")



    def load(self, tool_name: Optional[str] = None) -> None:
        """
        Load tool instances into gpu memory.

        Args:
            tool_name: The name of the tool to load, or None to load all
        """
        if tool_name is None:
            for name, tool in self._tools.items():
                if hasattr(tool, 'load') and callable(tool.load) and self._state[name] == ModelWeightState.CPU:
                    tool.load()
                    self._state[name] = ModelWeightState.GPU
                    logging.info(f"Tool '{name}' loaded into gpu memory.")
        elif tool_name in self._tools and self._state[tool_name] == ModelWeightState.CPU:
            tool = self._tools[tool_name]
            if hasattr(tool, 'load') and callable(tool.load):
                tool.load()
                self._state[tool_name] = ModelWeightState.GPU
                logging.info(f"Tool '{tool_name}' loaded into memory.")
        else:
            logging.warning(f"Tool '{tool_name}' not found in registry.")

    def swap(self, tool_name: Optional[str] = None) -> None:
        if tool_name is None:
            for name, tool in list(self._tools.items()):
                if hasattr(tool, 'swap') and callable(tool.swap) and self._state[name] == ModelWeightState.GPU:
                    tool.swap()
                    self._state[name] = ModelWeightState.CPU
        elif tool_name in self._tools and self._state[tool_name] == ModelWeightState.GPU:
            tool = self._tools[tool_name]
            if hasattr(tool, 'swap') and callable(tool.swap):
                tool.swap()
                self._state[tool_name] = ModelWeightState.CPU
            logging.info(f"Tool '{tool_name}' swap from registry.")
        else:
            logging.warning(f"Tool '{tool_name}' not found in registry.")

    def discord(self, tool_name: Optional[str] = None) -> None:
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
                    self._state[name] = ModelWeightState.DISK
            logging.info("All tools cleared from registry to disk.")
        elif tool_name in self._tools:
            # Clear specific tool
            tool = self._tools[tool_name]
            if hasattr(tool, 'discord') and callable(tool.discord):
                tool.discord()
                self._state[tool_name] = ModelWeightState.DISK
            logging.info(f"Tool '{tool_name}' cleared from registry to disk.")
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
