"""
Tool Registry module for managing and accessing tool instances.

This module provides a centralized registry for all tool model instances,
allowing easy access, management, and memory optimization of AI tool models.
It includes thread safety mechanisms for concurrent access.
"""

from typing import Dict, Any, Optional, Type
import logging
import threading
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
    exists throughout the application. It uses threading locks to ensure
    thread-safety when multiple agents access the registry concurrently.
    """
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._counter = {}
                cls._instance._tools = {}
                cls._instance._state = {}
                cls._instance._locks = {}  # Tool-specific locks
                cls._instance._global_lock = threading.RLock()  # Global registry lock
        return cls._instance

    def register(self, tool_name: str, tool_instance: BaseModel) -> None:
        """
        Register a tool instance with the registry.

        Args:
            tool_name: A unique name for the tool
            tool_instance: The tool model instance
        """
        with self._global_lock:
            if tool_name in self._tools:
                logging.warning(f"Tool '{tool_name}' already registered. Overwriting.")
            self._tools[tool_name] = tool_instance
            self._state[tool_name] = ModelWeightState.DISK
            # Create a tool-specific lock if it doesn't exist
            if tool_name not in self._locks:
                self._locks[tool_name] = threading.RLock()
            logging.info(f"Tool '{tool_name}' registered successfully.")

    def register_model_map(self, model_map: Dict[str, BaseModel]) -> None:
        """
        Register a map of tool instances with the registry.

        Args:
            model_map: A dictionary mapping tool names to their instances
        """
        with self._global_lock:
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
        with self._global_lock:
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
        with self._global_lock:
            self._tools[tool_name] = LazyToolLoader(tool_class)
            self._state[tool_name] = ModelWeightState.DISK
            # Create a tool-specific lock if it doesn't exist
            if tool_name not in self._locks:
                self._locks[tool_name] = threading.RLock()
            logging.info(f"Tool '{tool_name}' registered for lazy loading.")

    def list_tools(self) -> Dict[str, Any]:
        """
        List all registered tools.

        Returns:
            Dictionary with tool names as keys and tool instances as values
        """
        with self._global_lock:
            return {k: v for k, v in self._tools.items()}

    def preload(self, tool_name: Optional[str] = None) -> None:
        """
        Preload tool instances into memory.
        Args:
            tool_name: The name of the tool to preload, or None to preload all
        """
        if tool_name is None:
            # When preloading all tools, use the global lock to read the tool list
            # but use individual locks for each tool's preload operation
            with self._global_lock:
                tool_items = list(self._tools.items())

            for name, tool in tool_items:
                # Get the tool-specific lock
                with self._global_lock:
                    if name not in self._locks:
                        self._locks[name] = threading.RLock()
                    tool_lock = self._locks[name]

                # Use the tool-specific lock for preloading
                with tool_lock:
                    # Check state again after acquiring the lock
                    with self._global_lock:
                        if self._state[name] != ModelWeightState.DISK:
                            continue

                    if hasattr(tool, 'preload') and callable(tool.preload):
                        tool.preload()

                        with self._global_lock:
                            self._state[name] = ModelWeightState.CPU

                        logging.info(f"Tool '{name}' preloaded into memory.")
        else:
            # For a specific tool, first check/register with the global lock
            with self._global_lock:
                if tool_name not in self._tools:
                    self.register(tool_name, MODEL_MAP[tool_name]())

                if self._state[tool_name] != ModelWeightState.DISK:
                    return

                if tool_name not in self._locks:
                    self._locks[tool_name] = threading.RLock()
                tool_lock = self._locks[tool_name]
                tool = self._tools[tool_name]

            # Then use the tool-specific lock for the preload operation
            with tool_lock:
                # Check state again after acquiring the lock
                with self._global_lock:
                    if self._state[tool_name] != ModelWeightState.DISK:
                        return

                if hasattr(tool, 'preload') and callable(tool.preload):
                    tool.preload()

                    with self._global_lock:
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
            # When loading all tools, use the global lock to read the tool list
            # but use individual locks for each tool's load operation
            with self._global_lock:
                tool_items = list(self._tools.items())

            for name, tool in tool_items:
                # Get the tool-specific lock
                with self._global_lock:
                    if name not in self._locks:
                        self._locks[name] = threading.RLock()
                    tool_lock = self._locks[name]

                # Use the tool-specific lock for loading
                with tool_lock:
                    # Check state again after acquiring the lock
                    with self._global_lock:
                        if self._state[name] != ModelWeightState.CPU:
                            continue

                    if hasattr(tool, 'load') and callable(tool.load):
                        tool.load()

                        with self._global_lock:
                            self._state[name] = ModelWeightState.GPU

                        logging.info(f"Tool '{name}' loaded into GPU memory.")
        else:
            # For a specific tool, first check with the global lock
            with self._global_lock:
                if tool_name not in self._tools:
                    logging.warning(f"Tool '{tool_name}' not found in registry.")
                    return

                if self._state[tool_name] != ModelWeightState.CPU:
                    logging.info(f"Tool '{tool_name}' not in CPU(but in {self._state[tool_name]}) , cannot load to GPU")
                    return

                if tool_name not in self._locks:
                    self._locks[tool_name] = threading.RLock()
                tool_lock = self._locks[tool_name]
                tool = self._tools[tool_name]

            # Then use the tool-specific lock for the load operation
            with tool_lock:
                # Check state again after acquiring the lock
                with self._global_lock:
                    if self._state[tool_name] != ModelWeightState.CPU:
                        return

                if hasattr(tool, 'load') and callable(tool.load):
                    tool.load()

                    with self._global_lock:
                        self._state[tool_name] = ModelWeightState.GPU

                    logging.info(f"Tool '{tool_name}' loaded into GPU memory.")

    def counter_add(self, tool_name: str) -> None:
        """
        Increment the usage counter for a tool.

        Args:
            tool_name: The name of the tool to increment the counter for
        """
        with self._global_lock:
            if tool_name not in self._counter:
                self._counter[tool_name] = 0
            self._counter[tool_name] += 1

    def swap(self, tool_name: Optional[str] = None) -> None:
        """
        Swap tool instances from GPU to CPU memory.

        Args:
            tool_name: The name of the tool to swap, or None to swap all
        """
        if tool_name is None:
            # When swapping all tools, use the global lock to read the tool list
            # but use individual locks for each tool's swap operation
            with self._global_lock:
                tool_items = list(self._tools.items())

            for name, tool in tool_items:
                # Get the tool-specific lock
                with self._global_lock:
                    if name not in self._locks:
                        self._locks[name] = threading.RLock()
                    tool_lock = self._locks[name]

                # Use the tool-specific lock for swapping
                with tool_lock:
                    # Check state again after acquiring the lock
                    with self._global_lock:
                        if self._state[name] != ModelWeightState.GPU:
                            continue

                    if hasattr(tool, 'swap') and callable(tool.swap):
                        tool.swap()

                        with self._global_lock:
                            self._state[name] = ModelWeightState.CPU

                        logging.info(f"Tool '{name}' swapped from GPU to CPU memory.")
        else:
            # For a specific tool, first check with the global lock
            with self._global_lock:
                if tool_name not in self._tools:
                    logging.warning(f"Tool '{tool_name}' not found in registry.")
                    return

                if self._state[tool_name] != ModelWeightState.GPU:
                    logging.info(f"Tool '{tool_name}' not in GPU state, cannot swap to CPU.")
                    return

                if tool_name not in self._locks:
                    self._locks[tool_name] = threading.RLock()
                tool_lock = self._locks[tool_name]
                tool = self._tools[tool_name]

            # Then use the tool-specific lock for the swap operation
            with tool_lock:
                # Check state again after acquiring the lock
                with self._global_lock:
                    if self._state[tool_name] != ModelWeightState.GPU:
                        return

                if hasattr(tool, 'swap') and callable(tool.swap):
                    tool.swap()

                    with self._global_lock:
                        self._state[tool_name] = ModelWeightState.CPU

                    logging.info(f"Tool '{tool_name}' swapped from GPU to CPU memory.")

    def discord(self, tool_name: Optional[str] = None) -> None:
        """
        Clear tool instances from the registry.

        Args:
            tool_name: The name of the tool to clear, or None to clear all
        """
        if tool_name is None:
            # When clearing all tools, use the global lock to read the tool list
            # but use individual locks for each tool's discord operation
            with self._global_lock:
                tool_items = list(self._tools.items())

            for name, tool in tool_items:
                # Get the tool-specific lock
                with self._global_lock:
                    if name not in self._locks:
                        self._locks[name] = threading.RLock()
                    tool_lock = self._locks[name]

                # Use the tool-specific lock for discord
                with tool_lock:
                    if hasattr(tool, 'discord') and callable(tool.discord):
                        tool.discord()

                        with self._global_lock:
                            self._state[name] = ModelWeightState.DISK

                        logging.info(f"Tool '{name}' cleared to disk.")

            logging.info("All tools cleared from registry to disk.")
        else:
            # For a specific tool, first check with the global lock
            with self._global_lock:
                if tool_name not in self._tools:
                    logging.warning(f"Tool '{tool_name}' not found in registry.")
                    return

                if tool_name not in self._locks:
                    self._locks[tool_name] = threading.RLock()
                tool_lock = self._locks[tool_name]
                tool = self._tools[tool_name]

            # Then use the tool-specific lock for the discord operation
            with tool_lock:
                if hasattr(tool, 'discord') and callable(tool.discord):
                    tool.discord()

                    with self._global_lock:
                        self._state[tool_name] = ModelWeightState.DISK

                    logging.info(f"Tool '{tool_name}' cleared to disk.")


    def get_counter_list(self) -> Dict[str, int]:
        """
        Get the usage counter for all tools.

        Returns:
            Dictionary with tool names as keys and their usage counts as values
        """
        with self._global_lock:
            return self._counter.copy()


class LazyToolLoader:
    """
    Helper class for lazy-loading tool models.

    Only instantiates the tool when it's actually needed.
    Thread-safe implementation to handle concurrent access.
    """
    def __init__(self, tool_class: Type[BaseModel]):
        self.tool_class = tool_class
        self.instance = None
        self.lock = threading.RLock()

    def __getattr__(self, name):
        with self.lock:
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
