"""
Tool Registry module for managing and accessing tool instances.

This module provides a centralized registry for all tool model instances,
allowing easy access, management, and memory optimization of AI tool models.
It includes thread safety mechanisms for concurrent access.
"""

from typing import Dict, Any, Optional, Type
import logging
import threading
import time
from tools.models.BaseModel import BaseModel
from enum import Enum
from tools.model_map import MODEL_MAP

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

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
                cls._instance._counter = {}  # Usage counter for analytics
                cls._instance._tools = {}    # Mapping from tool name to instance
                cls._instance._state = {}    # Current state of each tool (DISK/CPU/GPU)
                cls._instance._locks = {}    # Tool-specific locks
                cls._instance._global_lock = threading.RLock()  # Global registry lock
                cls._instance._ref_count = {}  # Reference counting for active tool usage
                cls._instance._oom_count = {}  # OOM counter for each tool
                cls._instance._oom_waiting_lock = threading.Lock()  # Lock for OOM waiting mechanism
                cls._instance._oom_waiting_tool = None  # Tool currently causing OOM and waiting for retry
                cls._instance._oom_retry_event = threading.Event()  # Event to signal when to retry OOM tool
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

    def _is_oom_error(self, exception: Exception) -> bool:
        """
        Check if an exception is an Out of Memory error.
        
        Args:
            exception: The exception to check
            
        Returns:
            True if it's an OOM error, False otherwise
        """
        # First, check if it's the specific torch CUDA OOM error
        if TORCH_AVAILABLE and torch is not None:
            try:
                if isinstance(exception, torch.cuda.OutOfMemoryError):
                    return True
            except AttributeError:
                # torch.cuda.OutOfMemoryError might not be available in older versions
                pass
        
        # Fallback to string matching for cases where torch is not available
        # or for other types of OOM errors
        error_message = str(exception).lower()
        oom_keywords = [
            'out of memory',
            'cuda out of memory',
            'cudnn_status_not_enough_workspace',
            'cuda error: out of memory',
            'runtime error: cuda out of memory',
            'torch.cuda.outofmemoryerror',
            'outofmemoryerror'
        ]
        return any(keyword in error_message for keyword in oom_keywords)

    def _wait_for_oom_retry(self, tool_name: str) -> None:
        """
        Wait for OOM retry mechanism when another tool is causing OOM.
        
        Args:
            tool_name: Name of the tool requesting to load
        """
        with self._oom_waiting_lock:
            if self._oom_waiting_tool is not None and self._oom_waiting_tool != tool_name:
                logging.info(f"Tool '{tool_name}' waiting for OOM tool '{self._oom_waiting_tool}' to complete")
                waiting_tool = self._oom_waiting_tool
            else:
                return
                
        # Wait outside the lock to avoid deadlock
        if waiting_tool is not None:
            self._oom_retry_event.wait()

    def _set_oom_waiting_tool(self, tool_name: str) -> None:
        """
        Set the tool that is currently waiting for OOM retry.
        
        Args:
            tool_name: Name of the tool causing OOM
        """
        with self._oom_waiting_lock:
            self._oom_waiting_tool = tool_name
            self._oom_retry_event.clear()
            logging.info(f"Set OOM waiting tool to '{tool_name}'")

    def _clear_oom_waiting_tool(self, tool_name: str) -> None:
        """
        Clear the OOM waiting tool and signal other threads to continue.
        
        Args:
            tool_name: Name of the tool that was causing OOM
        """
        with self._oom_waiting_lock:
            if self._oom_waiting_tool == tool_name:
                self._oom_waiting_tool = None
                self._oom_retry_event.set()
                logging.info(f"Cleared OOM waiting tool '{tool_name}', signaling other threads")

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



    def load(self, tool_name: Optional[str] = None, max_retries: int = 5, retry_delay: float = 1.0) -> None:
        """
        Load tool instances into gpu memory with OOM handling.

        Args:
            tool_name: The name of the tool to load, or None to load all
            max_retries: Maximum number of retries for OOM recovery
            retry_delay: Delay between retries in seconds
        """
        if tool_name is None:
            # When loading all tools, use the global lock to read the tool list
            # but use individual locks for each tool's load operation
            with self._global_lock:
                tool_items = list(self._tools.items())

            for name, tool in tool_items:
                self.load(name, max_retries, retry_delay)  # Recursive call for each tool
        else:
            # Wait if another tool is currently handling OOM
            self._wait_for_oom_retry(tool_name)
            
            # For a specific tool, first check with the global lock
            with self._global_lock:
                if tool_name not in self._tools:
                    logging.warning(f"Tool '{tool_name}' not found in registry.")
                    return

                if self._state[tool_name] == ModelWeightState.GPU:
                    # If already in GPU, just increment reference count
                    if tool_name not in self._ref_count:
                        self._ref_count[tool_name] = 0
                    self._ref_count[tool_name] += 1
                    logging.info(f"Tool '{tool_name}' already in GPU, incremented reference count to {self._ref_count[tool_name]}")
                    return
                elif self._state[tool_name] != ModelWeightState.CPU:
                    logging.info(f"Tool '{tool_name}' not in CPU (but in {self._state[tool_name]}), cannot load to GPU")
                    return

                if tool_name not in self._locks:
                    self._locks[tool_name] = threading.RLock()
                tool_lock = self._locks[tool_name]
                tool = self._tools[tool_name]

            # Then use the tool-specific lock for the load operation
            with tool_lock:
                # Check state again after acquiring the lock
                with self._global_lock:
                    if self._state[tool_name] == ModelWeightState.GPU:
                        # Double check if state changed while waiting for lock
                        if tool_name not in self._ref_count:
                            self._ref_count[tool_name] = 0
                        self._ref_count[tool_name] += 1
                        logging.info(f"Tool '{tool_name}' already in GPU, incremented reference count to {self._ref_count[tool_name]}")
                        return
                    elif self._state[tool_name] != ModelWeightState.CPU:
                        return

                if hasattr(tool, 'load') and callable(tool.load):
                    # Attempt to load with OOM handling
                    retry_count = 0
                    while retry_count < max_retries:
                        try:
                            tool.load()

                            with self._global_lock:
                                self._state[tool_name] = ModelWeightState.GPU
                                if tool_name not in self._ref_count:
                                    self._ref_count[tool_name] = 0
                                self._ref_count[tool_name] += 1

                            logging.info(f"Tool '{tool_name}' loaded into GPU memory. Reference count: {self._ref_count[tool_name]}")
                            
                            # Clear OOM waiting if this tool was causing it
                            self._clear_oom_waiting_tool(tool_name)
                            return
                            
                        except Exception as e:
                            # Check for CUDA OOM first with specific exception type
                            is_oom = False
                            if TORCH_AVAILABLE and torch is not None:
                                try:
                                    is_oom = isinstance(e, torch.cuda.OutOfMemoryError)
                                except AttributeError:
                                    # Fallback to string-based detection
                                    is_oom = self._is_oom_error(e)
                            else:
                                is_oom = self._is_oom_error(e)
                            
                            if is_oom:
                                retry_count += 1
                                
                                # Record OOM occurrence
                                with self._global_lock:
                                    if tool_name not in self._oom_count:
                                        self._oom_count[tool_name] = 0
                                    self._oom_count[tool_name] += 1
                                
                                logging.warning(f"OOM error loading tool '{tool_name}' (attempt {retry_count}/{max_retries}): {e}")
                                logging.info(f"Total OOM count for '{tool_name}': {self._oom_count[tool_name]}")
                                
                                if retry_count < max_retries:
                                    # Set this tool as the OOM waiting tool to block other loads
                                    self._set_oom_waiting_tool(tool_name)
                                    
                                    # Try to swap some tools to free memory
                                    logging.info(f"Attempting to free GPU memory by swapping tools...")
                                    self._attempt_memory_recovery()
                                    
                                    # Wait for either timeout or memory availability signal
                                    logging.info(f"Waiting {retry_delay}s before retry or for memory to become available...")
                                    
                                    # Wait for either the timeout or a memory availability signal
                                    self._oom_retry_event.wait(timeout=retry_delay)
                                    
                                    # Clear the event for the next iteration
                                    with self._oom_waiting_lock:
                                        self._oom_retry_event.clear()
                                else:
                                    logging.error(f"Failed to load tool '{tool_name}' after {max_retries} attempts due to OOM")
                                    self._clear_oom_waiting_tool(tool_name)
                                    raise e
                            else:
                                # Non-OOM error, re-raise immediately
                                logging.error(f"Error loading tool '{tool_name}': {e}")
                                raise e
                else:
                    logging.error(f"Tool '{tool_name}' does not implement load method.")

    def _attempt_memory_recovery(self) -> None:
        """
        Attempt to recover GPU memory by swapping tools that have zero reference count.
        """
        with self._global_lock:
            tools_to_swap = []
            for name, state in self._state.items():
                if (state == ModelWeightState.GPU and 
                    (name not in self._ref_count or self._ref_count[name] == 0)):
                    tools_to_swap.append(name)
            
        if tools_to_swap:
            logging.info(f"Attempting to swap {len(tools_to_swap)} tools to free GPU memory: {tools_to_swap}")
            for tool_name in tools_to_swap:
                try:
                    self.swap(tool_name)
                except Exception as e:
                    logging.warning(f"Failed to swap tool '{tool_name}': {e}")
        else:
            logging.info("No tools available for swapping to free GPU memory")

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
        
        This method decrements the reference count for a tool and only
        actually swaps the model when the reference count reaches zero.

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
                    # Check state and reference count
                    with self._global_lock:
                        if self._state[name] != ModelWeightState.GPU:
                            continue
                            
                        # Decrement reference count
                        if name in self._ref_count and self._ref_count[name] > 0:
                            self._ref_count[name] -= 1
                            logging.info(f"Decremented reference count for tool '{name}' to {self._ref_count[name]}")
                            
                            # Only swap if reference count reaches zero
                            if self._ref_count[name] > 0:
                                logging.info(f"Tool '{name}' still in use by {self._ref_count[name]} other processes, not swapping.")
                                continue

                    if hasattr(tool, 'swap') and callable(tool.swap):
                        tool.swap()

                        with self._global_lock:
                            self._state[name] = ModelWeightState.CPU

                        logging.info(f"Tool '{name}' swapped from GPU to CPU memory, no more active users.")
                        
                        # Signal OOM waiting tools that memory may be available
                        self._signal_memory_available()
        else:
            # For a specific tool, first check with the global lock
            with self._global_lock:
                if tool_name not in self._tools:
                    logging.warning(f"Tool '{tool_name}' not found in registry.")
                    return

                if self._state[tool_name] != ModelWeightState.GPU:
                    logging.info(f"Tool '{tool_name}' not in GPU state, cannot swap to CPU.")
                    return
                    
                # Decrement reference count
                if tool_name in self._ref_count and self._ref_count[tool_name] > 0:
                    self._ref_count[tool_name] -= 1
                    logging.info(f"Decremented reference count for tool '{tool_name}' to {self._ref_count[tool_name]}")
                    
                    # Only proceed with swap if reference count is zero
                    if self._ref_count[tool_name] > 0:
                        logging.info(f"Tool '{tool_name}' still in use by {self._ref_count[tool_name]} other processes, not swapping.")
                        return

                if tool_name not in self._locks:
                    self._locks[tool_name] = threading.RLock()
                tool_lock = self._locks[tool_name]
                tool = self._tools[tool_name]

            # Then use the tool-specific lock for the swap operation
            with tool_lock:
                # Double check state and reference count after acquiring the lock
                with self._global_lock:
                    if self._state[tool_name] != ModelWeightState.GPU:
                        return
                    
                    # Double check reference count
                    if tool_name in self._ref_count and self._ref_count[tool_name] > 0:
                        logging.info(f"Tool '{tool_name}' reference count changed while waiting for lock, not swapping.")
                        return

                if hasattr(tool, 'swap') and callable(tool.swap):
                    tool.swap()

                    with self._global_lock:
                        self._state[tool_name] = ModelWeightState.CPU

                    logging.info(f"Tool '{tool_name}' swapped from GPU to CPU memory, no more active users.")
                    
                    # Signal OOM waiting tools that memory may be available
                    self._signal_memory_available()

    def discord(self, tool_name: Optional[str] = None, force: bool = False) -> None:
        """
        Clear tool instances from the registry.

        Args:
            tool_name: The name of the tool to clear, or None to clear all
            force: If True, forcibly clears the tool regardless of reference count
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
                    # Check reference count unless force=True
                    with self._global_lock:
                        if not force and name in self._ref_count and self._ref_count[name] > 0:
                            logging.warning(f"Tool '{name}' still in use by {self._ref_count[name]} processes, not clearing.")
                            continue
                        
                        # Reset reference count
                        self._ref_count[name] = 0

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

                # Check reference count unless force=True
                if not force and tool_name in self._ref_count and self._ref_count[tool_name] > 0:
                    logging.warning(f"Tool '{tool_name}' still in use by {self._ref_count[tool_name]} processes, not clearing.")
                    return

                if tool_name not in self._locks:
                    self._locks[tool_name] = threading.RLock()
                tool_lock = self._locks[tool_name]
                tool = self._tools[tool_name]

            # Then use the tool-specific lock for the discord operation
            with tool_lock:
                # Double check reference count after acquiring lock
                with self._global_lock:
                    if not force and tool_name in self._ref_count and self._ref_count[tool_name] > 0:
                        logging.warning(f"Tool '{tool_name}' reference count changed while waiting for lock, not clearing.")
                        return
                        
                    # Reset reference count
                    self._ref_count[tool_name] = 0

                if hasattr(tool, 'discord') and callable(tool.discord):
                    tool.discord()

                    with self._global_lock:
                        self._state[tool_name] = ModelWeightState.DISK

                    logging.info(f"Tool '{tool_name}' cleared to disk.")


    def _signal_memory_available(self) -> None:
        """
        Signal that GPU memory may be available after a swap operation.
        This can help OOM waiting tools to retry sooner.
        """
        # We don't clear the waiting tool here, just set the event briefly
        # to allow the waiting tool to retry its load operation
        with self._oom_waiting_lock:
            if self._oom_waiting_tool is not None:
                logging.info("Signaling that GPU memory may be available after swap")
                # Temporarily set the event to allow retry, but don't clear the waiting tool yet
                self._oom_retry_event.set()
                # The event will be cleared again when the OOM tool starts its next retry

    def get_counter_list(self) -> Dict[str, int]:
        """
        Get the usage counter for all tools.

        Returns:
            Dictionary with tool names as keys and their usage counts as values
        """
        with self._global_lock:
            return self._counter.copy()
            
    def get_reference_counts(self) -> Dict[str, int]:
        """
        Get the current reference counts for all tools.
        
        This is useful for debugging and monitoring which tools are actively being used.

        Returns:
            Dictionary with tool names as keys and their active reference counts as values
        """
        with self._global_lock:
            return self._ref_count.copy()

    def get_oom_counts(self) -> Dict[str, int]:
        """
        Get the OOM (Out of Memory) counts for all tools.
        
        This is useful for monitoring which tools frequently encounter memory issues.

        Returns:
            Dictionary with tool names as keys and their OOM counts as values
        """
        with self._global_lock:
            return self._oom_count.copy()

    def get_oom_status(self) -> Dict[str, Any]:
        """
        Get the current OOM status including waiting tool and statistics.
        
        Returns:
            Dictionary containing OOM status information
        """
        with self._global_lock:
            with self._oom_waiting_lock:
                return {
                    'waiting_tool': self._oom_waiting_tool,
                    'oom_counts': self._oom_count.copy(),
                    'retry_event_set': self._oom_retry_event.is_set()
                }

    def execute_tool_with_oom_handling(self, tool_name: str, func, *args, **kwargs) -> Any:
        """
        Execute a tool function with OOM handling and retry mechanism.
        
        This method provides centralized OOM handling for tool execution,
        including retry logic and coordination with other threads.
        
        Args:
            tool_name: Name of the tool being executed
            func: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function execution
            
        Raises:
            Exception: If execution fails after all retries or timeout
        """
        max_retry_time = 15.0  # Maximum retry time in seconds
        retry_interval = 1.0   # Initial retry interval in seconds
        max_retry_interval = 3.0  # Maximum retry interval
        
        start_time = time.time()
        attempt = 0
        
        while True:
            attempt += 1
            try:
                # Wait if another tool is currently handling OOM
                self._wait_for_oom_retry(tool_name)
                
                # Execute the function
                logging.debug(f"Executing {tool_name}, attempt {attempt}")
                result = func(*args, **kwargs)
                
                # If successful, clear any OOM waiting state for this tool
                self._clear_oom_waiting_tool(tool_name)
                return result
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                
                # Check if it's an OOM error
                if self._is_oom_error(e):
                    # Update OOM statistics
                    with self._global_lock:
                        self._oom_count[tool_name] = self._oom_count.get(tool_name, 0) + 1
                    
                    logging.warning(f"OOM error in tool '{tool_name}' execution (attempt {attempt}): {str(e)}")
                    
                    # Check if we've exceeded the maximum retry time
                    if elapsed_time >= max_retry_time:
                        self._clear_oom_waiting_tool(tool_name)
                        error_msg = f"Tool '{tool_name}' execution failed after {max_retry_time}s of OOM retries. Giving up."
                        logging.error(error_msg)
                        raise Exception(error_msg) from e
                    
                    # Set this tool as the OOM waiting tool to coordinate with other threads
                    self._set_oom_waiting_tool(tool_name)
                    
                    # Attempt memory recovery
                    try:
                        logging.info(f"Attempting memory recovery for OOM in tool '{tool_name}'")
                        self._attempt_memory_recovery()
                        
                        # Signal that memory might be available
                        self._signal_memory_available()
                        
                    except Exception as recovery_error:
                        logging.warning(f"Memory recovery failed: {str(recovery_error)}")
                    
                    # Calculate next retry interval with exponential backoff
                    remaining_time = max_retry_time - elapsed_time
                    next_interval = min(retry_interval * (1.5 ** (attempt - 1)), max_retry_interval)
                    actual_wait = min(next_interval, remaining_time - 0.1)  # Leave some buffer
                    
                    if actual_wait > 0:
                        logging.info(f"Waiting {actual_wait:.1f}s before retrying tool '{tool_name}' (attempt {attempt})")
                        time.sleep(actual_wait)
                    else:
                        # No time left for retry
                        self._clear_oom_waiting_tool(tool_name)
                        error_msg = f"Tool '{tool_name}' execution timeout after {max_retry_time}s"
                        logging.error(error_msg)
                        raise Exception(error_msg) from e
                        
                else:
                    # Non-OOM error, don't retry
                    self._clear_oom_waiting_tool(tool_name)
                    logging.error(f"Non-OOM error in tool '{tool_name}' execution: {str(e)}")
                    raise e


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

def execute_tool_with_oom_handling(tool_name: str, func, *args, **kwargs) -> Any:
    """
    Execute a tool function with OOM handling using the global registry.
    
    Args:
        tool_name: Name of the tool being executed
        func: The function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        The result of the function execution
    """
    return tool_registry.execute_tool_with_oom_handling(tool_name, func, *args, **kwargs)
