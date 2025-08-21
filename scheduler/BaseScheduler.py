from typing import List, Dict, Any
from enum import Enum
from tools.models import BaseModel
import logging

class BaseScheduler():
    def __init__(self, model_map: Dict[str, BaseModel], tools_map: Dict[str, Any]):
        self.task_tools = {}
        self.function_map = tools_map
        self.model_map = model_map

    def add_tasks(self, taskid: int, tools: List[Any]) -> None:
        """
        Add tasks to the scheduler.
        
        Args:
            tasks: List of tasks to be added
        """
        self.task_tools[taskid] = tools
    
    def execute(self, taskid: int) -> str:
        """
        Execute a task by its ID.
        
        Args:
            taskid: ID of the task to execute
        
        Returns:
            Result of the task execution
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """
        Execute a tool using the function map.

        Args:
            tool_name: The name of the tool to execute
            tool_args: Arguments to pass to the tool

        Returns:
            The result of the tool execution
        """
        # Get the function name from the tool name

        # Get the function from the function map
        func = self.function_map[tool_name]

        try:
            # Execute the function with the provided arguments
            if isinstance(tool_args, dict):
                # Call the function with the arguments as kwargs
                result = func(**tool_args)
            else:
                result = func(tool_args)

            return result
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            logging.error(error_msg)
            return {"error": error_msg}