from scheduler.BaseScheduler import BaseScheduler
from langchain_core.messages import ToolMessage
from typing import Dict, Any, List
import logging
from agent.registry import tool_registry, get_tool, register_tool


class SerialAliveScheduler(BaseScheduler):
    def manual_preload(self, tool_names: List[str]) -> None:
        for tool_name in tool_names:
            tool_registry.preload(tool_name)

    def execute(self, taskid: int) -> List[ToolMessage]:
        """
        Execute a task by its ID.

        Args:
            taskid: ID of the task to execute

        Returns:
            Result of the task execution
        """
        tools = self.task_tools[taskid]
        messages = []

        for idx, tool_call in enumerate(tools):
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            call_id = tool_call['id']

            tool_registry.preload(tool_name)
            # Load the tool if not already loaded
            tool_registry.load(tool_name)

            # Execute the tool
            logging.info(f"Executing tool_{idx}: {tool_name} with args: {tool_args}")
            #TODO: state manager

            tool_result = self._execute_tool(tool_name, tool_args)

            # Check if there was an error in tool execution
            if isinstance(tool_result, dict) and "error" in tool_result:
                has_tool_execution_error = True

            # Add the tool result to the conversation
            tool_message = ToolMessage(
                content=str(tool_result),
                tool_call_id=call_id
            )
            messages.append(tool_message)

        return messages

