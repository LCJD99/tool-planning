from scheduler.BaseScheduler import BaseScheduler
from langchain_core.messages import ToolMessage
from typing import Dict, Any, List
import logging
import threading
import concurrent.futures
from agent.registry import tool_registry, get_tool, register_tool


class ParallelLastToolsScheduler(BaseScheduler):
    def __init__(self, model_map: Dict[str, Any], tools_map: Dict[str, Any], session_id: str = "session_unknown"):
        """
        Initialize the ParallelLastToolsScheduler.
        This scheduler executes the last two tools in parallel while maintaining the correct order in messages.

        Args:
            model_map: Dictionary mapping tool names to model classes
            tools_map: Dictionary mapping function names to callable functions
        """
        super().__init__(model_map, tools_map, session_id)
        self.execution_lock = threading.Lock()  # Lock for task execution

    def manual_preload(self, tool_names: List[str]) -> None:
        """
        Preload multiple tools into memory.

        Args:
            tool_names: List of tool names to preload
        """
        # Preload can be done in parallel by the registry
        for tool_name in tool_names:
            tool_registry.preload(tool_name)

    def _execute_single_tool(self, tool_name: str, tool_args: Dict[str, Any], call_id: str) -> tuple:
        """
        Execute a single tool and prepare its message.

        Args:
            tool_name: The name of the tool to execute
            tool_args: Arguments for the tool
            call_id: The ID of the tool call

        Returns:
            tuple: (tool_name, tool_result, call_id, has_error)
        """
        # Registry methods are already thread-safe
        tool_registry.preload(tool_name)
        # Load the tool if not already loaded
        tool_registry.load(tool_name)

        tool_registry.counter_add(tool_name)

        # Execute the tool
        logging.info(f"Executing tool: {tool_name} with args: {tool_args}")

        tool_result = self._execute_tool(tool_name, tool_args)
        tool_registry.swap(tool_name)

        # Check if there was an error in tool execution
        has_error = isinstance(tool_result, dict) and "error" in tool_result

        return (tool_name, tool_result, call_id, has_error)

    def execute(self, taskid: int) -> List[ToolMessage]:
        """
        Execute a task by its ID.
        If there are multiple tools, the last two tools will be executed in parallel,
        but their results will be added to the messages list in the correct order.

        Args:
            taskid: ID of the task to execute

        Returns:
            Result of the task execution
        """
        with self.execution_lock:
            tools = self.task_tools[taskid]
            messages = []
            has_tool_execution_error = False

            # If there are less than 3 tools, execute them serially
            if len(tools) < 3:
                for idx, tool_call in enumerate(tools):
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    call_id = tool_call['id']

                    result = self._execute_single_tool(tool_name, tool_args, call_id)
                    _, tool_result, call_id, has_error = result

                    if has_error:
                        has_tool_execution_error = True

                    # Add the tool result to the conversation
                    tool_message = ToolMessage(
                        content=str(tool_result),
                        tool_call_id=call_id
                    )
                    messages.append(tool_message)
            else:
                # Execute all but the last two tools serially
                for idx, tool_call in enumerate(tools[:-2]):
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    call_id = tool_call['id']

                    result = self._execute_single_tool(tool_name, tool_args, call_id)
                    _, tool_result, call_id, has_error = result

                    if has_error:
                        has_tool_execution_error = True

                    # Add the tool result to the conversation
                    tool_message = ToolMessage(
                        content=str(tool_result),
                        tool_call_id=call_id
                    )
                    messages.append(tool_message)

                # Execute the last two tools in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    # Submit the last two tools for parallel execution
                    future_results = []
                    for tool_call in tools[-2:]:
                        tool_name = tool_call['name']
                        tool_args = tool_call['args']
                        call_id = tool_call['id']
                        future = executor.submit(self._execute_single_tool, tool_name, tool_args, call_id)
                        future_results.append((future, tools.index(tool_call)))

                    # Wait for all futures to complete and collect results in original order
                    sorted_results = sorted(future_results, key=lambda x: x[1])
                    for future, original_idx in sorted_results:
                        _, tool_result, call_id, has_error = future.result()

                        if has_error:
                            has_tool_execution_error = True

                        # Add the tool result to the conversation in the correct order
                        tool_message = ToolMessage(
                            content=str(tool_result),
                            tool_call_id=call_id
                        )
                        messages.append(tool_message)

            return messages
