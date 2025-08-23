"""
Asynchronous Multi-Model Agent for handling LLM interactions and tool executions.

This class provides an asynchronous version of MulModelAgent, supporting
parallel processing of prompts and tool executions using asyncio.
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from agent.Tools import tools
from utils.utils import create_function_name_map
from scheduler.SerialAliveScheduler import SerialAliveScheduler
from tools.model_map import MODEL_MAP
from agent.registry import tool_registry
import os


class MulModelAgent:
    """
    Asynchronous Multi-Model Agent that orchestrates LLM interactions and tool executions.

    This agent handles:
    1. Processing input prompts through an LLM asynchronously
    2. Parsing tool calls from LLM responses
    3. Executing tools in parallel using AsyncParallelScheduler
    4. Managing multi-turn conversation with the LLM
    """

    def __init__(self, model: str = "./qwen2.5", api_key: str = "fake api", 
                 base_url: str = "http://localhost:8000/v1", temperature: float = 0.0,
                 max_workers: Optional[int] = None):
        """
        Initialize the Async Multi-Model Agent.

        Args:
            model: The name of the LLM model to use
            api_key: API key for accessing the LLM
            base_url: Base URL for LLM API endpoint
            temperature: Sampling temperature for LLM generation
            max_workers: Maximum number of worker threads in the thread pool
        """
        self.llm = ChatOpenAI(model=model, base_url=base_url, temperature=temperature, api_key=api_key)
        self.messages = []  # Conversation history

        self.tools = tools
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Create function map for tool execution
        tool_functions = [t.func for t in tools]
        self.function_map = create_function_name_map(tool_functions)
        
        # Create the async scheduler
        self.scheduler = SerialAliveScheduler(MODEL_MAP, self.function_map)

        logging.info(f"Initialized async agent with {len(tools)} tools")

    def bind_tools_to_model(self, tools) -> None:
        """
        Bind tools to the LLM.

        This enhances the LLM's ability to make appropriate tool calls.
        """
        if not tools:
            logging.warning("No tools available for the agent.")
            return

        self.llm_with_tools = self.llm.bind_tools(tools)
        logging.info(f"Bound {len(tools)} tools to the LLM.")

    async def process_async(self, prompt: str, max_iterations: int = 10, is_cot: bool = True) -> str:
        """
        Process a prompt asynchronously through the agent, potentially invoking tools in parallel.

        Args:
            prompt: The user's input prompt
            max_iterations: Maximum number of LLM-tool interaction rounds
            is_cot: Whether to use chain-of-thought prompting

        Returns:
            The final response after all tool executions
        """
        if not is_cot:
            prompt = f"{prompt}, plan all tool should use in only one iteration"

        return await self._cot_process_async(prompt, max_iterations)

    async def _cot_process_async(self, prompt: str, max_iterations: int = 10) -> str:
        """
        Process a prompt through the agent asynchronously, potentially invoking tools.

        Args:
            prompt: The user's input prompt
            max_iterations: Maximum number of LLM-tool interaction rounds

        Returns:
            The final response after all tool executions
        """
        # Start with the user's prompt
        self.messages = [HumanMessage(content=prompt)]
        
        # Preload commonly used tools
        self.scheduler.manual_preload(['image_super_resolution', 'image_captioning', 'machine_translation'])

        for iteration in range(max_iterations):
            # Get LLM response
            logging.info(f"StageRecord: LLM Request{iteration}")
            
            # Run LLM invoke in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            ai_msg = await loop.run_in_executor(
                None, lambda: self.llm_with_tools.invoke(self.messages)
            )
            self.messages.append(ai_msg)

            # Log the current iteration
            logging.info(f"Iteration {iteration + 1}/{max_iterations}")

            # Check for tool calls
            if not ai_msg.tool_calls:
                # Release GPU resources when done
                await loop.run_in_executor(
                    None, lambda: tool_registry.swap()
                ) 
                logging.info("No tool calls in LLM response, returning answer")
                return ai_msg.content

            # Process tool calls in parallel
            self.scheduler.add_tasks(iteration, ai_msg.tool_calls)
            tool_messages = await loop.run_in_executor(
                None, lambda: self.scheduler.execute(iteration)
            )
            self.messages.extend(tool_messages)

            # Check for tool execution errors
            has_tool_execution_error = any(
                "error" in msg.content for msg in tool_messages 
                if isinstance(msg.content, str) and msg.content.startswith("{")
            )

            if has_tool_execution_error:
                logging.warning("Encountered error during tool execution")
                # Optionally add special handling here

        # If we've reached max iterations, get a final response
        final_response = await loop.run_in_executor(
            None, lambda: self.llm.invoke(self.messages)
        )
        return final_response.content

    def reset(self) -> None:
        """
        Reset the agent's conversation history.
        """
        self.messages = []
        logging.info("Agent conversation history has been reset.")
        
    def close(self) -> None:
        """
        Clean up resources used by the agent.
        """
        if hasattr(self, 'scheduler'):
            self.scheduler.close()
        logging.info("Async agent resources cleaned up.")
