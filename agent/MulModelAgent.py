"""
Multi-Model Agent that handles LLM interactions and tool executions.

This class orchestrates interactions between an LLM and registered tools,
handling tool invocation and multi-turn conversations without relying on Langchain.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from agent.Tools import tools
from utils.utils import create_function_name_map
import os


class MulModelAgent:
    """
    Multi-Model Agent that orchestrates LLM interactions and tool executions.
    
    This agent handles:
    1. Processing input prompts through an LLM
    2. Parsing tool calls from LLM responses
    3. Executing tools from the function map
    4. Managing multi-turn conversation with the LLM
    """
    
    def __init__(self, model_name: str = "./qwen2.5", url: str = "http://localhost:8000",temperature: float = 0.0):
        """
        Initialize the Multi-Model Agent.
        
        Args:
            model_name: The name of the LLM to use
            temperature: Sampling temperature for LLM generation
        """
        self.llm = ChatOpenAI(model=model_name, base_url= url, temperature=temperature)
        # self.llm = ChatOpenAI(model="qwen-plus", api_key=os.getenv("DASHSCOPE_API_KEY"),base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",)
        self.messages = []  # Conversation history
        
        self.tools = tools
        # self.llm.bind_tools(self.tools)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Create function map for tool execution
        self.function_map = {}
        tool_functions = [t.func for t in tools]
        self.function_map = create_function_name_map(tool_functions)
        
        
        logging.info(f"Initialized agent with {len(tools)} tools")
        
    def bind_tools_to_model(self, tools) -> None:
        """
        Bind tools to the LLM.
        
        This enhances the LLM's ability to make appropriate tool calls.
        """
        if not self.tools_info:
            logging.warning("No tools available for the agent.")
            return
            
        self.llm_with_tools = self.llm.bind_tools(self.tools_info)
        logging.info(f"Bound {len(self.tools_info)} tools to the LLM.")
    
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
    
    def process(self, prompt: str, max_iterations: int = 10) -> str:
        """
        Process a prompt through the agent, potentially invoking tools.
        
        Args:
            prompt: The user's input prompt
            max_iterations: Maximum number of LLM-tool interaction rounds
            
        Returns:
            The final response after all tool executions
        """
            
        # Start with the user's prompt
        self.messages = [HumanMessage(content=prompt)]
        
        for iteration in range(max_iterations):
            # Get LLM response
            ai_msg = self.llm_with_tools.invoke(self.messages)
            self.messages.append(ai_msg)
            
            # Log the current iteration
            logging.info(f"Iteration {iteration + 1}/{max_iterations}")
            
            # Check for tool calls
            if not ai_msg.tool_calls:
                # LLM provided a direct answer without tool calls
                logging.info("No tool calls in LLM response, returning answer")
                return ai_msg.content
                
            # Process tool calls
            has_tool_execution_error = False
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                call_id = tool_call['id']
                
                logging.info(f"Executing tool: {tool_name} with args: {tool_args}")
                
                # Execute the tool
                tool_result = self._execute_tool(tool_name, tool_args)
                
                # Check if there was an error in tool execution
                if isinstance(tool_result, dict) and "error" in tool_result:
                    has_tool_execution_error = True
                
                # Add the tool result to the conversation
                tool_message = ToolMessage(
                    content=str(tool_result),
                    tool_call_id=call_id
                )
                self.messages.append(tool_message)
            
            if has_tool_execution_error:
                logging.warning("Encountered error during tool execution")
                # Optionally add special handling here
            
        # If we've reached max iterations, get a final response
        final_response = self.llm.invoke(self.messages)
        return final_response.content
        
    def reset(self) -> None:
        """
        Reset the agent's conversation history.
        """
        self.messages = []
        logging.info("Agent conversation history has been reset.")