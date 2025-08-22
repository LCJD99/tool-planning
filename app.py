"""
FastAPI Wrapper for tooling planning
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import uvicorn
import signal
import sys
from logger.config import setup_logger


from agent.MulModelAgent import MulModelAgent

logger = logging.getLogger("uvicorn.error")

# Initialize FastAPI app
app = FastAPI(title="Multi-Model Agent API",
              description="API for interacting with the Multi-Model Agent")

# Initialize the agent
agent = MulModelAgent()

# Custom signal handler to avoid rich_toolkit conflicts
def handle_signal(signum, frame):
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


# Define API Models
class PromptRequest(BaseModel):
    prompt: str
    max_iterations: Optional[int] = 5

class AgentResponse(BaseModel):
    response: str

@app.post("/process", response_model=AgentResponse)
def process_prompt(request: PromptRequest):
    """
    Process a prompt through the Multi-Model Agent.

    This endpoint accepts a prompt and optional max_iterations parameter,
    and returns the agent's response after processing the prompt and
    potentially executing tools.
    """
    try:
        response = agent.process(request.prompt, request.max_iterations)
        return AgentResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing prompt: {str(e)}")

@app.post("/reset")
def reset_agent():
    """
    Reset the agent's conversation history.
    """
    agent.reset()
    return {"status": "success", "message": "Agent reset successfully"}

def main():
    """Run the FastAPI application."""
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        log_level="debug"
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Server shutting down gracefully...")
        sys.exit(0)
