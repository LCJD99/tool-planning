import requests
import numpy as np
import time
import json
import argparse
import logging
import asyncio
from typing import List, Dict, Any
from benchmark.openagi import OpenAGI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("request_client.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("request_client")

class ApiClient:
    """Client for making requests to the Multi-Model Agent API with Poisson distributed intervals."""
    
    def __init__(self, base_url: str = "http://localhost:8001", rate: float = 1.0):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL for the API
            rate: Average request rate per minute (lambda parameter for Poisson distribution)
        """
        self.base_url = base_url
        self.rate = rate
        logger.info(f"Initialized API client with base URL: {base_url}, rate: {rate:.2f} req/min")
        
    def generate_intervals(self, num_requests: int) -> List[float]:
        rate_per_second = self.rate / 60
        
        # For a Poisson process, the time between events follows an exponential distribution
        # with parameter lambda = rate
        intervals = np.random.exponential(scale=1/rate_per_second, size=num_requests)
        logger.debug(f"Generated {num_requests} intervals with mean: {np.mean(intervals):.2f}s")
        return intervals.tolist()
    
    def send_request(self, prompt: str, max_iterations: int = 5) -> Dict[str, Any]:
        """
        Send a single request to the API.
        
        Args:
            prompt: The prompt to process
            max_iterations: Maximum number of iterations
            
        Returns:
            API response as dictionary
        """
        endpoint = f"{self.base_url}/process"
        payload = {
            "prompt": prompt,
            "max_iterations": max_iterations
        }
        
        try:
            logger.debug(f"Sending request: {payload}")
            start_time = time.time()
            response = requests.post(endpoint, json=payload)
            elapsed = time.time() - start_time
            logger.info(f"Request completed, {elapsed:.2f}s")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {"error": str(e)}
            
    async def send_request_async(self, prompt: str, max_iterations: int = 5) -> Dict[str, Any]:
        """
        Send a single request to the API asynchronously.
        
        Args:
            prompt: The prompt to process
            max_iterations: Maximum number of iterations
            
        Returns:
            API response as dictionary
        """
        endpoint = f"{self.base_url}/process"
        payload = {
            "prompt": prompt,
            "max_iterations": max_iterations
        }
        
        logger.debug(f"Sending async request: {payload}")
        start_time = time.time()
        
        # Create a function that will be run in a separate thread
        def make_request():
            try:
                response = requests.post(endpoint, json=payload)
                response.raise_for_status()
                elapsed = time.time() - start_time
                logger.info(f"Async request completed, {elapsed:.2f}s")
                return response.json()
            except requests.RequestException as e:
                logger.error(f"Async request failed: {e}")
                return {"error": str(e)}
                
        # Run the request in a thread pool to avoid blocking the event loop
        return await asyncio.to_thread(make_request)
    
    def reset_agent(self) -> Dict[str, Any]:
        """Reset the agent's conversation history."""
        endpoint = f"{self.base_url}/reset"
        
        try:
            response = requests.post(endpoint)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Reset request failed: {e}")
            return {"error": str(e)}
    
    async def reset_agent_async(self) -> Dict[str, Any]:
        """Reset the agent's conversation history asynchronously."""
        endpoint = f"{self.base_url}/reset"
        
        # Create a function that will be run in a separate thread
        def make_request():
            try:
                response = requests.post(endpoint)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                logger.error(f"Async reset request failed: {e}")
                return {"error": str(e)}
                
        # Run the request in a thread pool to avoid blocking the event loop
        return await asyncio.to_thread(make_request)
    
    async def run_simulation_async(self, prompts: List[str], max_iterations: int = 5, num_requests: int = 10) -> None:
        """
        Run a simulation sending requests asynchronously with Poisson distributed intervals.
        
        Args:
            prompts: List of prompts to send
            max_iterations: Maximum number of iterations for each request
        """
        intervals = self.generate_intervals(num_requests)
        prompt_len = len(prompts)
        
        logger.info(f"Starting async simulation with {num_requests} requests")
        
        # Keep track of all pending tasks
        pending_tasks = []

        for i, interval in enumerate(intervals):
            prompt = prompts[i % prompt_len]
            request_index = i + 1
            task = asyncio.create_task(self._handle_async_request(prompt, max_iterations, request_index, num_requests))
            pending_tasks.append(task)
            
            # Wait for the next interval if this isn't the last request
            if i < num_requests - 1:
                logger.info(f"Waiting {interval:.2f}s before next request")
                await asyncio.sleep(interval)
        
        # Wait for all requests to complete
        logger.info("All requests sent, waiting for responses...")
        await asyncio.gather(*pending_tasks)
        
        logger.info("Async simulation completed")
    
    async def _handle_async_request(self, prompt: str, max_iterations: int, request_index: int, total_requests: int) -> None:
        """
        Handle a single async request, including logging.
        
        Args:
            prompt: The prompt to process
            max_iterations: Maximum iterations
            request_index: Current request number
            total_requests: Total number of requests
        """
        logger.info(f"Sending request {request_index}/{total_requests}")
        start_time = time.time()
        response = await self.send_request_async(prompt, max_iterations)
        elapsed = time.time() - start_time
        
        # Log the response
        if "error" in response:
            logger.error(f"Request {request_index}/{total_requests} failed: {response['error']}")
        else:
            logger.info(f"Request {request_index}/{total_requests} succeeded in {elapsed:.2f}s")
            logger.debug(f"Response: {response}")
    
    def run_simulation(self, prompts: List[str], max_iterations: int = 5) -> None:
        """
        Run a simulation sending requests with Poisson distributed intervals.
        This is a synchronous wrapper around the async implementation.
        
        Args:
            prompts: List of prompts to send
            max_iterations: Maximum number of iterations for each request
        """
        asyncio.run(self.run_simulation_async(prompts, max_iterations))


async def async_main():
    """Async main entry point for the script."""
    parser = argparse.ArgumentParser(description="Client for the Multi-Model Agent API with Poisson distributed requests")
    parser.add_argument("--url", default="http://localhost:8001", help="Base URL for the API")
    parser.add_argument("--rate", type=float, default=0.5, help="Average request rate per minute (lambda)")
    parser.add_argument("--prompts", help="JSON file containing prompts to send")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum iterations for each request")
    parser.add_argument("--reset", action="store_true", help="Reset the agent before starting")
    args = parser.parse_args()
    
    # Initialize client
    client = ApiClient(base_url=args.url, rate=args.rate)
    
    # Reset agent if requested
    if args.reset:
        logger.info("Resetting agent")
        reset_response = await client.reset_agent_async()
        logger.info(f"Reset response: {reset_response}")
    
    # Load prompts
    if args.prompts:
        try:
            with open(args.prompts, 'r') as f:
                prompts = json.load(f)
                if not isinstance(prompts, list):
                    logger.error("Prompts file must contain a JSON array of strings")
                    return
        except Exception as e:
            logger.error(f"Failed to load prompts file: {e}")
            return
    else:
        openagi = OpenAGI('/home/zhangjingzhou/tool-planning/datasets/openagi/', [27])
        prompts = [openagi.get_task_prompt_from_index(i) for i in range(0, 1)]
    
    # Run the simulation asynchronously
    await client.run_simulation_async(prompts, args.max_iterations)
    
def main():
    """Main entry point for the script."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
