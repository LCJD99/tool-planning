#!/usr/bin/env python3
"""
Test script for OOM handling in ToolRegistry.

This script simulates OOM conditions to test the registry's OOM handling mechanisms.
"""

import threading
import time
import logging
from typing import Optional
from unittest.mock import Mock, patch
from agent.registry import tool_registry, ToolRegistry, ModelWeightState

# Try to import torch for more realistic OOM simulation
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

class MockTool:
    """Mock tool that can simulate OOM errors."""
    
    def __init__(self, name: str, should_oom: bool = False, oom_count: int = 2):
        self.name = name
        self.should_oom = should_oom
        self.oom_count = oom_count
        self.current_oom_count = 0
        self.loaded = False
        self.swapped = False
        
    def preload(self):
        logging.info(f"MockTool {self.name}: preload called")
        
    def load(self):
        logging.info(f"MockTool {self.name}: load called")
        if self.should_oom and self.current_oom_count < self.oom_count:
            self.current_oom_count += 1
            
            # Try to raise the actual torch CUDA OOM error if available
            if TORCH_AVAILABLE and torch is not None:
                try:
                    # Create a realistic CUDA OOM error
                    if hasattr(torch.cuda, 'OutOfMemoryError'):
                        raise torch.cuda.OutOfMemoryError(f"CUDA out of memory for {self.name} (attempt {self.current_oom_count})")
                except AttributeError:
                    pass
            
            # Fallback to generic RuntimeError with OOM message
            raise RuntimeError(f"CUDA out of memory for {self.name} (attempt {self.current_oom_count})")
            
        self.loaded = True
        logging.info(f"MockTool {self.name}: successfully loaded")
        
    def swap(self):
        logging.info(f"MockTool {self.name}: swap called")
        self.loaded = False
        self.swapped = True
        
    def discord(self):
        logging.info(f"MockTool {self.name}: discord called")
        self.loaded = False
        self.swapped = False

def test_oom_handling():
    """Test OOM handling with multiple tools and threads."""
    
    # Create a fresh registry instance for testing
    registry = ToolRegistry()
    
    # Register mock tools
    oom_tool = MockTool("oom_tool", should_oom=True, oom_count=2)
    normal_tool1 = MockTool("normal_tool1")
    normal_tool2 = MockTool("normal_tool2")
    victim_tool = MockTool("victim_tool")  # Tool that will be swapped to free memory
    
    registry.register("oom_tool", oom_tool)
    registry.register("normal_tool1", normal_tool1)
    registry.register("normal_tool2", normal_tool2)
    registry.register("victim_tool", victim_tool)
    
    # Preload all tools to CPU
    for tool_name in ["oom_tool", "normal_tool1", "normal_tool2", "victim_tool"]:
        registry.preload(tool_name)
    
    # Load victim tool to GPU first (so it can be swapped later)
    registry.load("victim_tool")
    # Immediately swap it to simulate available memory
    registry.swap("victim_tool")
    
    def load_tool_thread(tool_name: str, thread_id: int):
        """Thread function to load a tool."""
        try:
            logging.info(f"Thread {thread_id}: Starting to load {tool_name}")
            registry.load(tool_name)
            logging.info(f"Thread {thread_id}: Successfully loaded {tool_name}")
        except Exception as e:
            logging.error(f"Thread {thread_id}: Failed to load {tool_name}: {e}")
    
    # Start multiple threads trying to load tools
    threads = []
    
    # Thread 1: Load OOM tool (will cause OOM initially)
    thread1 = threading.Thread(target=load_tool_thread, args=("oom_tool", 1))
    threads.append(thread1)
    
    # Thread 2: Load normal tool (should wait for OOM tool to complete)
    thread2 = threading.Thread(target=load_tool_thread, args=("normal_tool1", 2))
    threads.append(thread2)
    
    # Thread 3: Load another normal tool (should also wait)
    thread3 = threading.Thread(target=load_tool_thread, args=("normal_tool2", 3))
    threads.append(thread3)
    
    # Start all threads
    for thread in threads:
        thread.start()
        time.sleep(0.1)  # Small delay to ensure ordering
    
    # Monitor OOM status
    def monitor_oom_status():
        """Monitor and log OOM status."""
        for i in range(20):  # Monitor for 20 seconds
            time.sleep(1)
            status = registry.get_oom_status()
            oom_counts = registry.get_oom_counts()
            logging.info(f"OOM Status: waiting_tool={status['waiting_tool']}, "
                        f"oom_counts={oom_counts}, retry_event_set={status['retry_event_set']}")
            
            if status['waiting_tool'] is None and all(t.is_alive() == False for t in threads):
                break
    
    monitor_thread = threading.Thread(target=monitor_oom_status)
    monitor_thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    monitor_thread.join()
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print("="*50)
    print(f"OOM Counts: {registry.get_oom_counts()}")
    print(f"Reference Counts: {registry.get_reference_counts()}")
    print(f"Final OOM Status: {registry.get_oom_status()}")
    
    # Check tool states
    for tool_name in ["oom_tool", "normal_tool1", "normal_tool2", "victim_tool"]:
        tool = registry.get(tool_name)
        print(f"{tool_name}: loaded={getattr(tool, 'loaded', 'N/A')}")

def test_basic_oom_detection():
    """Test basic OOM error detection."""
    registry = ToolRegistry()
    
    # Test various OOM error messages and types
    oom_errors = [
        RuntimeError("CUDA out of memory"),
        RuntimeError("RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"),
        Exception("torch.cuda.OutOfMemoryError: CUDA out of memory"),
        Exception("OutOfMemoryError: CUDA device ran out of memory"),
    ]
    
    # Add torch.cuda.OutOfMemoryError if available
    if TORCH_AVAILABLE and torch is not None:
        try:
            if hasattr(torch.cuda, 'OutOfMemoryError'):
                oom_errors.append(torch.cuda.OutOfMemoryError("CUDA out of memory: Tried to allocate 1.00 GiB"))
        except AttributeError:
            pass
    
    non_oom_errors = [
        RuntimeError("Some other error"),
        ValueError("Invalid input"),
        Exception("Network timeout"),
    ]
    
    print("\nTesting OOM Error Detection:")
    print("-" * 30)
    
    for error in oom_errors:
        is_oom = registry._is_oom_error(error)
        error_type = type(error).__name__
        print(f"'{error_type}: {str(error)[:50]}...' -> OOM: {is_oom}")
        assert is_oom, f"Should detect OOM: {error}"
    
    for error in non_oom_errors:
        is_oom = registry._is_oom_error(error)
        error_type = type(error).__name__
        print(f"'{error_type}: {str(error)[:50]}...' -> OOM: {is_oom}")
        assert not is_oom, f"Should NOT detect OOM: {error}"
    
    print("✓ OOM detection tests passed!")

if __name__ == "__main__":
    print("Testing OOM Handling in ToolRegistry")
    print("=" * 50)
    
    # Test basic OOM detection
    test_basic_oom_detection()
    
    # Test full OOM handling with threading
    test_oom_handling()
    
    print("\n" + "="*50)
    print("All tests completed!")
