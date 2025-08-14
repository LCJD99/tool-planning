import psutil
import logging
import time
import threading
import os
from datetime import datetime

"""
CPU and Physical Memory Monitoring Module

This module focuses on monitoring physical memory (RAM) usage rather than virtual memory.

Key Memory Concepts:
- Physical Memory (RAM): The actual hardware memory installed in the system
- Virtual Memory: A memory management technique that includes both RAM and swap space
- RSS (Resident Set Size): The portion of a process's memory that is held in physical RAM
- VMS (Virtual Memory Size): The total virtual memory used by a process

Note: psutil.virtual_memory() actually returns physical memory (RAM) information,
despite its confusing name. This is the industry standard naming convention.
"""

# Get current process object for monitoring
current_process = psutil.Process(os.getpid())

def start_cpu_monitoring(interval=0.1, output_file='cpu_memory_record.csv', stop_event=None):
    """
    Start continuous monitoring of current process's CPU and memory usage in a separate thread
    
    :param interval: Time interval between measurements in seconds
    :param output_file: File to write the monitoring data
    :param stop_event: Threading event to signal when to stop monitoring
    :return: The monitoring thread object and stop event
    """
    if stop_event is None:
        stop_event = threading.Event()
    
    def monitoring_worker():
        # Initialize the CSV file with headers
        with open(output_file, 'w') as f:
            f.write("timestamp,sys_physical_mem_total_mb,sys_physical_mem_used_mb,proc_physical_mem_mb\n")
            
        logging.info(f"Starting continuous process monitoring (interval: {interval}s)")
        
        while not stop_event.is_set():
            try:
                # Get process memory information
                proc_memory_info = current_process.memory_info()
                proc_physical_mem_mb = proc_memory_info.rss / (1024**2)  # Physical memory (RSS) in MB
                
                # Get global system physical memory information
                sys_physical_memory = psutil.virtual_memory()  # Note: virtual_memory() returns physical RAM info
                sys_physical_mem_total_mb = sys_physical_memory.total / (1024**2)
                sys_physical_mem_used_mb = sys_physical_memory.used / (1024**2)
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                # Write to file
                with open(output_file, 'a') as f:
                    f.write(f"{timestamp},{sys_physical_mem_total_mb:.2f},{sys_physical_mem_used_mb:.2f},{proc_physical_mem_mb:.2f}\n")
                
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                logging.error(f"Error monitoring process: {str(e)}")
            
            # Sleep for the specified interval
            time.sleep(interval)
        
        logging.info("Process monitoring stopped")
    
    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitoring_worker, daemon=True)
    monitor_thread.start()
    
    return monitor_thread, stop_event

def stop_cpu_monitoring(monitor_thread, stop_event):
    """
    Stop the continuous process monitoring
    
    :param monitor_thread: The monitoring thread to stop
    :param stop_event: The event to signal to stop monitoring
    """
    if monitor_thread and monitor_thread.is_alive():
        stop_event.set()
        monitor_thread.join(timeout=2.0)
        logging.info("Process monitoring thread joined")