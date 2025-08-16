import pynvml
import torch
import logging
import GPUtil

handle = None

def init_pynvml():
    global handle
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        logging.error(f"Failed to initialize pynvml: {e}")
        handle = None

def start_gpu_monitoring(interval=0.1, output_file='gpu_memory_record.csv', stop_event=None):
    """
    Start continuous GPU memory monitoring in a separate thread

    :param interval: Time interval between measurements in seconds
    :param output_file: File to write the monitoring data
    :param stop_event: Threading event to signal when to stop monitoring
    :return: The monitoring thread object
    """
    import threading
    import time
    import os
    from datetime import datetime

    if not torch.cuda.is_available():
        logging.warning("CUDA is not available. Continuous monitoring disabled.")
        return None

    global handle
    if handle is None:
        init_pynvml()

    if stop_event is None:
        stop_event = threading.Event()

    def monitoring_worker():
        with open(output_file, 'w') as f:
            f.write("timestamp,used_memory_mb,allocated_memory_mb,reserved_memory_mb,gpu_utilization_percent\n")

        start_time = time.time()
        logging.info(f"Starting GPU memory monitoring (interval: {interval}s)")

        while not stop_event.is_set():
            # Get memory usage
            if handle:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_mem = mem_info.used / 1024**2
                # Get GPU utilization
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization = utilization.gpu  # GPU utilization percentage
                except Exception as e:
                    logging.error(f"Failed to get GPU utilization: {e}")
                    gpu_utilization = 0
            else:
                used_mem = 0
                gpu_utilization = 0
                
                # Try using GPUtil as an alternative
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_utilization = gpus[0].load * 100  # Convert to percentage
                except Exception as e:
                    logging.error(f"Failed to get GPU utilization via GPUtil: {e}")

            allocated_mem = torch.cuda.memory_allocated(0) / 1024**2
            reserved_mem = torch.cuda.memory_reserved(0) / 1024**2

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            with open(output_file, 'a') as f:
                f.write(f"{timestamp},{used_mem:.2f},{allocated_mem:.2f},{reserved_mem:.2f},{gpu_utilization:.2f}\n")

            time.sleep(interval)

        logging.info("GPU memory monitoring stopped")

    monitor_thread = threading.Thread(target=monitoring_worker, daemon=True)
    monitor_thread.start()

    return monitor_thread, stop_event

def stop_gpu_monitoring(monitor_thread, stop_event):
    """
    Stop the continuous GPU memory monitoring

    :param monitor_thread: The monitoring thread to stop
    :param stop_event: The event to signal to stop monitoring
    """
    if monitor_thread and monitor_thread.is_alive():
        stop_event.set()
        monitor_thread.join(timeout=2.0)
        logging.info("GPU memory monitoring thread joined")

def cleanup():
    """
    Clean up NVML resources when monitoring is no longer needed
    """
    global handle
    if handle:
        try:
            pynvml.nvmlShutdown()
            logging.info("pynvml shut down.")
        except Exception as e:
            logging.error(f"Error shutting down pynvml: {e}")
    handle = None