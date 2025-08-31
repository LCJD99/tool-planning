import requests
import re
import logging
import time
import threading
from datetime import datetime

def start_kvcache_monitoring(server_url="http://localhost:8000/metrics",
                             interval=0.2,
                             output_file='kvcache_usage_record.csv',
                             stop_event=None):
    """
    Start continuous monitoring of VLLM KV cache usage in a separate thread

    :param server_url: URL of the VLLM metrics endpoint
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
            f.write("timestamp,kvcache_usage_percent\n")

        logging.info(f"Starting continuous KV cache monitoring (interval: {interval}s)")

        # Regular expression to extract the cache usage percentage
        pattern = r'vllm:gpu_cache_usage_perc\{.*\}\s+(\d+\.\d+)'

        while not stop_event.is_set():
            try:
                # Send request to VLLM metrics endpoint
                response = requests.get(server_url)

                if response.status_code == 200:
                    # Extract lines containing gpu_cache_usage_perc
                    metrics_text = response.text
                    for line in metrics_text.split('\n'):
                        if 'gpu_cache_usage_perc' in line:
                            # Extract the cache usage percentage using regex
                            match = re.search(pattern, line)
                            if match:
                                kvcache_percent = float(match.group(1))

                                # Get current timestamp
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                                # Write to CSV file
                                with open(output_file, 'a') as f:
                                    f.write(f"{timestamp},{kvcache_percent:.6f}\n")

                                break
                else:
                    logging.error(f"Failed to get metrics: HTTP {response.status_code}")

            except Exception as e:
                logging.error(f"Error monitoring KV cache: {str(e)}")

            # Sleep for the specified interval
            time.sleep(interval)

        logging.info("KV cache monitoring stopped")

    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitoring_worker, daemon=True)
    monitor_thread.start()

    return monitor_thread, stop_event

def stop_kvcache_monitoring(monitor_thread, stop_event):
    """
    Stop the continuous KV cache monitoring

    :param monitor_thread: The monitoring thread to stop
    :param stop_event: The event to signal to stop monitoring
    """
    if monitor_thread and monitor_thread.is_alive():
        stop_event.set()
        monitor_thread.join(timeout=2.0)
        logging.info("KV cache monitoring thread joined")

