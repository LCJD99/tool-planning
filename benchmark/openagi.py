from .agi_utils import *
from datetime import datetime
import random
from agent.MulModelAgent import MulModelAgent
from utils.logger import setup_logger
import logging
from benchmark.general_dataset import GeneralDataset
from monitor import *
from typing import List
from utils.utils import generate_intervals
import threading
import time
from scheduler.SerialAliveScheduler import SerialAliveScheduler
from tools.model_map import MODEL_MAP
from agent.Tools import tools
from utils.utils import create_function_name_map
from queue import Queue
from agent.registry import tool_registry
import csv
import os

class OpenAGI():
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.task_descriptions = txt_loader(os.path.join(data_path, "task_description.txt"))

    def get_task_prompt_from_index(self, task_index: int) -> str:
        prompt = self.task_descriptions[task_index].strip()
        dataset = GeneralDataset(str(task_index), self.data_path)

        j = random.randint(0, self.batch_size - 1)
        batch = dataset[j]

        if batch['input']['text'] is not None:
            prompt = f"{prompt}, text: {batch['input']['text'][j]}"

        if batch['input']['image'] is not None:
            prompt = f"{prompt}, picture path: {batch['input']['image'][j]} "
            #if is low quality, you should use super resolution generate intermediate image path in /tmp"

        if task_index <= 14:
            prompt = f"{prompt}, onle return new picture path in " ",  easy for me to parse"

        if 105 <= task_index <= 106:
            prompt = f"{prompt}, generated picture path /tmp/img.jpg"
        # else:
            # prompt = f"{prompt}, give me last tool output only"

        return prompt

def create_agent_and_process(prompt: str, session_id: str, max_iterations: int, task_type: str) -> str:
    """
    Create a MulModelAgent instance and process the given prompt.

    Args:
        prompt: The input prompt to process
        session_id: Unique identifier for the session
        max_iterations: Maximum number of iterations for the agent

    Returns:
        The response from the agent
    """
    logging.info(f"Creating agent for session {session_id}")

    parts = task_type.split('_')
    scheduler_type = parts[0]

    # Create a new agent instance for this request
    agent = MulModelAgent(
        model="./qwen2.5",
        api_key="fake api",
        base_url="http://localhost:8000/v1",
        temperature=0.0,
        id = session_id,
        scheduler_type=scheduler_type
    )

    try:
        # Process the prompt
        response = agent.process(prompt, task_type, max_iterations, is_cot = False)
        logging.info(f"Session {session_id} completed successfully, response: {response}")
        return response
    except Exception as e:
        error_msg = f"Error processing session {session_id}: {str(e)}"
        logging.error(error_msg)
        return error_msg

def record_request_timing(session_id: str, type_name: str, workload: str, start_time: float, end_time: float) -> None:
    """
    Record request timing data to a CSV file (for record VRAM Usage)

    Args:
        session_id: Session identifier
        type_name: Type extracted from task_type (e.g., "serial", "parallel")
        workload: Workload extracted from task_type (e.g., "1", "2")
        start_time: Request start time
        end_time: Request end time
    """
    csv_path = "datas/request_timing.csv"
    file_exists = os.path.isfile(csv_path)
    
    # Convert timestamps to readable format
    start_time_formatted = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    end_time_formatted = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['session_id', 'type', 'workload', 'start_time', 'end_time'])

        writer.writerow([session_id, type_name, workload, start_time_formatted, end_time_formatted])


def process_request(prompt: str, session_id: str, max_iterations: int, result_queue: Queue, task_type: str) -> None:
    """
    Process a request and put the result in the queue.

    Args:
        prompt: The input prompt to process
        session_id: Unique identifier for the session
        max_iterations: Maximum number of iterations for the agent
        result_queue: Queue to store the result
        task_type: Task type in format "type_workload" (e.g., "serial_1")
    """
    start_time = time.time()
    
    # Parse task_type to extract type and workload
    parts = task_type.split('_')
    type_name = parts[0] if len(parts) > 0 else ""
    workload = parts[1] if len(parts) > 1 else ""
    
    result = create_agent_and_process(prompt, session_id, max_iterations, task_type)
    end_time = time.time()
    
    # Record timing data to CSV
    record_request_timing(session_id, type_name, workload, start_time, end_time)
    
    result_queue.put((session_id, result))

def seq_request(num_requests: int, prompts: List, task_type: str, batch_size: int = 1):
    """
    Execute requests in batches with the specified batch size.
    Each batch of requests executes in parallel, and the next batch starts only after
    all requests in the current batch have completed.

    Args:
        num_requests: Total number of requests to process
        prompts: List of prompts to use for requests
        batch_size: Number of concurrent requests per batch
    """
    result_queue = Queue()

    # Process requests in batches
    for batch_start in range(0, num_requests, batch_size):
        batch_end = min(batch_start + batch_size, num_requests)
        batch_threads = []
        batch_id = batch_start // batch_size + 1
        logging.info(f"Starting batch {batch_id}, requests {batch_start+1}-{batch_end}/{num_requests}")

        # Start all threads in this batch
        for i in range(batch_start, batch_end):
            logging.info(f"Starting request {i+1}/{num_requests}")

            # Create and start a thread for this request
            thread = threading.Thread(
                target=process_request,
                args=(prompts[i % len(prompts)], f"session_{i}", 20, result_queue, task_type)
            )
            thread.start()
            batch_threads.append((thread, i))

        # Wait for all threads in this batch to complete
        for thread, i in batch_threads:
            thread.join()

        # Process results from this batch
        for _ in range(batch_end - batch_start):
            session_id, response = result_queue.get()
            # logging.info(f"Result from {session_id}: {response[:100]}...")
        

def simulate_requests(num_requests: int, interval: float, prompts):
    threads = []
    result_queue = Queue()

    for i in range(num_requests):
        logging.info(f"Simulating request {i+1}/{num_requests}")
        start_time = time.time()

        # Create and start a thread for this request
        thread = threading.Thread(
            target=process_request,
            args=(prompts[i % len(prompts)], f"session_{i}", 20, result_queue)
        )
        thread.start()
        threads.append((thread, start_time, i))

        if i < num_requests - 1:
            # Wait between requests
            time.sleep(interval)

    # Wait for all threads to complete
    for thread, start_time, i in threads:
        thread.join()

    # Process results from the queue
    while not result_queue.empty():
        session_id, response = result_queue.get()
        logging.info(f"Result from {session_id}: {response[:100]}...")

def run_with_monitor():
    try:
        # cpu_monitor_thread, cpu_stop_event = start_cpu_monitoring()
        gpu_monitor_thread, gpu_stop_event = start_gpu_monitoring()
        # kv_monitor_thread, kv_stop_event = start_kvcache_monitoring()
        testcase()
    finally:
        # stop_cpu_monitoring(cpu_monitor_thread, cpu_stop_event)
        stop_gpu_monitoring(gpu_monitor_thread, gpu_stop_event)
        # stop_kvcache_monitoring(kv_monitor_thread, kv_stop_event)


def run_without_monitor():
    testcase()

def testcase():
    case = 1
    num_requests = 10
    openagi = OpenAGI(data_path="/home/zhangjingzhou/tool-planning/datasets/openagi/", eval_device="cuda", batch_size=1)
    task_list = [100, 101, 102, 111, 175]
    batch1 = [100, 101]
    batch2 = [111, 175, 102]
    bp1 = [openagi.get_task_prompt_from_index(i) for i in batch1]
    bp2 = [openagi.get_task_prompt_from_index(i) for i in batch2]
    prompts = bp1 + bp2

    # case 1 - batch execution with configurable batch_size
    if case == 1:
        # batch_sizes = [1, 2, 3, 4, 5]
        batch_sizes = [1, 3]
        execute_types = ['serial', 'parallel']
        # execute_types = ['parallel']
        for execute_type in execute_types:
            for batch_size in batch_sizes:
                logging.info(f"StageRecord: Running {execute_type}, {batch_size}")
                seq_request(num_requests * batch_size, prompts, f"{execute_type}_{batch_size}", batch_size=batch_size)
            # seq_request(num_requests * batch_size, prompts, f"serial_{batch_size}", batch_size=batch_size)
    elif case == 3:
        simulate_requests(num_requests, 3, prompts)
    else:
        # simulate_requests(len(bp1), 0.1,  bp1)
        # simulate_requests(len(bp2), 0.1,  bp2)
        simulate_requests(len(task_list), 0.1,  prompts)

if __name__ == '__main__':
    setup_logger(log_level = logging.INFO)
    tool_functions = [t.func for t in tools]
    function_map = create_function_name_map(tool_functions)
    scheduler = SerialAliveScheduler(MODEL_MAP, function_map)
    scheduler.manual_preload(['image_super_resolution', 'image_captioning', 'machine_translation', 'object_detection', 'image_classification', 'fill_mask', 'visual_question_answering', 'text_summarization'])
    run_with_monitor()
    # without_monitor()
    # test_datasets()
    logging.info(f"Tools: {tool_registry.get_counter_list()}")