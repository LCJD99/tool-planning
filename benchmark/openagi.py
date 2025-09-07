from sentence_transformers import SentenceTransformer, util
import os
from torch.utils.data import DataLoader
import torch
from .agi_utils import *
from datetime import datetime
import torch
import openai
import numpy as np
from IPython.utils import io
import random
from tqdm import tqdm
from evaluate import load
from torchvision import transforms
from transformers import AutoModel, AutoFeatureExtractor
from torchmetrics.multimodal import CLIPScore
from agent.MulModelAgent import MulModelAgent
from logger.config import setup_logger
import logging
from benchmark.general_dataset import GeneralDataset
from monitor import *
from typing import List, Any
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


def record_e2e_time(batch_id: int, batch_size: int, start_time: float, end_time: float, task_ids: str) -> None:
    """
    Record end-to-end execution time of a batch to a CSV file.

    Args:
        batch_id: ID of the batch
        batch_size: Number of threads executing simultaneously in the batch
        start_time: Start time of batch execution
        end_time: End time of batch execution
        task_ids: Comma-separated task identifiers for the batch
    """
    csv_path = os.path.join(os.path.dirname(__file__), "e2e_time.csv")
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['batch_id', 'batch_size', 'start_time', 'end_time', 'execution_time', 'task_ids'])

        execution_time = end_time - start_time
        writer.writerow([batch_id, batch_size, start_time, end_time, execution_time, task_ids])

    logging.info(f"Recorded execution time for batch {batch_id}: {execution_time:.4f} seconds")


_N = 1

class OpenAGI():
    def __init__(self, data_path: str, task_set: List[Any], eval_device: str = "cuda", batch_size: int = 1):
        self.data_path = data_path
        self.eval_device = eval_device
        self.batch_size = batch_size
        self.task_descriptions = txt_loader(os.path.join(data_path, "task_description.txt"))
        self.test_task_idx = task_set

    def get_task_prompt_from_index(self, task_index: int) -> str:
        prompt = self.task_descriptions[task_index].strip()
        dataset = GeneralDataset(str(task_index), self.data_path)
        logging.info(f"=== start task {task_index} ===")

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

        logging.info(f"prompt: {prompt}")
        return prompt



def run_zero_gpt():
    """
    assign openagi data path
    """
    data_path = "/home/zhangjingzhou/tool-planning/datasets/openagi/"
    eval_device = "cuda"
    openai.api_key = "fake_openai"
    batch_size = 1
    # os.environ['TRANSFORMERS_CACHE'] = args.huggingface_cache

    print("Begin loading datasets...")
    task_descriptions = txt_loader(data_path+"task_description.txt")
    # task_idx = [0,21,61,105,110,120,10,35,62,107,115]
    # test_task_idx = [2,3,10,15,20,35,45,55,65,70,90,106,107]
    test_task_idx = [27]
    # test_dataloaders = []
    # for i in test_task_idx:
    #     dataset = GeneralDataset(i, data_path)
    #     dataloader = DataLoader(dataset, batch_size=batch_size)
    #     test_dataloaders.append(dataloader)

    test_tasks = [task_descriptions[i].strip() for i in test_task_idx]
    print("Finish loading datasets!")

    print("Begin loading evaluation metrics...")
    clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

    # Load a pre-trained Vision Transformer model and its feature extractor
    vit_ckpt = "nateraw/vit-base-beans"
    vit = AutoModel.from_pretrained(vit_ckpt)
    vit.eval()
    vit_extractor = AutoFeatureExtractor.from_pretrained(vit_ckpt)

    f = transforms.ToPILImage()
    bertscore = load("bertscore")

    sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
    print("Finish loading metrics.")

    # seqCombination = SeqCombine(args)

    rewards = []
    clips = []
    berts = []
    similairies = []


    task_len = len(test_tasks)
    print("Begin testing...")

    # Testing
    agent = MulModelAgent()

    for i in range(task_len):
        prompt = test_tasks[i]
        task_index = test_task_idx[i]
        task_rewards = []
        dataset = GeneralDataset(str(task_index), data_path)
        logging.info(f"=== start task {task_index} ===")

        for j in range(_N):
            batch = dataset[j]
            if batch['input']['text'] is not None:
                prompt = f"{prompt}, text: {batch['input']['text'][j]}"
            if batch['input']['image'] is not None:
                prompt = f"{prompt}, picture path: {batch['input']['image'][j]}, if is low quality, you should use super resolution generate intermediate image path: /tmp/xxx.jpg"
            if task_index <= 14:
                prompt = f"{prompt}, onle return new picture path in " ",  easy for me to parse"
            if 105 <= task_index <= 106:
                prompt = f"{prompt}, generated picture path /tmp/img.jpg"

            else:
                prompt = f"{prompt}, give me last tool output only"
            logging.info(f"prompt: {prompt}")
            prompt = f"{prompt}, if use translation, only use once at last"

            response = agent.process(prompt, max_iterations=20)
            logging.info(f"Response: {response}")

            # inputs = list(batch['input'][0])

            predictions = [response]
            if 0 <= task_index <= 14:
                path = text2picpath(response)
                output = batch['output']['image'][j]
                dist = image_similarity([path], [output], vit, vit_extractor)
                task_rewards.append(dist/100)
            elif 15<= task_index <=104 or 107<= task_index:
                output = batch['output']['text'][j]
                f1 = np.mean(txt_eval(predictions, [output], bertscore, device=eval_device))
                task_rewards.append(f1)
            else:
                img = Image.open('/tmp/img.png')
                vec = img2vec(img)
                score = clip_score(vec, batch['input']['text'][j])
                task_rewards.append(score.detach()/100)

        ave_task_reward = np.mean(task_rewards)

        if 0 <=test_task_idx[i] <=14:
            similairies.append(ave_task_reward)
        elif 15<=test_task_idx[i]<=104 or 107<=test_task_idx[i]:
            berts.append(ave_task_reward)
        else:
            clips.append(ave_task_reward)

        rewards.append(ave_task_reward)


    print("Finished testing!")

    logging.info(f"Evaluation results: clips: {np.mean(clips)}, berts: {np.mean(berts)}, image: {np.mean(similairies)}, all: {np.mean(rewards)}")

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
    Record request timing data to a CSV file.

    Args:
        session_id: Session identifier
        type_name: Type extracted from task_type (e.g., "serial", "parallel")
        workload: Workload extracted from task_type (e.g., "1", "2")
        start_time: Request start time
        end_time: Request end time
    """
    csv_path = os.path.join(os.path.dirname(__file__), "request_timing.csv")
    file_exists = os.path.isfile(csv_path)
    
    # Convert timestamps to readable format
    start_time_formatted = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    end_time_formatted = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['session_id', 'type', 'workload', 'start_time', 'end_time'])

        writer.writerow([session_id, type_name, workload, start_time_formatted, end_time_formatted])

    logging.info(f"Recorded timing for session {session_id}: type={type_name}, workload={workload}")


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
        batch_start_time = time.time()
        task_ids = []

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
            task_id = f"task_{prompts[i % len(prompts)].split(',')[0][:20]}"
            task_ids.append(task_id)
            batch_threads.append((thread, i))

        # Wait for all threads in this batch to complete
        for thread, i in batch_threads:
            thread.join()

        # Process results from this batch
        for _ in range(batch_end - batch_start):
            session_id, response = result_queue.get()
            logging.info(f"Result from {session_id}: {response[:100]}...")

        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time

        # Record execution time for the entire batch
        task_ids_str = ",".join(task_ids)
        record_e2e_time(batch_id, batch_size, batch_start_time, batch_end_time, task_ids_str)

        logging.info(f"Completed batch {batch_id} in {batch_duration:.2f} seconds")

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

def with_monitor():
    try:
        cpu_monitor_thread, cpu_stop_event = start_cpu_monitoring()
        gpu_monitor_thread, gpu_stop_event = start_gpu_monitoring()
        kv_monitor_thread, kv_stop_event = start_kvcache_monitoring()
        # run_zero_gpt()
        testcase()
    finally:
        stop_cpu_monitoring(cpu_monitor_thread, cpu_stop_event)
        stop_gpu_monitoring(gpu_monitor_thread, gpu_stop_event)
        stop_kvcache_monitoring(kv_monitor_thread, kv_stop_event)


def without_monitor():
    testcase()

def testcase():
    case = 1
    rate = 5
    num_requests = 10
    intervals = generate_intervals(rate, num_requests)
    openagi = OpenAGI(data_path="/home/zhangjingzhou/tool-planning/datasets/openagi/", task_set=[27], eval_device="cuda", batch_size=1)
    task_list = [100, 101, 102, 111, 175]
    batch1 = [100, 101]
    batch2 = [111, 175, 102]
    # task_list = [101]
    bp1 = [openagi.get_task_prompt_from_index(i) for i in batch1]
    bp2 = [openagi.get_task_prompt_from_index(i) for i in batch2]
    prompts = bp1 + bp2

    # case 1 - batch execution with configurable batch_size
    if case == 1:
        # Execute requests in batches of 3
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



def test_datasets():
    openagi = OpenAGI(data_path="/home/zhangjingzhou/tool-planning/datasets/openagi/", task_set=[27], eval_device="cuda", batch_size=1)
    task_list = [100, 101, 102, 111, 175]
    for task in task_list:
        print(f"task_{task}: {openagi.get_task_prompt_from_index(task)}")


if __name__ == '__main__':
    setup_logger(log_level = logging.INFO)
    tool_functions = [t.func for t in tools]
    function_map = create_function_name_map(tool_functions)
    scheduler = SerialAliveScheduler(MODEL_MAP, function_map)
    scheduler.manual_preload(['image_super_resolution', 'image_captioning', 'machine_translation', 'object_detection', 'image_classification', 'fill_mask', 'visual_question_answering', 'text_summarization'])
    with_monitor()
    # without_monitor()
    # test_datasets()
    logging.info(f"Tools: {tool_registry.get_counter_list()}")

