from sentence_transformers import SentenceTransformer, util
import os
from torch.utils.data import DataLoader
import torch
from .agi_utils import *
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
            prompt = f"{prompt}, picture path: {batch['input']['image'][j]} if is low quality, you should use super resolution generate intermediate image path: /tmp/xxx.jpg"

        if task_index <= 14:
            prompt = f"{prompt}, onle return new picture path in " ",  easy for me to parse"

        if 105 <= task_index <= 106:
            prompt = f"{prompt}, generated picture path /tmp/img.jpg"
        else:
            prompt = f"{prompt}, give me last tool output only"

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

def create_agent_and_process(prompt: str, session_id: str, max_iterations: int) -> str:
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

    # Create a new agent instance for this request
    agent = MulModelAgent(
        model="./qwen2.5",
        api_key="fake api",
        base_url="http://localhost:8000/v1",
        temperature=0.0,
    )

    try:
        # Process the prompt
        response = agent.process(prompt, max_iterations)
        logging.info(f"Session {session_id} completed successfully, response: {response}")
        return response
    except Exception as e:
        error_msg = f"Error processing session {session_id}: {str(e)}"
        logging.error(error_msg)
        return error_msg

def process_request(prompt: str, session_id: str, max_iterations: int, result_queue: Queue) -> None:
    """
    Process a request and put the result in the queue.

    Args:
        prompt: The input prompt to process
        session_id: Unique identifier for the session
        max_iterations: Maximum number of iterations for the agent
        result_queue: Queue to store the result
    """
    result = create_agent_and_process(prompt, session_id, max_iterations)
    result_queue.put((session_id, result))

def simulate_requests(num_requests: int, rate: float):
    intervals = generate_intervals(rate, num_requests)
    openagi = OpenAGI(data_path="/home/zhangjingzhou/tool-planning/datasets/openagi/", task_set=[27], eval_device="cuda", batch_size=1)
    prompts = [openagi.get_task_prompt_from_index(27)]
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
            time.sleep(3)

    # Wait for all threads to complete
    for thread, start_time, i in threads:
        thread.join()
        end_time = time.time()
        logging.info(f"Time taken for request {i+1}: {end_time - start_time:.2f} seconds")

    # Process results from the queue
    while not result_queue.empty():
        session_id, response = result_queue.get()
        logging.info(f"Result from {session_id}: {response[:100]}...")

def with_monitor():
    try:
        cpu_monitor_thread, cpu_stop_event = start_cpu_monitoring()
        gpu_monitor_thread, gpu_stop_event = start_gpu_monitoring()
        run_zero_gpt()
    finally:
        stop_cpu_monitoring(cpu_monitor_thread, cpu_stop_event)
        stop_gpu_monitoring(gpu_monitor_thread, gpu_stop_event)


def without_monitor():
    run_zero_gpt()

def thread_run():
    tool_functions = [t.func for t in tools]
    function_map = create_function_name_map(tool_functions)
    scheduler = SerialAliveScheduler(MODEL_MAP, function_map)
    scheduler.manual_preload(['image_super_resolution', 'image_captioning', 'machine_translation', 'image_classification'])
    simulate_requests(2, 5)

if __name__ == '__main__':
    setup_logger(log_level = logging.INFO)
    thread_run()

