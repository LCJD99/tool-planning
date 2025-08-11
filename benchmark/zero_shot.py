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

_N = 1



def run_zero_gpt():
    """
    assign openagi data path
    """
    data_path = "/home/zhangjingzhou/tool-planning/datasets/"
    eval_device = "cuda"
    openai.api_key = "fake_openai"
    batch_size = 1
    # os.environ['TRANSFORMERS_CACHE'] = args.huggingface_cache

    print("Begin loading datasets...")
    task_discriptions = txt_loader(data_path+"task_description.txt")
    # task_idx = [0,21,61,105,110,120,10,35,62,107,115]
    # test_task_idx = [2,3,10,15,20,35,45,55,65,70,70,90,106,107]
    test_task_idx = [36]
    # test_dataloaders = []
    # for i in test_task_idx:
    #     dataset = GeneralDataset(i, data_path)
    #     dataloader = DataLoader(dataset, batch_size=batch_size)
    #     test_dataloaders.append(dataloader)

    test_tasks = [task_discriptions[i].strip() for i in test_task_idx]
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

        for j in range(_N):
            batch = dataset[j]
            if batch['input']['text'] is not None:
                prompt = f"{prompt}, text: {batch['input']['text'][j]}"
            if batch['input']['image'] is not None:
                prompt = f"{prompt}, picture path: {batch['input']['image'][j]}"
            response = agent.process(prompt, max_iterations=10)
            logging.info(f"Response: {response}")
            return

            inputs = list(batch['input'][0])

            if 0 <= task_index <= 14:
                outputs = list(batch['output'][0])
                dist = image_similarity(predictions, outputs, vit, vit_extractor)
                task_rewards.append(dist/100)
            elif 15<= task_index <=104 or 107<= task_index:
                outputs = list(batch['output'][0])
                f1 = np.mean(txt_eval(predictions, outputs, bertscore, device=eval_device))
                task_rewards.append(f1)
            else:
                score = clip_score(predictions, inputs)
                task_rewards.append(score.detach()/100)

        ave_task_reward = np.mean(task_rewards)


        seqCombination.close_module_seq()


        if 0 <=test_task_idx[i] <=14:
            similairies.append(ave_task_reward)
        elif 15<=test_task_idx[i]<=104 or 107<=test_task_idx[i]:
            berts.append(ave_task_reward)
        else:
            clips.append(ave_task_reward)

        rewards.append(ave_task_reward)


    print("Finished testing!")

    print("Evaluation results: ", np.mean(clips), np.mean(berts), np.mean(similairies), np.mean(rewards))

if __name__ == '__main__':
    setup_logger(log_level = logging.INFO)
    run_zero_gpt()
