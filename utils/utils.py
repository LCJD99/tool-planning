from typing import Callable, List, Dict, Any, Union
import torch
from PIL import Image
from torchvision import transforms

def create_function_name_map(functions: List[Callable[..., Any]]) -> Dict[str, Callable[..., Any]]:
    function_map = {}
    for func in functions:
        if callable(func):
            # 使用 inspect.getsourcefile(func) 可以检查函数是否是可调用的，
            # 这里我们也可以直接用 func.__name__ 获取函数名
            # 不过 inspect 模块提供了更健壮的方式来获取函数信息
            function_map[func.__name__] = func
    return function_map

def imgpath2vec(img_path: str) -> torch.Tensor:
    img = Image.open(img_path)
    img_transform = transforms.Compose([ transforms.Resize(256),
                                        transforms.CenterCrop(256),
                                        transforms.PILToTensor(),
                                        ])
    emb = img_transform(img)
    return emb




def load_image_for_colorization(image_paths: List[str]) -> List[torch.Tensor]:
    # 定义一个转换流程：将PIL图像转换为PyTorch张量
    # ToTensor() 会自动完成两件事：
    # 1. 将像素值从 [0, 255] 的整数范围，转换为 [0.0, 1.0] 的浮点数范围。
    # 2. 将图像维度从 (H, W, C) 转换为 PyTorch 需要的 (C, H, W)。
    transform = transforms.ToTensor()

    loaded_tensors = []
    for path in image_paths:
        try:
            # 使用Pillow库打开图像
            img = Image.open(path)

            # 确保图像是RGB格式（即使是灰度图也转为3通道），以保证通道数统一
            img_rgb = img.convert("RGB")

            # 应用转换，得到符合要求的张量
            tensor = transform(img_rgb)

            loaded_tensors.append(tensor)
        except FileNotFoundError:
            print(f"警告：文件未找到，已跳过 -> {path}")
        except Exception as e:
            print(f"警告：加载文件时出错 {path}，错误信息: {e}")

    return loaded_tensors


def generate_intervals(num_requests: int, rate: int) -> List[float]:
    import numpy as np
    import logging
    rate_per_second = rate / 60

    # For a Poisson process, the time between events follows an exponential distribution
    # with parameter lambda = rate
    intervals = np.random.exponential(scale=1/rate_per_second, size=num_requests)
    logging.debug(f"Generated {num_requests} intervals with mean: {np.mean(intervals):.2f}s")
    return intervals.tolist()


def record_timing(task_type: str, time0: float, time1: float, time2: float, time3: float, csv_path: str = None) -> None:
    """
    Record timing data to a CSV file for performance analysis.
    
    Args:
        task_type: The type of task being timed
        time0: Start time of first LLM call
        time1: End time of first LLM call
        time2: Start time of second LLM call
        time3: End time of second LLM call
        csv_path: Path to the CSV file (default is latency.csv in project root)
    """
    import os
    import csv
    import logging
    
    if csv_path is None:
        # Get the project root directory (assuming utils.py is in utils/ directory)
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(current_dir, 'latency.csv')
    
    # Calculate timing metrics
    llm_time1 = time1 - time0
    tools_time = time2 - time1
    llm_time2 = time3 - time2
    
    # Check if file exists and has header
    file_exists = os.path.isfile(csv_path)
    
    try:
        with open(csv_path, 'a', newline='') as csvfile:
            fieldnames = ['task_type', 'llm_time1', 'tools_time', 'llm_time2']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file doesn't exist
            if not file_exists:
                writer.writeheader()
            
            # Write the timing data
            writer.writerow({
                'task_type': task_type,
                'llm_time1': f"{llm_time1:.4f}",
                'tools_time': f"{tools_time:.4f}",
                'llm_time2': f"{llm_time2:.4f}"
            })
            
        logging.info(f"Timing data recorded to {csv_path}")
    except Exception as e:
        logging.error(f"Error recording timing data: {e}")
