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
