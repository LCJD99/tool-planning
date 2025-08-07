import gc
import logging
import torch
import time
from utils.decorator import time_it

class BaseModel:
    def __init__(self):
        pass

    @time_it(task_name="BaseModel_Preload")
    def preload(self):
        """Preload model weights to CPU"""
        pass

    @time_it(task_name="BaseModel_Load")
    def load(self):
        """Load model weights to the device"""
        pass

    @time_it(task_name="BaseModel_Predict")
    def predict(self, *args, **kwargs):
        pass
    
    @time_it(task_name="Model_swap")
    def swap(self):
        """Swap model weights to CPU and clear GPU memory"""
        if self.device.type != "cuda":
            logging.DEBUG("Model is already on CPU, no need to swap")
            return
            
        # Move model to CPU
        self.model.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()
    
    @time_it(task_name="Model_clear")
    def discord(self):
        """Clear model and tokenizer to free memory"""
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()
        gc.collect()