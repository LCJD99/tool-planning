import gc
import logging
import torch
import time

class BaseModel:
    def __init__(self):
        pass
    
    def _swap_to_cpu_and_clear_gpu(self):
        """Swap model weights to CPU and clear GPU memory"""
        if self.device.type != "cuda":
            logging.DEBUG("Model is already on CPU, no need to swap")
            return
            
        swap_start_time = time.time()
        logging.info("ImageCaptioning_Swap_To_CPU_Start")
        
        # Move model to CPU
        self.model.to("cpu")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        swap_end_time = time.time()
        swap_duration = swap_end_time - swap_start_time
        
        logging.info(f"ImageCaptioning_Swap_To_CPU_Finish with {swap_duration:.3f}s")