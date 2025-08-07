"""Image super-resolution model using Swin2SR."""
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any
from tools.models.BaseModel import BaseModel
from agent.registry import register_tool, get_tool
from utils.decorator import time_it


class ImageSuperResolutionModel(BaseModel):
    def __init__(self):
        self.taskname = "ImageSuperResolution"
        self.name = "caidas/swin2SR-classical-sr-x2-64"

    @time_it(task_name="ImageSuperResolution_Preload")
    def preload(self):
        """Preload model weights to CPU"""
        self.model = Swin2SRForImageSuperResolution.from_pretrained(self.name)
        self.processor = AutoImageProcessor.from_pretrained(self.name)

    @time_it(task_name="ImageSuperResolution_Load")
    def load(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @time_it(task_name="ImageSuperResolution_Predict")
    def predict(self, image_paths: List[str]) -> List[np.ndarray]:
        """Enhance resolution of images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of enhanced image arrays
        """
        results = []
        for image_path in image_paths:
            image = Image.open(image_path)
            
            inputs = self.processor(image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.moveaxis(output, source=0, destination=-1)
            output = (output * 255.0).round().astype(np.uint8)
            
            results.append(output)
        
        return results

    def __del__(self):
        """Clear model and processor to free memory."""
        self.discord()
        

def image_super_resolution(image_path: str) -> np.ndarray:
    """Enhance resolution of an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Enhanced image array
    """
    # Get from registry or create and register if not exists
    model_instance = get_tool('image_super_resolution')
    if model_instance is None:
        model_instance = ImageSuperResolutionModel()
        register_tool('image_super_resolution', model_instance)
    
    model_instance.preload()
    model_instance.load()
    enhanced_images = model_instance.predict([image_path])
    
    # Convert to PIL Image and save or return as needed
    if enhanced_images:
        return enhanced_images[0]
    return None


if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/image.jpg"
    enhanced_image = image_super_resolution(image_path)
    if enhanced_image is not None:
        output_image = Image.fromarray(enhanced_image)
        output_image.save("enhanced_image.jpg")
        print(f"Enhanced image saved as enhanced_image.jpg")
