"""Text-to-image generation model using Stable Diffusion."""
#from diffusers import StableDiffusionPipeline
import torch
from typing import List, Optional
from PIL import Image
import os
from tools.models.BaseModel import BaseModel
from agent.registry import register_tool, get_tool
from utils.decorator import time_it


class Text2ImageGenerationModel(BaseModel):
    def __init__(self):
        self.taskname = "Text2ImageGeneration"
        self.name = "CompVis/stable-diffusion-v1-4"
        self.num_inference_steps = 50
        self.guidance_scale = 7.5
        self.height = 512
        self.width = 512

    @time_it(task_name="Text2ImageGeneration_Preload")
    def preload(self):
        """Preload model weights to CPU"""
        # For Stable Diffusion, we don't preload to CPU as it's inefficient
        # Just prepare the configuration
        pass

    @time_it(task_name="Text2ImageGeneration_Load")
    def load(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use float16 precision if on CUDA
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.name,
            torch_dtype=dtype,
            safety_checker=None  # For performance, can be re-enabled if needed
        )
        self.pipe = self.pipe.to(self.device)

    @time_it(task_name="Text2ImageGeneration_Predict")
    def predict(self, prompts: List[str], 
                num_inference_steps: Optional[int] = None,
                guidance_scale: Optional[float] = None,
                height: Optional[int] = None,
                width: Optional[int] = None) -> List[Image.Image]:
        """Generate images from text prompts.
        
        Args:
            prompts: List of text prompts
            num_inference_steps: Number of denoising steps (default: self.num_inference_steps)
            guidance_scale: Scale for classifier-free guidance (default: self.guidance_scale)
            height: Height of generated image (default: self.height)
            width: Width of generated image (default: self.width)
            
        Returns:
            List of generated PIL images
        """
        # Use default values if not specified
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        if height is None:
            height = self.height
        if width is None:
            width = self.width
            
        images = []
        
        for prompt in prompts:
            with torch.autocast(self.device.type):
                result = self.pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width
                )
                
                if hasattr(result, "nsfw_content_detected") and result.nsfw_content_detected:
                    # If NSFW content is detected, return a blank image
                    image = Image.new("RGB", (width, height), color=(255, 255, 255))
                else:
                    image = result.images[0]
                    
                images.append(image)
        
        return images

    def __del__(self):
        """Clear model to free memory."""
        self.discord()
        

def generate_image_from_text(prompt: str, 
                           num_inference_steps: int = 50,
                           guidance_scale: float = 7.5,
                           height: int = 512,
                           width: int = 512,
                           output_path: Optional[str] = None) -> Image.Image:
    """Generate image from text prompt.
    
    Args:
        prompt: Text description of the image to generate
        num_inference_steps: Number of denoising steps (default: 50)
        guidance_scale: Scale for classifier-free guidance (default: 7.5)
        height: Height of generated image (default: 512)
        width: Width of generated image (default: 512)
        output_path: Optional path to save the generated image
        
    Returns:
        Generated PIL image
    """
    # Get from registry or create and register if not exists
    model_instance = get_tool('text2image_generation')
    if model_instance is None:
        model_instance = Text2ImageGenerationModel()
        register_tool('text2image_generation', model_instance)
    
    model_instance.load()  # Direct load since we don't preload for Stable Diffusion
    images = model_instance.predict(
        [prompt], 
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width
    )
    
    image = images[0] if images else None
    
    # Save the image if output path is provided
    if image and output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
    
    return image


if __name__ == "__main__":
    # Example usage
    prompt = "a photo of an astronaut riding a horse on mars"
    output_path = "astronaut_rides_horse.png"
    image = generate_image_from_text(prompt, output_path=output_path)
    print(f"Generated image saved to {output_path}")
