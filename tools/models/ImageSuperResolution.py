"""Image super-resolution model using Swin2SR."""
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any
from tools.models.BaseModel import BaseModel
from utils.decorator import time_it
import logging


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
    def predict(self, image_paths: List[str], output_paths: List[str]) -> List[str]:
        """Enhance resolution of images.

        Args:
            image_paths: List of image file paths
            output_paths: Optional list of output paths to save enhanced images.
                          Must have the same length as image_paths if provided.

        Returns:
            output_path: output path
        """
        results = []

        # Validate output_paths if provided
        if output_paths is not None and len(output_paths) != len(image_paths):
            raise ValueError("If output_paths is provided, it must have the same length as image_paths")

        for i, image_path in enumerate(image_paths):
            image = Image.open(image_path)

            inputs = self.processor(image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.moveaxis(output, source=0, destination=-1)
            output = (output * 255.0).round().astype(np.uint8)

            # Save the enhanced image if output path is provided
            output_image = Image.fromarray(output)
            output_image.save(output_paths[i])


        return output_paths

    def __del__(self):
        """Clear model and processor to free memory."""
        self.discord()


def image_super_resolution(image_path: str, output_path: str ) -> str:
    """Enhance resolution of an image.

    Args:
        image_path: Path to the image file
        output_path: Optional path to save the enhanced image directly

    Returns:
        Enhanced image path
    """
    # Get from registry or create and register if not exists
    from agent.registry import register_tool, get_tool
    model_instance = get_tool('image_super_resolution')
    if model_instance is None:
        model_instance = ImageSuperResolutionModel()
        logging.error("Image classification model not found in registry.")
        return "Model not found"


    # If output_path is provided, pass it to predict method
    output_paths = [output_path] if output_path is not None else None
    enhanced_images = model_instance.predict([image_path], output_paths)

    # Return the enhanced image array
    if enhanced_images:
        return enhanced_images[0]
    return None


if __name__ == "__main__":
    # Example usage 1: With output path - directly save the enhanced image
    image_path = "path/to/your/image.jpg"
    output_path = "path/to/output/enhanced_image.jpg"
    enhanced_image = image_super_resolution(image_path, output_path)
    print(f"Enhanced image saved as {output_path}")

    # Example usage 2: Without output path - get the enhanced image array
    image_path = "path/to/your/image.jpg"
    enhanced_image = image_super_resolution(image_path)
    if enhanced_image is not None:
        # Process the enhanced image as needed
        output_image = Image.fromarray(enhanced_image)
        output_image.save("enhanced_image_2.jpg")
        print(f"Enhanced image saved as enhanced_image_2.jpg")
