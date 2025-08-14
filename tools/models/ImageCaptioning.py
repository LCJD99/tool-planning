"""Image captioning model using ViT-GPT2."""
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
from typing import List
from tools.models.BaseModel import BaseModel
from agent.registry import register_tool, get_tool
from utils.decorator import time_it


class ImageCaptioningModel(BaseModel):
    def __init__(self):
        self.taskname = "ImageCaptioning"
        self.name = "nlpconnect/vit-gpt2-image-captioning"
        self.max_length = 16
        self.num_beams = 1
        self.gen_kwargs = {"max_length": self.max_length, "num_beams": self.num_beams}

    @time_it(task_name="ImageCaptioning_Preload")
    def preload(self):
        """Preload model weights to CPU"""
        self.model = VisionEncoderDecoderModel.from_pretrained(self.name)
        self.feature_extractor = ViTImageProcessor.from_pretrained(self.name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)


    @time_it(task_name="ImageCaptioning_Load")
    def load(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    @time_it(task_name="ImageCaptioning_Predict")
    def predict(self, image_paths: List[str]) -> List[str]:
        """Generate captions for given images.

        Args:
            image_paths: List of image file paths

        Returns:
            List of generated captions
        """
        images = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")
            images.append(i_image)

        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(pixel_values, **self.gen_kwargs)
        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]

        return preds

    def __del__(self):
        """Clear model and tokenizer to free memory."""
        self.discord()


def image_captioning(image_path: str) -> str:
    """Generate caption for a single image.

    Args:
        image_path: Path to the image file

    Returns:
        Generated caption as a string
    """
    # Get from registry or create and register if not exists
    model_instance = get_tool('image_captioning')
    if model_instance is None:
        model_instance = ImageCaptioningModel()
        register_tool('image_captioning', model_instance)

    model_instance.preload()
    model_instance.load()
    captions = model_instance.predict([image_path])
    model_instance.discord()
    return captions[0] if captions else "No caption generated"


if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/image.jpg"
    caption = image_captioning(image_path)
    print(f"Generated caption: {caption}")
