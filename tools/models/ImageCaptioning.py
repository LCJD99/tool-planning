"""Image captioning model using ViT-GPT2."""
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import os
from typing import List
import time
import logging
import tools.models.BaseModel as BaseModel


class ImageCaptioningModel(BaseModel.BaseModel):
    def __init__(self):
        logging.info("ImageCaptioning_Model_Loading_Start")
        load_start_time = time.time()

        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.max_length = 16
        self.num_beams = 1
        self.gen_kwargs = {"max_length": self.max_length, "num_beams": self.num_beams}
        
        load_end_time = time.time()
        duration = load_end_time - load_start_time
        logging.info(f"ImageCaptioning_Model_Loading_Finish with {duration:3f}s" )

    def predict(self, image_paths: List[str]) -> List[str]:
        """Generate captions for given images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of generated captions
        """
        logging.info("ImageCaptioning_Prediction_Start")
        start_computing_time = time.time()
        
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

        end_computing_time = time.time()
        computing_duration = end_computing_time - start_computing_time
        
        logging.info(f"ImageCaptioning_Prediction_Finish with {computing_duration:.3f}s")
        
        # Swap model weights to CPU and clear GPU memory if FRONTEND_SWAP is enabled
        if os.getenv('FRONTEND_SWAP', 'false').lower() == 'true':
            self._swap_to_cpu_and_clear_gpu()
        
        return preds
        
# Global instance
_model_instance = None


def get_image_caption(image_path: str) -> str:
    """Generate caption for a single image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Generated caption as a string
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = ImageCaptioningModel()
    
    captions = _model_instance.predict([image_path])
    return captions[0] if captions else "No caption generated"


if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/image.jpg"
    caption = get_image_caption(image_path)
    print(f"Generated caption: {caption}")
