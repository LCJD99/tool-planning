"""Image classification model using ViT."""
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from PIL import Image
from typing import List, Dict, Any
from tools.models.BaseModel import BaseModel
from agent.registry import register_tool, get_tool
from utils.decorator import time_it


class ImageClassificationModel(BaseModel):
    def __init__(self):
        self.taskname = "ImageClassification"
        self.name = "google/vit-base-patch16-224"
        self.top_k = 5

    @time_it(task_name="ImageClassification_Preload")
    def preload(self):
        """Preload model weights to CPU"""
        self.model = ViTForImageClassification.from_pretrained(self.name)
        self.processor = ViTImageProcessor.from_pretrained(self.name)

    @time_it(task_name="ImageClassification_Load")
    def load(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @time_it(task_name="ImageClassification_Predict")
    def predict(self, image_paths: List[str]) -> List[List[Dict[str, Any]]]:
        """Classify images into predefined categories.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of lists containing top-k class predictions with label and score
        """
        results = []
        for image_path in image_paths:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert(mode="RGB")
                
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Get top-k predictions
            top_k_values, top_k_indices = torch.topk(logits, self.top_k, dim=1)
            
            predictions = []
            for values, indices in zip(top_k_values, top_k_indices):
                for value, idx in zip(values, indices):
                    predictions.append({
                        "label": self.model.config.id2label[idx.item()],
                        "score": value.item()
                    })
            
            results.append(predictions)
        
        return results

    def __del__(self):
        """Clear model and processor to free memory."""
        self.discord()
        

def image_classification(image_path: str) -> List[Dict[str, Any]]:
    """Classify an image into predefined categories.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of dictionaries with label and score for top predictions
    """
    # Get from registry or create and register if not exists
    model_instance = get_tool('image_classification')
    if model_instance is None:
        model_instance = ImageClassificationModel()
        register_tool('image_classification', model_instance)
    
    model_instance.preload()
    model_instance.load()
    classifications = model_instance.predict([image_path])
    return classifications[0] if classifications else []


if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/image.jpg"
    results = image_classification(image_path)
    print(f"Image classifications:")
    for result in results:
        print(f"Label: {result['label']}, Score: {result['score']:.4f}")
