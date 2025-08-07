"""Object detection model using DETR."""
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
from typing import List, Dict, Any
from tools.models.BaseModel import BaseModel
from agent.registry import register_tool, get_tool
from utils.decorator import time_it


class ObjectDetectionModel(BaseModel):
    def __init__(self):
        self.taskname = "ObjectDetection"
        self.name = "facebook/detr-resnet-101"
        self.revision = "no_timm"  # To avoid timm dependency
        self.threshold = 0.9

    @time_it(task_name="ObjectDetection_Preload")
    def preload(self):
        """Preload model weights to CPU"""
        self.model = DetrForObjectDetection.from_pretrained(self.name, revision=self.revision)
        self.processor = DetrImageProcessor.from_pretrained(self.name, revision=self.revision)

    @time_it(task_name="ObjectDetection_Load")
    def load(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @time_it(task_name="ObjectDetection_Predict")
    def predict(self, image_paths: List[str], threshold: float = None) -> List[List[Dict[str, Any]]]:
        """Detect objects in images.
        
        Args:
            image_paths: List of image file paths
            threshold: Confidence threshold for detections (default: self.threshold)
            
        Returns:
            List of lists containing object detections with label, score, and box coordinates
        """
        if threshold is None:
            threshold = self.threshold
            
        results = []
        for image_path in image_paths:
            image = Image.open(image_path)
            
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            
            # Post-process to get detections above threshold
            target_sizes = torch.tensor([image.size[::-1]])
            detections = self.processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes, 
                threshold=threshold
            )[0]
            
            image_detections = []
            for score, label, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
                image_detections.append({
                    "label": self.model.config.id2label[label.item()],
                    "score": score.item(),
                    "box": [round(i, 2) for i in box.tolist()]
                })
            
            results.append(image_detections)
        
        return results

    def __del__(self):
        """Clear model and processor to free memory."""
        self.discord()
        

def detect_objects(image_path: str, threshold: float = 0.9) -> List[Dict[str, Any]]:
    """Detect objects in an image.
    
    Args:
        image_path: Path to the image file
        threshold: Confidence threshold for detections (default: 0.9)
        
    Returns:
        List of dictionaries with detected objects (label, score, box)
    """
    # Get from registry or create and register if not exists
    model_instance = get_tool('object_detection')
    if model_instance is None:
        model_instance = ObjectDetectionModel()
        register_tool('object_detection', model_instance)
    
    model_instance.preload()
    model_instance.load()
    detections = model_instance.predict([image_path], threshold)
    return detections[0] if detections else []


if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/image.jpg"
    objects = detect_objects(image_path)
    print(f"Detected objects in {image_path}:")
    for obj in objects:
        print(f"Label: {obj['label']}, Score: {obj['score']:.3f}, Box: {obj['box']}")
