"""Visual question answering model using GIT."""
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image
from typing import List, Dict, Any
from tools.models.BaseModel import BaseModel
from agent.registry import register_tool, get_tool
from utils.decorator import time_it


class VisualQuestionAnsweringModel(BaseModel):
    def __init__(self):
        self.taskname = "VisualQuestionAnswering"
        self.name = "microsoft/git-base"
        self.max_length = 50

    @time_it(task_name="VisualQuestionAnswering_Preload")
    def preload(self):
        """Preload model weights to CPU"""
        self.model = AutoModelForCausalLM.from_pretrained(self.name)
        self.processor = AutoProcessor.from_pretrained(self.name)

    @time_it(task_name="VisualQuestionAnswering_Load")
    def load(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @time_it(task_name="VisualQuestionAnswering_Predict")
    def predict(self, image_paths: List[str], questions: List[str]) -> List[str]:
        """Answer questions about images.

        Args:
            image_paths: List of image file paths
            questions: List of questions corresponding to each image

        Returns:
            List of answers to the questions
        """
        answers = []

        for image_path, question in zip(image_paths, questions):
            # Load the image
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert(mode="RGB")

            # Prepare inputs
            inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)

            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=5,
                    early_stopping=True
                )

            # Decode generated answer
            answer = self.processor.decode(outputs[0], skip_special_tokens=True)
            answers.append(answer)

        return answers

    def __del__(self):
        """Clear model and processor to free memory."""
        self.discord()


def answer_visual_question(image_path: str, question: str) -> str:
    """Answer a question about an image.

    Args:
        image_path: Path to the image file
        question: The question to answer about the image

    Returns:
        Answer to the question
    """
    # Get from registry or create and register if not exists
    model_instance = get_tool('visual_question_answering')
    if model_instance is None:
        model_instance = VisualQuestionAnsweringModel()
        register_tool('visual_question_answering', model_instance)

    model_instance.preload()
    model_instance.load()
    answers = model_instance.predict([image_path], [question])
    return answers[0] if answers else ""


if __name__ == "__main__":
    # Example usage
    image_path = "enhanced_image.jpg"
    question = "what are you doing?"
    answer = answer_visual_question(image_path, question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
