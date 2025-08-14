"""Question answering model using DistilBERT."""
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
from typing import List, Dict, Any
from tools.models.BaseModel import BaseModel
from agent.registry import register_tool, get_tool
from utils.decorator import time_it


class QuestionAnsweringModel(BaseModel):
    def __init__(self):
        self.taskname = "QuestionAnswering"
        self.name = "distilbert-base-cased-distilled-squad"

    @time_it(task_name="QuestionAnswering_Preload")
    def preload(self):
        """Preload model weights to CPU"""
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)

    @time_it(task_name="QuestionAnswering_Load")
    def load(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @time_it(task_name="QuestionAnswering_Predict")
    def predict(self, questions: List[str], contexts: List[str]) -> List[Dict[str, Any]]:
        """Answer questions based on context.

        Args:
            questions: List of questions
            contexts: List of context texts

        Returns:
            List of dictionaries containing answer text, start/end positions, and confidence score
        """
        results = []

        for question, context in zip(questions, contexts):
            # Tokenize inputs
            inputs = self.tokenizer(
                question,
                context,
                add_special_tokens=True,
                return_tensors="pt"
            ).to(self.device)

            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get answer start and end logits
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits

            # Get the most likely beginning and end of answer with the argmax of the score
            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores)

            # Convert tokens to string
            answer_tokens = inputs.input_ids[0, answer_start:answer_end+1]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

            # Calculate confidence score (mean of start/end scores)
            confidence = (answer_start_scores[0, answer_start].item() +
                          answer_end_scores[0, answer_end].item()) / 2.0

            results.append({
                "answer": answer,
                "start": answer_start.item(),
                "end": answer_end.item(),
                "confidence": confidence
            })

        return results

    def __del__(self):
        """Clear model and tokenizer to free memory."""
        self.discord()


def answer_question(question: str, context: str) -> Dict[str, Any]:
    """Answer a question based on a context.

    Args:
        question: The question to answer
        context: The context to extract the answer from

    Returns:
        Dictionary containing answer text and confidence score
    """
    # Get from registry or create and register if not exists
    model_instance = get_tool('question_answering')
    if model_instance is None:
        model_instance = QuestionAnsweringModel()
        register_tool('question_answering', model_instance)

    model_instance.preload()
    model_instance.load()
    answers = model_instance.predict([question], [context])
    model_instance.discord()
    return answers[0] if answers else {"answer": "", "confidence": 0.0}


if __name__ == "__main__":
    # Example usage
    question = "Who was Jim Henson?"
    context = "Jim Henson was a nice puppet master who created The Muppets."
    result = answer_question(question, context)
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Answer: {result['answer']} (confidence: {result['confidence']:.4f})")
