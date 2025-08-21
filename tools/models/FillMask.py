"""Fill mask model using DistilRoBERTa."""
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from typing import List, Dict, Any
from tools.models.BaseModel import BaseModel
from utils.decorator import time_it
import logging


class FillMaskModel(BaseModel):
    def __init__(self):
        self.taskname = "FillMask"
        self.name = "distilroberta-base"
        self.top_k = 5

    @time_it(task_name="FillMask_Preload")
    def preload(self):
        """Preload model weights to CPU"""
        self.model = AutoModelForMaskedLM.from_pretrained(self.name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)

    @time_it(task_name="FillMask_Load")
    def load(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @time_it(task_name="FillMask_Predict")
    def predict(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Fill masked tokens in given texts.

        Args:
            texts: List of text strings containing <mask> token

        Returns:
            List of lists containing dictionaries with token and score for each masked position
        """
        results = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]

            if len(mask_token_index) == 0:
                results.append([{"token": "", "score": 0.0}])
                continue

            token_logits = self.model(**inputs).logits
            mask_token_logits = token_logits[0, mask_token_index, :]

            top_tokens = torch.topk(mask_token_logits, self.top_k, dim=1)

            text_predictions = []
            for i, (token_ids, scores) in enumerate(zip(top_tokens.indices, top_tokens.values)):
                text_predictions.append([
                    {
                        "token": self.tokenizer.decode(token_id.item()),
                        "score": score.item()
                    }
                    for token_id, score in zip(token_ids, scores)
                ])

            results.append(text_predictions[0])

        return results

    def __del__(self):
        """Clear model and tokenizer to free memory."""
        self.discord()


def fill_mask(text: str) -> List[Dict[str, Any]]:
    """Fill masked tokens in a given text.

    Args:
        text: Text containing <mask> token

    Returns:
        List of dictionaries with token and score for the masked position
    """
    # Get from registry or create and register if not exists
    from agent.registry import register_tool, get_tool
    model_instance = get_tool('fill_mask')
    if model_instance is None:
        logging.error("Fill mask model not found in registry.")
        return [{"token": "", "score": 0.0}]

    predictions = model_instance.predict([text])
    return predictions[0] if predictions else [{"token": "", "score": 0.0}]


if __name__ == "__main__":
    # Example usage
    text = "The man worked as a <mask>."
    predictions = fill_mask(text)
    print(f"Masked text: {text}")
    for pred in predictions:
        print(f"Token: {pred['token']}, Score: {pred['score']:.4f}")
