"""Sentiment analysis model using FinBERT."""
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from typing import List, Dict, Any
from tools.models.BaseModel import BaseModel
from utils.decorator import time_it
import logging


class SentimentAnalysisModel(BaseModel):
    def __init__(self):
        self.taskname = "SentimentAnalysis"
        self.name = "yiyanghkust/finbert-tone"
        self.num_labels = 3
        self.id2label = {0: "neutral", 1: "positive", 2: "negative"}

    @time_it(task_name="SentimentAnalysis_Preload")
    def preload(self):
        """Preload model weights to CPU"""
        self.model = BertForSequenceClassification.from_pretrained(self.name, num_labels=self.num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(self.name)

    @time_it(task_name="SentimentAnalysis_Load")
    def load(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @time_it(task_name="SentimentAnalysis_Predict")
    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment of texts.

        Args:
            texts: List of text strings to analyze

        Returns:
            List of dictionaries containing sentiment label and score
        """
        results = []

        for text in texts:
            # Tokenize the text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Get the model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get predicted class and score
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)

            # Get highest probability class
            predicted_class = torch.argmax(probabilities, dim=1).item()
            score = probabilities[0, predicted_class].item()

            results.append({
                "label": self.id2label[predicted_class],
                "score": score
            })

        return results

    def __del__(self):
        """Clear model and tokenizer to free memory."""
        self.discord()


def sentiment_analysis(text: str) -> Dict[str, Any]:
    """Analyze sentiment of text.

    Args:
        text: Text to analyze

    Returns:
        Dictionary containing sentiment label and score
    """
    # Get from registry or create and register if not exists
    from agent.registry import register_tool, get_tool
    model_instance = get_tool('sentiment_analysis')
    if model_instance is None:
        logging.error("Sentiment analysis model not found in registry.")
        return {"label": "neutral", "score": 0.0}

    sentiments = model_instance.predict([text])
    return sentiments[0] if sentiments else {"label": "neutral", "score": 0.0}


if __name__ == "__main__":
    # Example usage
    texts = [
        "there is a shortage of capital, and we need extra financing",
        "growth is strong and we have plenty of liquidity",
        "there are doubts about our finances",
        "profits are flat"
    ]

    for text in texts:
        result = sentiment_analysis(text)
        print(f"Text: '{text}'")
        print(f"Sentiment: {result['label']} (score: {result['score']:.4f})")
        print()
