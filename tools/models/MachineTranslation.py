"""Machine translation model using T5."""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from typing import List
from tools.models.BaseModel import BaseModel
from agent.registry import register_tool, get_tool
from utils.decorator import time_it
import logging


class MachineTranslationModel(BaseModel):
    def __init__(self):
        self.taskname = "MachineTranslation"
        self.name = "google-t5/t5-base"
        self.max_length = 128

    @time_it(task_name="MachineTranslation_Preload")
    def preload(self):
        """Preload model weights to CPU"""
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)

    @time_it(task_name="MachineTranslation_Load")
    def load(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @time_it(task_name="MachineTranslation_Predict")
    def predict(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """Translate text between languages.

        Args:
            texts: List of texts to translate
            source_lang: Source language code (e.g., 'English')
            target_lang: Target language code (e.g., 'French')

        Returns:
            List of translated texts
        """
        translations = []

        for text in texts:
            # Format the input with T5's translation prefix
            input_text = f"translate {source_lang} to {target_lang}: {text}"
            input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.device)

            # Generate translation
            outputs = self.model.generate(
                **input_ids,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True
            )

            # Decode the generated tokens
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            translations.append(translation)

        return translations

    def __del__(self):
        """Clear model and tokenizer to free memory."""
        self.discord()


def machine_translation(text: str, source_lang: str = "English", target_lang: str = "French") -> str:
    """Translate text between languages.

    Args:
        text: Text to translate
        source_lang: Source language (default: "English")
        target_lang: Target language (default: "French")

    Returns:
        Translated text
    """
    # Get from registry or create and register if not exists
    model_instance = get_tool('machine_translation')
    if model_instance is None:
        logging.error("Machine translation model not found in registry.")
        return ""

    translations = model_instance.predict([text], source_lang, target_lang)
    return translations[0] if translations else ""


if __name__ == "__main__":
    # Example usage
    text = "The weather is nice today."
    translated_text = machine_translation(text, "English", "French")
    print(f"Original: {text}")
    print(f"Translation: {translated_text}")
