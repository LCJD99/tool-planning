"""Text summarization model using BART."""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from typing import List, Dict, Any, Optional
from tools.models.BaseModel import BaseModel
from tools.registry import register_tool, get_tool
from utils.decorator import time_it


class TextSummarizationModel(BaseModel):
    def __init__(self):
        self.taskname = "TextSummarization"
        self.name = "facebook/bart-large-cnn"
        self.max_length = 130
        self.min_length = 30
        self.do_sample = False

    @time_it(task_name="TextSummarization_Preload")
    def preload(self):
        """Preload model weights to CPU"""
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)

    @time_it(task_name="TextSummarization_Load")
    def load(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @time_it(task_name="TextSummarization_Predict")
    def predict(self, texts: List[str], 
                max_length: Optional[int] = None, 
                min_length: Optional[int] = None, 
                do_sample: Optional[bool] = None) -> List[str]:
        """Generate summaries for input texts.
        
        Args:
            texts: List of texts to summarize
            max_length: Maximum length of generated summary (default: self.max_length)
            min_length: Minimum length of generated summary (default: self.min_length)
            do_sample: Whether to use sampling (default: self.do_sample)
            
        Returns:
            List of generated summaries
        """
        # Use default values if not specified
        if max_length is None:
            max_length = self.max_length
        if min_length is None:
            min_length = self.min_length
        if do_sample is None:
            do_sample = self.do_sample
            
        summaries = []
        
        for text in texts:
            # Tokenize the text
            inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
            
            # Generate summary
            summary_ids = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode the generated summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        
        return summaries

    def __del__(self):
        """Clear model and tokenizer to free memory."""
        self.discord()
        

def summarize_text(text: str, 
                  max_length: int = 130, 
                  min_length: int = 30, 
                  do_sample: bool = False) -> str:
    """Generate summary for text.
    
    Args:
        text: Text to summarize
        max_length: Maximum length of generated summary (default: 130)
        min_length: Minimum length of generated summary (default: 30)
        do_sample: Whether to use sampling (default: False)
        
    Returns:
        Generated summary
    """
    # Get from registry or create and register if not exists
    model_instance = get_tool('text_summarization')
    if model_instance is None:
        model_instance = TextSummarizationModel()
        register_tool('text_summarization', model_instance)
    
    model_instance.preload()
    model_instance.load()
    summaries = model_instance.predict(
        [text], 
        max_length=max_length, 
        min_length=min_length, 
        do_sample=do_sample
    )
    return summaries[0] if summaries else ""


if __name__ == "__main__":
    # Example usage
    article = """New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
    A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
    Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
    In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
    Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
    2010 marriage license application, according to court documents.
    Prosecutors said the marriages were part of an immigration scam.
    """
    summary = summarize_text(article)
    print(f"Original article length: {len(article)} characters")
    print(f"Summary length: {len(summary)} characters")
    print(f"Summary: {summary}")
