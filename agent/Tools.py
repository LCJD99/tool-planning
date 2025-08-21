from langchain_core.tools import tool
from tools.models import *

tools = [
    tool(image_captioning, description="Generate captions for images"),
    tool(fill_mask, description="Fill in the blanks in a sentence"),
    tool(image_classification, description="Classify images into categories"),
    tool(image_super_resolution, description="Enhance image resolution"),
    tool(machine_translation, description="Translate text between languages"),
    tool(object_detection, description="Detect objects in images"),
    tool(question_answering, description="Answer questions based on provided context"),
    tool(sentiment_analysis, description="Analyze sentiment of text"),
    tool(text_summarization, description="Summarize long texts"),
    tool(text2image_generation, description="Generate images from text descriptions"),
    tool(visual_question_answering, description="Answer questions about images"),
    # tool(image_colorization, description="colorized the image")
]
