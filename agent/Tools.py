from langchain_core.tools import tool
from tools.models import *

tools = [
    tool(image_captioning, description="Generate captions for images"),
    tool(fill_mask, description="Fill in the blanks in a sentence"),
    tool(image_classification, description="Classify images into categories"),
    tool(image_super_resolution, description="Enhance image resolution"),
    tool(translate_text, description="Translate text between languages"),
    tool(detect_objects, description="Detect objects in images"),
    tool(answer_question, description="Answer questions based on provided context"),
    tool(analyze_sentiment, description="Analyze sentiment of text"),
    tool(summarize_text, description="Summarize long texts"),
    tool(generate_image_from_text, description="Generate images from text descriptions"),
    tool(answer_visual_question, description="Answer questions about images")
]
