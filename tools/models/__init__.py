from .ImageCaptioning import image_captioning
from .FillMask import fill_mask
from .ImageClassification import image_classification
from .ImageSuperResolution import image_super_resolution
from .MachineTranslation import machine_translation
from .ObjectDetection import object_detection
from .QuestionAnswering import question_answering
from .SentimentAnalysis import sentiment_analysis
from .TextSummarization import text_summarization
from .Text2ImageGeneration import text2image_genenration
from .VisualQuestionAnswering import visual_question_answering



from BaseModel import BaseModel
from FillMask import FillMask
from ImageCaptioning import ImageCaptioning
from ImageClassification import ImageClassification
from ImageSuperResolution import ImageSuperResolution
from MachineTranslation import MachineTranslation
from ObjectDetection import ObjectDetection
from QuestionAnswering import QuestionAnswering
from SentimentAnalysis import SentimentAnalysis
from TextSummarization import TextSummarization
from Text2ImageGeneration import Text2ImageGeneration
from VisualQuestionAnswering import VisualQuestionAnswering

MODEL_MAP = {
    "image_captioning": ImageCaptioning,
    "fill_mask": FillMask,
    "image_classification": ImageClassification,
    "image_super_resolution": ImageSuperResolution,
    "machine_translation": MachineTranslation,
    "object_detection": ObjectDetection,
    "question_answering": QuestionAnswering,
    "sentiment_analysis": SentimentAnalysis,
    "text_summarization": TextSummarization,
    "text2image_generation": Text2ImageGeneration,
    "visual_question_answering": VisualQuestionAnswering
}

all = [
    "image_captioning",
    "fill_mask",
    "image_classification",
    "image_super_resolution",
    "machine_translation",
    "object_detection",
    "question_answering",
    "sentiment_analysis",
    "text_summarization",
    "text2image_generation",
    "visual_question_answering",
    "BaseModel",
]

