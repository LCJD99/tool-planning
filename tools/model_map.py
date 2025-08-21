from .models.FillMask import FillMaskModel
from .models.ImageCaptioning import ImageCaptioningModel
from .models.ImageClassification import ImageClassificationModel
from .models.ImageSuperResolution import ImageSuperResolutionModel
from .models.MachineTranslation import MachineTranslationModel
from .models.ObjectDetection import ObjectDetectionModel
from .models.QuestionAnswering import QuestionAnsweringModel
from .models.SentimentAnalysis import SentimentAnalysisModel
from .models.TextSummarization import TextSummarizationModel
from .models.Text2ImageGeneration import Text2ImageGenerationModel
from .models.VisualQuestionAnswering import VisualQuestionAnsweringModel

MODEL_MAP = {
    "image_captioning": ImageCaptioningModel,
    "fill_mask": FillMaskModel,
    "image_classification": ImageClassificationModel,
    "image_super_resolution": ImageSuperResolutionModel,
    "machine_translation": MachineTranslationModel,
    "object_detection": ObjectDetectionModel,
    "question_answering": QuestionAnsweringModel,
    "sentiment_analysis": SentimentAnalysisModel,
    "text_summarization": TextSummarizationModel,
    "text2image_generation": Text2ImageGenerationModel,
    "visual_question_answering": VisualQuestionAnsweringModel
}

