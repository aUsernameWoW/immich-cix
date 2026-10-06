from typing import Any

from immich_ml.models.base import InferenceModel
from immich_ml.models.clip.textual import MClipTextualEncoder, OpenClipTextualEncoder
from immich_ml.models.clip.visual import OpenClipVisualEncoder
from immich_ml.schemas import ModelSource, ModelTask, ModelType

from .constants import get_model_source
from .facial_recognition.detection import FaceDetector
from .facial_recognition.recognition import FaceRecognizer

# Lazy imports for OCR — rapidocr may not be installed
_TextDetector = None
_TextRecognizer = None


def _get_ocr_classes():
    global _TextDetector, _TextRecognizer
    if _TextDetector is None:
        from immich_ml.models.ocr.detection import TextDetector as _TD
        from immich_ml.models.ocr.recognition import TextRecognizer as _TR
        _TextDetector = _TD
        _TextRecognizer = _TR
    return _TextDetector, _TextRecognizer


def get_model_class(model_name: str, model_type: ModelType, model_task: ModelTask) -> type[InferenceModel]:
    """Resolve model class from name, type, and task.

    Uses equality checks instead of structural pattern matching for
    Python 3.11 StrEnum compatibility (match/case with StrEnum patterns
    doesn't match plain strings on Python <3.12).
    """
    source = get_model_source(model_name)

    if source in (ModelSource.OPENCLIP, ModelSource.MCLIP) and model_type == ModelType.VISUAL and model_task == ModelTask.SEARCH:
        return OpenClipVisualEncoder

    if source == ModelSource.OPENCLIP and model_type == ModelType.TEXTUAL and model_task == ModelTask.SEARCH:
        return OpenClipTextualEncoder

    if source == ModelSource.MCLIP and model_type == ModelType.TEXTUAL and model_task == ModelTask.SEARCH:
        return MClipTextualEncoder

    if source == ModelSource.INSIGHTFACE and model_type == ModelType.DETECTION and model_task == ModelTask.FACIAL_RECOGNITION:
        return FaceDetector

    if source == ModelSource.INSIGHTFACE and model_type == ModelType.RECOGNITION and model_task == ModelTask.FACIAL_RECOGNITION:
        return FaceRecognizer

    if source == ModelSource.PADDLE and model_type == ModelType.DETECTION and model_task == ModelTask.OCR:
        td, _ = _get_ocr_classes()
        return td

    if source == ModelSource.PADDLE and model_type == ModelType.RECOGNITION and model_task == ModelTask.OCR:
        _, tr = _get_ocr_classes()
        return tr

    raise ValueError(f"Unknown model combination: {source}, {model_type}, {model_task}")


def from_model_type(model_name: str, model_type: ModelType, model_task: ModelTask, **kwargs: Any) -> InferenceModel:
    return get_model_class(model_name, model_type, model_task)(model_name, **kwargs)


def get_model_deps(model_name: str, model_type: ModelType, model_task: ModelTask) -> list[tuple[ModelType, ModelTask]]:
    return get_model_class(model_name, model_type, model_task).depends
