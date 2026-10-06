from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from insightface.model_zoo import ArcFaceONNX
from insightface.utils.face_align import norm_crop
from numpy.typing import NDArray
from onnx.tools.update_model_dims import update_inputs_outputs_dims
from PIL import Image

from immich_ml.config import log, settings
from immich_ml.models.base import InferenceModel
from immich_ml.models.transforms import decode_cv2, serialize_np_array
from immich_ml.schemas import (
    FaceDetectionOutput,
    FacialRecognitionOutput,
    ModelFormat,
    ModelSession,
    ModelTask,
    ModelType,
)


class _CixArcFaceWrapper:
    """ArcFaceONNX-compatible wrapper for CIX NPU sessions.

    ArcFaceONNX.__init__ does onnx.load(model_file) to detect normalization
    parameters, which fails for .cix files. This wrapper provides the same
    get_feat() interface with hardcoded normalization for standard ArcFace models.
    """

    def __init__(self, session: ModelSession) -> None:
        self.session = session
        # Standard ArcFace normalization (non-mxnet models)
        self.input_mean = 127.5
        self.input_std = 127.5
        input_cfg = session.get_inputs()[0]
        self.input_size = tuple(input_cfg.shape[2:4][::-1])
        self.input_name = input_cfg.name
        self.output_names = [o.name for o in session.get_outputs()]

    def get_feat(self, imgs: list[NDArray[np.uint8]]) -> NDArray[np.float32]:
        if not isinstance(imgs, list):
            imgs = [imgs]
        blob = cv2.dnn.blobFromImages(
            imgs, 1.0 / self.input_std, self.input_size,
            (self.input_mean, self.input_mean, self.input_mean), swapRB=True,
        )
        net_out: NDArray[np.float32] = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out


class FaceRecognizer(InferenceModel):
    depends = [(ModelType.DETECTION, ModelTask.FACIAL_RECOGNITION)]
    identity = (ModelType.RECOGNITION, ModelTask.FACIAL_RECOGNITION)

    def __init__(self, model_name: str, **model_kwargs: Any) -> None:
        super().__init__(model_name, **model_kwargs)
        max_batch_size = settings.max_batch_size and settings.max_batch_size.facial_recognition
        self.batch_size = max_batch_size if max_batch_size else self._batch_size_default

    def _load(self) -> ModelSession:
        session = self._make_session(self.model_path)
        if self.model_format == ModelFormat.CIX:
            # ArcFaceONNX does onnx.load() which fails for .cix files
            self.model = _CixArcFaceWrapper(session)
        else:
            if (not self.batch_size or self.batch_size > 1) and str(session.get_inputs()[0].shape[0]) != "batch":
                self._add_batch_axis(self.model_path)
                session = self._make_session(self.model_path)
            self.model = ArcFaceONNX(
                self.model_path_for_format(ModelFormat.ONNX).as_posix(),
                session=session,
            )
        return session

    def _predict(
        self, inputs: NDArray[np.uint8] | bytes | Image.Image, faces: FaceDetectionOutput
    ) -> FacialRecognitionOutput:
        if faces["boxes"].shape[0] == 0:
            return []
        inputs = decode_cv2(inputs)
        cropped_faces = self._crop(inputs, faces)
        embeddings = self._predict_batch(cropped_faces)
        return self.postprocess(faces, embeddings)

    def _predict_batch(self, cropped_faces: list[NDArray[np.uint8]]) -> NDArray[np.float32]:
        if not self.batch_size or len(cropped_faces) <= self.batch_size:
            embeddings: NDArray[np.float32] = self.model.get_feat(cropped_faces)
            return embeddings

        batch_embeddings: list[NDArray[np.float32]] = []
        for i in range(0, len(cropped_faces), self.batch_size):
            batch_embeddings.append(self.model.get_feat(cropped_faces[i : i + self.batch_size]))
        return np.concatenate(batch_embeddings, axis=0)

    def postprocess(self, faces: FaceDetectionOutput, embeddings: NDArray[np.float32]) -> FacialRecognitionOutput:
        return [
            {
                "boundingBox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "embedding": serialize_np_array(embedding),
                "score": score,
            }
            for (x1, y1, x2, y2), embedding, score in zip(faces["boxes"], embeddings, faces["scores"])
        ]

    def _crop(self, image: NDArray[np.uint8], faces: FaceDetectionOutput) -> list[NDArray[np.uint8]]:
        return [norm_crop(image, landmark) for landmark in faces["landmarks"]]

    def _add_batch_axis(self, model_path: Path) -> None:
        log.debug(f"Adding batch axis to model {model_path}")
        proto = onnx.load(model_path)
        static_input_dims = [shape.dim_value for shape in proto.graph.input[0].type.tensor_type.shape.dim[1:]]
        static_output_dims = [shape.dim_value for shape in proto.graph.output[0].type.tensor_type.shape.dim[1:]]
        input_dims = {proto.graph.input[0].name: ["batch"] + static_input_dims}
        output_dims = {proto.graph.output[0].name: ["batch"] + static_output_dims}
        updated_proto = update_inputs_outputs_dims(proto, input_dims, output_dims)
        onnx.save(updated_proto, model_path)

    @property
    def _batch_size_default(self) -> int | None:
        providers = ort.get_available_providers()
        return None if self.model_format == ModelFormat.ONNX and "OpenVINOExecutionProvider" not in providers else 1
