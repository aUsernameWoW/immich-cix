# CIX NPU Session for Immich ML
# Supports CIX P1 SoC (Orion O6) with Zhouyi NPU
from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from immich_ml.config import log, settings


# Check if CIX NPU is available
def _check_npu_available() -> bool:
    """Check if CIX NPU runtime (libnoe) is available."""
    try:
        from libnoe import NPU, noe_status_t

        npu = NPU()
        ret = npu.noe_init_context()
        # ret can be noe_status_t enum or int
        if ret == noe_status_t.NOE_STATUS_SUCCESS or ret == 0:
            npu.noe_deinit_context()
            return True
    except ImportError:
        log.debug("libnoe not found, CIX NPU not available")
    except Exception as e:
        log.debug(f"CIX NPU init failed: {e}")
    return False


is_available = _check_npu_available() and getattr(settings, "cix", True)
model_prefix = Path("cix") if is_available else None


# Input/output tensor mappings for each model type
# These match the CIX pre-quantized models from ai_model_hub
INPUT_OUTPUT_MAPPING: dict[str, dict[str, Any]] = {
    "clip_visual": {
        "input": {"input.1": (1, 3, 224, 224)},
        "output": {"output": (1, 512)},
    },
    "clip_textual": {
        "input": {"text": (1, 77)},
        "output": {"output": (1, 512)},
    },
    "detection": {
        "input": {"input.1": (1, 3, 640, 640)},
        "output": {
            "scores_8": (12800, 1),
            "scores_16": (3200, 1),
            "scores_32": (800, 1),
            "bboxes_8": (12800, 4),
            "bboxes_16": (3200, 4),
            "bboxes_32": (800, 4),
            "kps_8": (12800, 10),
            "kps_16": (3200, 10),
            "kps_32": (800, 10),
        },
    },
    "recognition": {
        "input": {"input.1": (1, 3, 112, 112)},
        "output": {"output": (1, 512)},
    },
    # OCR models (PP-OCRv4)
    "ocr_detection": {
        "input": {"x": (1, 3, 960, 608)},
        "output": {"sigmoid_0.tmp_0": (1, 1, 960, 608)},
    },
    "ocr_recognition": {
        "input": {"x": (1, 3, 32, 400)},
        "output": {"linear_1.tmp_1": (1, 100, 6625)},
    },
}


class CixNode(NamedTuple):
    """Represents a tensor node with name and shape."""

    name: str | None
    shape: tuple[int, ...]


def _numpy_dtype(data_type: Any) -> np.dtype[Any]:
    """Map a libnoe tensor data type to the matching numpy dtype."""
    from libnoe import noe_data_type_t as t

    dtypes = {
        t.NOE_DATA_TYPE_S8: np.int8,
        t.NOE_DATA_TYPE_U8: np.uint8,
        t.NOE_DATA_TYPE_S16: np.int16,
        t.NOE_DATA_TYPE_U16: np.uint16,
        t.NOE_DATA_TYPE_S32: np.int32,
        t.NOE_DATA_TYPE_U32: np.uint32,
        t.NOE_DATA_TYPE_S64: np.int64,
        t.NOE_DATA_TYPE_U64: np.uint64,
        t.NOE_DATA_TYPE_F16: np.float16,
        t.NOE_DATA_TYPE_F32: np.float32,
    }
    if data_type not in dtypes:
        raise ValueError(f"Unsupported CIX tensor data type: {data_type}")
    return np.dtype(dtypes[data_type])


class CixSession:
    """
    CIX NPU Session - wraps libnoe for Immich ML.

    This session provides the same interface as OrtSession/RknnSession,
    allowing seamless integration with Immich's model loading system.
    """

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model_type = self._detect_model_type(model_path)
        self._initialized = False
        # one NPU job per session: concurrent requests would overwrite each other's tensors
        self._lock = threading.Lock()

        log.info(f"Loading CIX NPU model from {model_path}")
        self._init_engine()
        log.info(f"Loaded CIX NPU model from {model_path} (type: {self.model_type})")

    def _detect_model_type(self, path: Path) -> str:
        """Detect model type from file path."""
        # Check for OCR models first (path pattern: ocr/*/detection/cix or ocr/*/recognition/cix)
        path_str = str(path).lower()
        if "/ocr/" in path_str:
            grandparent = path.parent.parent.name.lower()
            if grandparent == "detection":
                return "ocr_detection"
            elif grandparent == "recognition":
                return "ocr_recognition"

        # Check parent directory name first (cix/model.cix -> parent is "cix", grandparent is type)
        grandparent = path.parent.parent.name.lower()
        if grandparent == "visual":
            return "clip_visual"
        elif grandparent == "textual":
            return "clip_textual"
        elif grandparent == "detection":
            return "detection"
        elif grandparent == "recognition":
            return "recognition"

        # Check parent directly
        parent = path.parent.name.lower()
        if parent == "visual":
            return "clip_visual"
        elif parent == "textual":
            return "clip_textual"
        elif parent == "detection":
            return "detection"
        elif parent == "recognition":
            return "recognition"

        # Fallback: check filename
        name = path.stem.lower()
        if "visual" in name or "clip_visual" in name:
            return "clip_visual"
        elif "text" in name or "clip_txt" in name:
            return "clip_textual"
        elif "scrfd" in name or "det" in name:
            return "detection"
        elif "arcface" in name or "rec" in name:
            return "recognition"

        log.warning(f"Could not detect model type for {path}, defaulting to detection")
        return "detection"

    @staticmethod
    def _check_status(ret: Any, msg: str) -> None:
        """Check libnoe return value (enum or int) and raise on failure."""
        val = ret.value if hasattr(ret, "value") else int(ret)
        if val != 0:
            raise RuntimeError(f"{msg}: {ret}")

    @staticmethod
    def _unpack(result: Any) -> tuple[Any, Any]:
        """Unpack libnoe result — new API returns (status, data) tuple,
        old API returned {'ret': ..., 'data': ...} dict."""
        if isinstance(result, dict):
            return result["ret"], result["data"]
        elif isinstance(result, tuple):
            return result[0], result[1]
        else:
            return result, None

    def _init_engine(self) -> None:
        """Initialize the CIX NPU engine."""
        from libnoe import NOE_TENSOR_TYPE_INPUT, NOE_TENSOR_TYPE_OUTPUT, NPU

        self.npu = NPU()

        # Initialize context
        ret = self.npu.noe_init_context()
        self._check_status(ret, "Failed to initialize CIX NPU context")

        # Load graph
        status, graph_id = self._unpack(self.npu.noe_load_graph(str(self.model_path)))
        self._check_status(status, f"Failed to load CIX graph: {self.model_path}")
        self.graph_id = graph_id

        # Create job — new API needs noe_create_job_cfg_t(), old used {}
        try:
            from libnoe import noe_create_job_cfg_t

            job_cfg: Any = noe_create_job_cfg_t()
        except ImportError:
            job_cfg = {}
        status, job_id = self._unpack(self.npu.noe_create_job(self.graph_id, job_cfg))
        self._check_status(status, "Failed to create CIX job")
        self.job_id = job_id

        # Setup tensor descriptors
        self.input_descs = []
        self.output_descs = []

        # Get input tensors
        _, in_count = self._unpack(self.npu.noe_get_tensor_count(self.graph_id, NOE_TENSOR_TYPE_INPUT))
        for i in range(in_count):
            desc = self.npu.noe_get_tensor_descriptor(self.graph_id, NOE_TENSOR_TYPE_INPUT, i)
            self.input_descs.append(desc)
            log.debug(f"Input tensor {i}: scale={desc.scale}, zp={desc.zero_point}, size={desc.size}")

        # Get output tensors
        _, out_count = self._unpack(self.npu.noe_get_tensor_count(self.graph_id, NOE_TENSOR_TYPE_OUTPUT))
        for i in range(out_count):
            desc = self.npu.noe_get_tensor_descriptor(self.graph_id, NOE_TENSOR_TYPE_OUTPUT, i)
            self.output_descs.append(desc)
            log.debug(f"Output tensor {i}: scale={desc.scale}, zp={desc.zero_point}, size={desc.size}")

        self.input_dtypes = [_numpy_dtype(desc.data_type) for desc in self.input_descs]
        self.output_dtypes = [_numpy_dtype(desc.data_type) for desc in self.output_descs]

        self._initialized = True
        log.debug(f"CIX model initialized: {in_count} inputs, {out_count} outputs")

    @property
    def is_clip(self) -> bool:
        return self.model_type in ("clip_visual", "clip_textual")

    def _clip_shape(self, desc: Any, dtype: np.dtype[Any], is_input: bool) -> tuple[int, ...]:
        """CLIP models differ in context length and embedding size, so derive their shapes from the descriptors."""
        count = desc.size // dtype.itemsize
        if is_input and self.model_type == "clip_visual":
            side = math.isqrt(count // 3)
            return (1, 3, side, side)
        return (1, count)

    def get_inputs(self) -> list[CixNode]:
        """Get input tensor specifications."""
        mapping = INPUT_OUTPUT_MAPPING.get(self.model_type, {}).get("input", {})
        if self.is_clip:
            names = list(mapping) or [None]
            return [
                CixNode(name=names[0] if i == 0 else None, shape=self._clip_shape(desc, dtype, is_input=True))
                for i, (desc, dtype) in enumerate(zip(self.input_descs, self.input_dtypes))
            ]
        return [CixNode(name=k, shape=v) for k, v in mapping.items()]

    def get_outputs(self) -> list[CixNode]:
        """Get output tensor specifications."""
        mapping = INPUT_OUTPUT_MAPPING.get(self.model_type, {}).get("output", {})
        if self.is_clip:
            names = list(mapping) or [None]
            return [
                CixNode(name=names[0] if i == 0 else None, shape=self._clip_shape(desc, dtype, is_input=False))
                for i, (desc, dtype) in enumerate(zip(self.output_descs, self.output_dtypes))
            ]
        return [CixNode(name=k, shape=v) for k, v in mapping.items()]

    @staticmethod
    def _to_tensor(data: NDArray[Any], desc: Any, dtype: np.dtype[Any]) -> NDArray[Any]:
        """Convert an input array to the tensor's data type, quantizing float data for integer tensors."""
        if np.issubdtype(dtype, np.floating) or np.issubdtype(data.dtype, np.integer):
            # float tensors take the values as is, and integer data (e.g. token ids) isn't quantized
            return data.astype(dtype)
        info = np.iinfo(dtype)
        quantized = np.round(data.astype(np.float32) * desc.scale - desc.zero_point)
        return np.clip(quantized, info.min, info.max).astype(dtype)

    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, NDArray[np.float32]] | dict[str, NDArray[np.int32]],
        run_options: Any = None,
    ) -> list[NDArray[np.float32]]:
        """
        Run inference on the CIX NPU.

        Args:
            output_names: Optional list of output names (ignored, returns all outputs)
            input_feed: Dictionary mapping input names to numpy arrays
            run_options: Optional run options (ignored)

        Returns:
            List of output numpy arrays in float32 format
        """
        with self._lock:
            return self._run(input_feed)

    def _run(
        self, input_feed: dict[str, NDArray[np.float32]] | dict[str, NDArray[np.int32]]
    ) -> list[NDArray[np.float32]]:
        for i, (name, data) in enumerate(input_feed.items()):
            desc, dtype = self.input_descs[i], self.input_dtypes[i]
            q_data = self._to_tensor(data, desc, dtype)

            # Verify size matches
            expected_size = desc.size
            actual_size = len(q_data.tobytes())
            if actual_size != expected_size:
                raise RuntimeError(f"Input size mismatch for '{name}': expected {expected_size}, got {actual_size}")

            self.npu.noe_load_tensor(self.job_id, i, q_data.tobytes())

        # Run inference synchronously
        self.npu.noe_job_infer_sync(self.job_id, -1)

        # Get and dequantize outputs
        from libnoe import tensor_type_t

        outputs: list[NDArray[np.float32]] = []
        output_shapes = list(INPUT_OUTPUT_MAPPING.get(self.model_type, {}).get("output", {}).values())
        for i, (desc, dtype) in enumerate(zip(self.output_descs, self.output_dtypes)):
            # New API: noe_get_tensor(job_id, tensor_type, index) → (status, bytes)
            # Old API: noe_get_tensor(job_id, tensor_type, index, d_type) → {'ret': ..., 'data': ...}
            result = self.npu.noe_get_tensor(self.job_id, tensor_type_t.NOE_TENSOR_TYPE_OUTPUT, i)
            status, data = self._unpack(result)
            self._check_status(status, f"Failed to get output tensor {i}")

            # New API returns bytes, old API returned list
            if isinstance(data, (bytes, bytearray)):
                out_array = np.frombuffer(data, dtype=dtype)
            else:
                out_array = np.array(data, dtype=dtype)

            if np.issubdtype(dtype, np.floating):
                out_float = out_array.astype(np.float32)
            else:
                # Dequantize: x = (q + zero_point) / scale
                out_float = (out_array.astype(np.float32) + desc.zero_point) / desc.scale

            if self.is_clip:
                out_float = out_float.reshape(1, -1)
            elif i < len(output_shapes):
                try:
                    out_float = out_float.reshape(output_shapes[i])
                except ValueError:
                    pass  # Keep flat if reshape fails

            outputs.append(out_float)

        return outputs

    def set_providers(self, providers: list[str], **kwargs: Any) -> None:
        """No-op for compatibility with insightface's RetinaFace/ArcFaceONNX."""
        pass

    def release(self) -> None:
        """Release NPU resources."""
        if hasattr(self, "_initialized") and self._initialized:
            try:
                self.npu.noe_clean_job(self.job_id)
                self.npu.noe_unload_graph(self.graph_id)
                self.npu.noe_deinit_context()
                self._initialized = False
                log.debug("CIX NPU resources released")
            except Exception as e:
                log.warning(f"Error releasing CIX NPU resources: {e}")

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass  # Ignore cleanup errors during interpreter shutdown


__all__ = ["CixSession", "CixNode", "is_available", "model_prefix", "INPUT_OUTPUT_MAPPING"]
