"""Engine registry: pick and construct the right EngineAdapter for a model id."""

from .base import (  # noqa: F401 — re-exported interface
    EngineAdapter,
    EngineCapabilities,
    LanguageProbe,
    TranscribeContext,
    TranscribeOutput,
)

ENGINE_WHISPERX = 'whisperx'
ENGINE_MLX = 'mlx'
ENGINE_PARAKEET_ONNX = 'parakeet-onnx'


def engine_kind_for_model(stt_model_id: str) -> str:
    """Map an installed STT model id to the engine that runs it."""
    model_id = (stt_model_id or '').strip()
    if model_id.startswith('parakeet-'):
        return ENGINE_PARAKEET_ONNX
    if model_id.startswith('mlx-'):
        return ENGINE_MLX
    return ENGINE_WHISPERX


def select_engine_config(stt_model_id: str, prefer_mps: bool,
                         detected_device: str | None = None,
                         detected_compute: str | None = None):
    """Return (engine, device, compute_type) for the requested STT model.

    ``detected_device``/``detected_compute`` carry an earlier CUDA/CPU probe so
    the choice is stable across calls.
    """
    kind = engine_kind_for_model(stt_model_id)
    if kind == ENGINE_PARAKEET_ONNX:
        # onnxruntime picks its own provider and precision from what it finds
        # (see detect_onnx_device); the torch-derived probe does not apply, and
        # onnxruntime works on Metal where CTranslate2 does not.
        from .parakeet_onnx_engine import detect_onnx_device
        device, _providers, quantization = detect_onnx_device()
        return ENGINE_PARAKEET_ONNX, device, quantization or 'fp32'
    if kind == ENGINE_MLX:
        return ENGINE_MLX, 'cpu', None
    compute_type = 'int8'
    if prefer_mps:
        # WhisperX/CTranslate2 can't use MPS; CPU int8 is the best Mac fallback.
        return ENGINE_WHISPERX, 'cpu', compute_type
    return (ENGINE_WHISPERX,
            detected_device or 'cpu',
            detected_compute or compute_type)


def engine_capabilities(stt_model_id: str) -> EngineCapabilities:
    """Capabilities of the engine that would run this model (no ML imports)."""
    from .mlx_engine import MlxWhisperEngine
    from .whisperx_engine import WhisperXEngine
    kind = engine_kind_for_model(stt_model_id)
    if kind == ENGINE_PARAKEET_ONNX:
        from .parakeet_onnx_engine import ParakeetOnnxEngine
        return ParakeetOnnxEngine.capabilities
    if kind == ENGINE_MLX:
        return MlxWhisperEngine.capabilities
    return WhisperXEngine.capabilities


def create_engine(env, prefer_mps: bool = False) -> EngineAdapter:
    """Construct (but do not load) the adapter for ``env.stt_model_id``."""
    kind = engine_kind_for_model(env.stt_model_id)
    if kind == ENGINE_PARAKEET_ONNX:
        from .parakeet_onnx_engine import ParakeetOnnxEngine
        return ParakeetOnnxEngine(env)
    if kind == ENGINE_MLX:
        from .mlx_engine import MlxWhisperEngine
        return MlxWhisperEngine(env)
    from .whisperx_engine import WhisperXEngine, detect_torch_device
    engine = WhisperXEngine(env)
    if prefer_mps:
        engine._device = 'cpu'
        engine._compute_type = 'int8'
    else:
        engine._device, engine._compute_type = detect_torch_device()
    return engine
