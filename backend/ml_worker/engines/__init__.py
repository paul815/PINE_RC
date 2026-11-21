"""Engine registry: pick and construct the right EngineAdapter for a model id."""

from .base import (  # noqa: F401 — re-exported interface
    EngineAdapter,
    EngineCapabilities,
    LanguageProbe,
    TranscribeContext,
    TranscribeOutput,
)

ENGINE_WHISPERX = 'whisperx'


def engine_kind_for_model(stt_model_id: str) -> str:
    """Map an installed STT model id to the engine that runs it.

    One engine on every platform: WhisperX. The Mac's retired
    ``mlx-whisper-*`` ids never get this far — ``normalize_stt_model_id()``
    maps them onto ``whisperx-large-v3`` before a job is built.
    """
    return ENGINE_WHISPERX


def select_engine_config(stt_model_id: str, prefer_mps: bool,
                         detected_device: str | None = None,
                         detected_compute: str | None = None,
                         metal: bool | None = None):
    """Return (engine, device, compute_type) for the requested STT model.

    ``detected_device``/``detected_compute`` carry an earlier CUDA/CPU probe so
    the choice is stable across calls. ``metal`` says whether CTranslate2 can
    use this Mac's GPU; None asks ``ct2_metal_available()``.
    """
    compute_type = 'int8'
    if prefer_mps:
        # Apple Silicon: CTranslate2 4.9+ runs on Metal (float16 only — see
        # whisperx_engine.detect_mac_device); without it, CPU int8.
        if metal is None:
            from .whisperx_engine import ct2_metal_available
            metal = ct2_metal_available()
        if metal:
            return ENGINE_WHISPERX, 'mps', 'float16'
        return ENGINE_WHISPERX, 'cpu', compute_type
    return (ENGINE_WHISPERX,
            detected_device or 'cpu',
            detected_compute or compute_type)


def engine_capabilities(stt_model_id: str) -> EngineCapabilities:
    """Capabilities of the engine that would run this model (no ML imports)."""
    from .whisperx_engine import WhisperXEngine
    return WhisperXEngine.capabilities


def create_engine(env, prefer_mps: bool = False) -> EngineAdapter:
    """Construct (but do not load) the adapter for ``env.stt_model_id``."""
    from .whisperx_engine import WhisperXEngine, detect_mac_device, detect_torch_device
    engine = WhisperXEngine(env)
    if prefer_mps:
        engine._device, engine._compute_type = detect_mac_device()
    else:
        engine._device, engine._compute_type = detect_torch_device()
    return engine
