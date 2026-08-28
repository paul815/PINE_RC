"""EngineAdapter — the contract every STT engine implements.

An engine turns an audio file into segments in the **unified transcript
format**; everything else (diarization, speaker mapping, persistence, UI
events) is engine-agnostic and lives outside the adapters.

Unified segment format::

    {
        'start': float,          # seconds
        'end': float,
        'text': str,
        'speaker': str,          # filled by the diarization stage, '' before it
        'words': [               # present when capabilities.word_timestamps
            {'word': str, 'start': float, 'end': float, 'score': float},
        ],
    }
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EngineCapabilities:
    """What an engine can do — used to route pipeline stages and (later) UI hints."""
    word_timestamps: bool = False
    # 'external' — segments are compatible with the shared pyannote diarization
    # stage; 'native' — the engine labels speakers itself; 'none' — no speakers.
    diarization: str = 'external'
    streaming: bool = False
    multilingual: bool = True


@dataclass
class LanguageProbe:
    """Result of a first-30-seconds language identification pass."""
    code: str
    confidence: float
    second_confidence: float = 0.0
    options: list = field(default_factory=list)   # [{'code': str, 'probability': float}]
    uncertain: bool = False


@dataclass
class TranscribeOutput:
    segments: list                 # unified format (see module docstring)
    language: str                  # detected or forced language code
    duration_seconds: float        # total audio duration
    audio: object = None           # decoded waveform if already in RAM (reuse for diarization)


def _noop_status(**kwargs):
    return None


def _noop_check_cancel():
    return None


@dataclass
class TranscribeContext:
    """Callbacks threaded through long-running engine work.

    on_status(stage=..., message=..., percent=..., eta_secs=...) — progress for the UI.
    check_cancel() — raises TranscriptionCancelled when the job must stop.
    """
    on_status: callable = _noop_status
    check_cancel: callable = _noop_check_cancel


class EngineAdapter(ABC):
    """Base class for STT engines.

    Lifecycle: construct with a JobEnv (may be None in unit tests), call
    ``load()`` once (idempotent), then ``transcribe()`` per recording. Engines
    keep expensive state (models, align caches) between recordings.
    """

    id: str = ''
    capabilities: EngineCapabilities = EngineCapabilities()

    # True when the caller has already removed the silence and spliced the
    # speech together, joining turns with a fixed gap — what multitrack does
    # before handing over a track. An engine that runs its own VAD must key its
    # thresholds to that gap: those joins are the only boundaries left, and
    # missing one merges a turn from minute 3 into a turn from minute 40.
    presegmented: bool = False

    # The regions that were spliced together, as ``[(start, end, ...)]`` in the
    # handed-over audio's own clock. Set alongside ``presegmented`` when the
    # caller knows them. They are the exact boundaries, which is worth more than
    # any threshold: a VAD cannot tell a 0.2 s splice from a 0.2 s breath, so an
    # engine given these must cut on them and not guess.
    presegmented_regions: list | None = None

    def __init__(self, env=None):
        self.env = env

    @abstractmethod
    def load(self):
        """Load model weights. Idempotent; raises on unrecoverable setup errors."""

    def probe_language(self, audio_path, probe_audio):
        """Identify the language from the first ~30 s.

        ``probe_audio`` is a 16 kHz float32 mono numpy array; ``audio_path``
        is available for engines that need file input. Returns a
        LanguageProbe or None when probing is unsupported/failed (auto-detect).
        """
        return None

    @abstractmethod
    def transcribe(self, audio_path, total_duration, language, ctx: TranscribeContext) -> TranscribeOutput:
        """Transcribe the recording, chunking long audio internally."""
