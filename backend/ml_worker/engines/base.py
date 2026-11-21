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
    """Result of a language identification pass over a short sample of audio."""
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

    def __init__(self, env=None):
        self.env = env

    @abstractmethod
    def load(self):
        """Load model weights. Idempotent; raises on unrecoverable setup errors."""

    def probe_window(self, audio_path, probe_secs, total_duration=0):
        """The audio ``probe_language`` should listen to, as a 16 kHz float32 array.

        The head of the file, which is right for any engine that gates its own
        audio: whisperx runs a VAD before the model and never offers it the
        ringback a phone call opens on. An engine without one sees whatever is
        there and should override this to find the speech first — 30 s of dial
        tone identifies a Russian interview as English at p=0.29, which is not an
        uncertain model but an unanswerable question, and the pipeline stops to
        put it to the user.
        """
        from ..audio import load_audio_range
        return load_audio_range(audio_path, 0, probe_secs)

    def probe_language(self, audio_path, probe_audio):
        """Identify the language from a short sample of audio.

        ``probe_audio`` is a 16 kHz float32 mono numpy array as returned by
        ``probe_window``; ``audio_path`` is available for engines that need file
        input. Returns a LanguageProbe or None when probing is
        unsupported/failed (auto-detect).
        """
        return None

    @abstractmethod
    def transcribe(self, audio_path, total_duration, language, ctx: TranscribeContext) -> TranscribeOutput:
        """Transcribe the recording, chunking long audio internally."""

    def release(self):
        """Hand memory back after a recording; the next job brings it back.

        Called once per job, after it ends however it ends. The default keeps
        everything resident; an engine that holds a GPU the rest of the machine
        shares (Metal) overrides it.
        """
        return None
