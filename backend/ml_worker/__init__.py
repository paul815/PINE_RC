"""PINE ML worker — the transcription pipeline, isolated from Flask.

This package has NO dependency on the Flask app: no ``app.*`` imports, no
database access, no SocketIO. Everything it needs arrives in a job config;
everything it produces leaves through event callbacks (or, when run as a
subprocess via ``python -m ml_worker``, as JSON lines on stdout).

Layout:
    constants.py   — chunking thresholds, speaker labels, language thresholds
    errors.py      — TranscriptionCancelled
    audio.py       — ffmpeg/ffprobe helpers, WAV writer
    compat.py      — runtime shims (torch.load, hf_hub, torchaudio, torchcodec)
    engines/       — EngineAdapter interface + the WhisperX adapter (CUDA/Metal/CPU)
    diarize.py     — pyannote speaker diarization stage (shared by all engines)
    pipeline.py    — MLPipeline: probe → transcribe → diarize → clean/map
    protocol.py    — JSON-lines message helpers for the subprocess boundary
    __main__.py    — stdio server: ``python -m ml_worker``

The Flask side (``app/services/transcription``) owns the queue, the database,
SocketIO status events and the watchdog; it talks to this package through
``app/services/transcription/worker_client.py``.
"""

__version__ = '1.2.1'
