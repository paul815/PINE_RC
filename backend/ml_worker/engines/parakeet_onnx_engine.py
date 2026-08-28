"""Parakeet TDT via ONNX Runtime — the light STT engine.

Runs `nvidia/parakeet-tdt-0.6b-v3` through onnx-asr. Unlike WhisperX (torch +
CTranslate2) and mlx-whisper (Metal), this one needs only numpy and
onnxruntime, and the same adapter covers CUDA on Windows and CoreML on Mac.

Two things it does that the other engines do not:

* **No language probe.** parakeet-tdt-v3 identifies the language itself, per
  segment, across 25 European languages. ``probe_language`` returns None, so
  the pipeline skips the confirm-language dialog rather than asking about a
  guess this engine never makes.
* **VAD does the chunking.** The ASR handles a bounded window, so onnx-asr
  slices the recording on silence instead of on a fixed grid. There is no
  overlap to reconcile and no ownership zone — cuts land in silence.

Every constant below was measured, not guessed; see
Documentation/stt-benchmark/onnx-port.md.
"""

import logging
import os
import sys
import sysconfig
import time
import types
from pathlib import Path

from ..audio import fmt_elapsed, load_audio_file
from ..constants import VAD_COMPACT_GAP_SEC
from .base import (
    EngineAdapter,
    EngineCapabilities,
    TranscribeContext,
    TranscribeOutput,
)

log = logging.getLogger(__name__)

MODEL_NAME = 'nemo-parakeet-tdt-0.6b-v3'
VAD_NAME = 'silero'

# VAD boundaries. The sweep in onnx-port.md moved divergence from the reference
# NeMo run from 2.44% to 1.50% purely by cutting less often: fewer boundaries
# means fewer words split across one. 45 s is where the curve flattens — at 60 s
# the cap barely binds any more (91 segments against 93) and nothing improves.
# onnx-asr's own docs suggest 20-30 s, but that warning is about "most models",
# not this one, and following it cost a third of the divergence.
VAD_MAX_SPEECH_SEC = 45.0
VAD_MIN_SILENCE_MS = 500.0
VAD_SPEECH_PAD_MS = 200.0

# Chunk target for pre-spliced audio. Regions are packed up to this before a
# chunk is closed — never split, only grouped. Longer is better up to the
# model's window: parakeet-tdt-v3 picks the language per call, and on a
# half-second fragment that choice collapses, transcribing Russian as English
# ("сервисами для" came back as "Service in Dia"). Context is what prevents it.
PRESEGMENTED_CHUNK_SEC = 40.0

# Multitrack is a different problem and the same numbers get it exactly wrong.
# There the silence is already gone: prepare_tracks splices the speech together
# and marks each join with a VAD_COMPACT_GAP_SEC pause, so those joins are the
# only boundaries in the file. At the single-track defaults the 0.2 s gap sits
# below the 0.5 s silence threshold and is invisible, while 0.2 s of padding
# bridges it outright — the engine then reads a turn from minute 3 straight into
# one from minute 40, and remapping scatters the words across both. So both
# numbers are derived from the gap instead of chosen: cut on anything shorter
# than it, and never pad far enough to close one.
MULTITRACK_MIN_SILENCE_MS = VAD_COMPACT_GAP_SEC * 1000 / 2
MULTITRACK_SPEECH_PAD_MS = VAD_COMPACT_GAP_SEC * 1000 / 4

# int8 is a memory trade, and only on CPU, where it runs as fast as fp32 (101 s
# against 103 s on a 20-minute clip) for 1.5 GB less RAM. On CUDA the same
# quantisation is three times *slower* than fp32, so it is never the GPU choice.
CUDA_QUANTIZATION = None
CPU_QUANTIZATION = 'int8'

_PIECE = '▁'   # sentencepiece word-start marker


def enable_bundled_cuda_libraries():
    """Put pip-installed CUDA/cuDNN DLLs where Windows will find them.

    onnxruntime-gpu depends on cudnn64_9.dll and cublas64_12.dll but, unlike
    torch, does nothing to locate them: the nvidia-* wheels drop them under
    site-packages/nvidia/*/bin, which is on no search path. Worse, the provider
    loader ignores ``os.add_dll_directory`` — only PATH actually takes effect —
    and the failure surfaces as ``Cannot load symbol cudnnCreate`` with no clue
    that a directory is missing. Returns the package names it found.
    """
    if not sys.platform.startswith('win'):
        return []
    root = Path(sysconfig.get_paths()['purelib']) / 'nvidia'
    names, dirs = [], []
    for bin_dir in sorted(root.glob('*/bin')):
        if any(bin_dir.glob('*.dll')):
            try:
                os.add_dll_directory(str(bin_dir))
            except OSError:
                continue
            dirs.append(str(bin_dir))
            names.append(bin_dir.parent.name)
    if dirs:
        os.environ['PATH'] = os.pathsep.join(dirs + [os.environ.get('PATH', '')])
    return names


def detect_onnx_device():
    """Return (device, providers, quantization) for this machine."""
    try:
        import onnxruntime
        available = onnxruntime.get_available_providers()
    except ImportError:
        available = []

    if 'CUDAExecutionProvider' in available:
        return 'cuda', ['CUDAExecutionProvider', 'CPUExecutionProvider'], CUDA_QUANTIZATION
    if 'CoreMLExecutionProvider' in available:
        # Apple Silicon. Precision follows the CPU rule: CoreML falls back to
        # CPU for anything it cannot place, and this has not been measured on
        # real hardware — see the open risk noted in onnx-port.md.
        return 'coreml', ['CoreMLExecutionProvider', 'CPUExecutionProvider'], CPU_QUANTIZATION
    return 'cpu', ['CPUExecutionProvider'], CPU_QUANTIZATION


def absolute_token_times(stamps, seg_start, seg_end):
    """Move token timestamps onto the recording's clock.

    onnx-asr hands the ASR a VAD-cut waveform, so stamps come back relative to
    the segment. Detected rather than assumed: if the largest stamp already
    sits past the segment's own span, they are absolute already.
    """
    if not stamps:
        return []
    if max(stamps) <= (seg_end - seg_start) + 0.5:
        return [s + seg_start for s in stamps]
    return list(stamps)


def words_from_tokens(tokens, stamps, seg_end):
    """Group sentencepiece tokens into PINE's word dicts.

    A TDT decoder emits one timestamp per token, so a word's ``end`` is the
    next word's start. Speaker attribution rides on ``start``, which is exact;
    only the drawn width of a word in a waveform is approximate.
    """
    words = []
    # strict=False on purpose: the two lists come back from onnx-asr, and a TDT
    # decoder is meant to emit one stamp per token. If a build ever disagrees,
    # dropping the tail beats raising and losing a finished 40-minute job.
    for token, stamp in zip(tokens, stamps, strict=False):
        starts_word = token.startswith((_PIECE, ' '))
        text = token.replace(_PIECE, ' ')
        if starts_word or not words:
            words.append({'start': float(stamp), 'end': None, 'word': text,
                          'score': 1.0})
        else:
            words[-1]['word'] += text

    out = []
    for i, w in enumerate(words):
        w['word'] = w['word'].strip()
        if not w['word']:
            continue
        w['end'] = float(words[i + 1]['start']) if i + 1 < len(words) else float(seg_end)
        out.append(w)
    return out


class ParakeetOnnxEngine(EngineAdapter):
    id = 'parakeet-onnx'
    capabilities = EngineCapabilities(
        word_timestamps=True,
        diarization='external',
        streaming=False,
        multilingual=True,
    )

    def __init__(self, env=None):
        super().__init__(env)
        self._asr = None
        self._vad = None
        self._models = {}          # presegmented -> wired model
        self._device = None
        self._quantization = None

    def load(self):
        if self._asr is not None:
            return

        enable_bundled_cuda_libraries()
        import onnx_asr

        self._device, providers, self._quantization = detect_onnx_device()
        model_dir = self.env.model_dir
        # The VAD is a separate 2 MB repo downloaded alongside the weights. Its
        # path is derived rather than configured so an offline install never
        # reaches for the hub mid-job.
        vad_dir = os.path.join(model_dir, 'vad')

        started = time.monotonic()
        self._asr = onnx_asr.load_model(
            MODEL_NAME, model_dir,
            quantization=self._quantization, providers=providers)
        self._vad = onnx_asr.load_vad(
            VAD_NAME, vad_dir if os.path.isdir(vad_dir) else None,
            providers=['CPUExecutionProvider'])
        self._models = {}

        log.info('parakeet-onnx loaded in %.1fs — device=%s quantization=%s',
                 time.monotonic() - started, self._device,
                 self._quantization or 'fp32')

    def _chunks_from_regions(self, regions, total_sec):
        """Group spliced regions into chunk spans, never splitting one.

        ``regions`` are ``(start, end, ...)`` tuples; only the first two matter
        here. Cutting exactly where the caller already cut is the whole point —
        no boundary is invented, so no word is clipped by a guess.
        """
        spans, lo, hi = [], None, None
        for region in regions:
            start, end = float(region[0]), float(region[1])
            if lo is None:
                lo, hi = start, end
            elif end - lo <= PRESEGMENTED_CHUNK_SEC:
                hi = end
            else:
                spans.append((lo, hi))
                lo, hi = start, end
        if lo is not None:
            spans.append((lo, min(hi, total_sec)))
        return spans

    def _recognize_presegmented(self, audio, regions, ctx):
        """Transcribe pre-spliced audio chunk by chunk, on the caller's cuts."""
        plain = self._asr.with_timestamps()
        total_sec = len(audio) / 16000.0
        for lo, hi in self._chunks_from_regions(regions, total_sec):
            ctx.check_cancel()
            piece = audio[int(lo * 16000):int(hi * 16000)]
            if not len(piece):
                continue
            res = plain.recognize(piece, sample_rate=16000)
            # Chunk-local times; the caller's clock is restored by the offset.
            yield types.SimpleNamespace(
                start=lo, end=hi, text=res.text,
                timestamps=[t + lo for t in (res.timestamps or [])],
                tokens=res.tokens)

    def _wired(self):
        """The VAD-wrapped model for the current mode, built once per mode.

        Built here rather than in ``load()`` because the thresholds depend on
        whether the caller pre-spliced the audio, and one loaded engine serves
        both kinds of job.
        """
        key = bool(self.presegmented)
        model = self._models.get(key)
        if model is None:
            silence = MULTITRACK_MIN_SILENCE_MS if key else VAD_MIN_SILENCE_MS
            pad = MULTITRACK_SPEECH_PAD_MS if key else VAD_SPEECH_PAD_MS
            # Order matters: with_vad first, then with_timestamps. The reverse
            # silently yields segments carrying no timestamps at all.
            model = self._asr.with_vad(
                self._vad,
                max_speech_duration_s=VAD_MAX_SPEECH_SEC,
                min_silence_duration_ms=silence,
                speech_pad_ms=pad,
            ).with_timestamps()
            self._models[key] = model
            log.info('parakeet-onnx VAD: presegmented=%s silence=%.0fms pad=%.0fms',
                     key, silence, pad)
        return model

    def probe_language(self, audio_path, probe_audio):
        """No probe: the model identifies the language itself, per segment."""
        return None

    def transcribe(self, audio_path, total_duration, language,
                   ctx: TranscribeContext) -> TranscribeOutput:
        self.load()

        device_note = {'cuda': '', 'coreml': ' (Metal-accelerated)'}.get(
            self._device, ' (CPU)')
        ctx.on_status(stage='transcribing',
                      message=f'Transcribing audio...{device_note}')

        # onnx-asr reads audio files itself, but only PCM WAV — a PINE recording
        # is whatever the user imported (m4a, mp3, float-encoded WAV), and
        # handing it a path fails with "file does not start with RIFF id".
        # ffmpeg decodes anything, it is what the other engines already use, and
        # the array is handed back for the diarization stage to reuse.
        audio = load_audio_file(audio_path)
        if total_duration <= 0:
            total_duration = len(audio) / 16000.0

        segments = []
        started = time.monotonic()
        last_end = 0.0

        # Pre-spliced audio comes with its real boundaries attached, so it is
        # cut on those. Only a recording nobody has segmented yet needs a VAD to
        # guess where the pauses are.
        if self.presegmented and self.presegmented_regions:
            results = self._recognize_presegmented(
                audio, self.presegmented_regions, ctx)
        else:
            results = self._wired().recognize(audio, sample_rate=16000)

        for result in results:
            ctx.check_cancel()

            text = (result.text or '').strip()
            last_end = float(result.end)
            if not text:
                continue

            stamps = absolute_token_times(
                result.timestamps or [], result.start, result.end)
            words = words_from_tokens(result.tokens or [], stamps, result.end)
            # Keep `text` exactly the space-joined words: recording.html locates
            # each word in the segment with indexOf, and any other spacing makes
            # the highlight drift.
            segments.append({
                'start': round(float(result.start), 3),
                'end': round(float(result.end), 3),
                'text': ' '.join(w['word'] for w in words) if words else text,
                'speaker': '',
                'words': words,
            })

            if total_duration > 0:
                percent = min(99, round(last_end / total_duration * 100))
                elapsed = time.monotonic() - started
                # Progress is real here — VAD segments arrive in order — so the
                # estimate extrapolates from audio actually consumed.
                remaining = (elapsed / last_end * (total_duration - last_end)
                             if last_end > 0 else None)
                ctx.on_status(
                    stage='transcribing',
                    message=(f'Transcribing... {percent}%'
                             + (f' — ~{fmt_elapsed(remaining)} remaining'
                                if remaining else '')),
                    percent=percent,
                    eta_secs=round(remaining) if remaining else None,
                )

        return TranscribeOutput(
            segments=segments,
            # The model detects per segment and PINE stores one language per
            # recording; a forced language wins, otherwise this is unset and
            # the caller keeps whatever the project preset had.
            language=language or '',
            duration_seconds=total_duration,
            audio=audio,
        )
