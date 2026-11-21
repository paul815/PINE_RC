"""WhisperX engine: faster-whisper decode + wav2vec2 forced alignment.

Runs on CUDA (float16, int8 below 6 GB VRAM), on Apple Silicon's GPU through
CTranslate2's Metal backend (float16, CTranslate2 4.9+), or on the CPU (int8).
The Mac uses the same engine and model as Windows: measured 2026-11-21 on an
88-min Russian interview (M4 Pro, 24 GB), it matched the Windows transcript
in 99.9% of words in ~1000s for the whole pipeline (mac-bench).
"""

import contextlib
import gc
import logging
import os
import subprocess
import sys
import threading
import time

from .. import compat
from ..audio import fmt_elapsed, load_audio_range
from ..constants import (
    ALIGN_MODEL_CANDIDATES,
    CHUNK_OVERLAP_SEC,
    CHUNK_SIZE_SEC,
    CHUNK_THRESHOLD_SEC,
    LANG_CONFIDENCE_MIN,
)
from ..punctuation import accept_reread, is_unpunctuated, prompt_before, tail_text
from .base import (
    EngineAdapter,
    EngineCapabilities,
    LanguageProbe,
    TranscribeContext,
    TranscribeOutput,
)

log = logging.getLogger(__name__)

_SAMPLE_RATE = 16000   # whisperx.load_audio and load_audio_range both decode at 16 kHz


# Phase timers, kept on purpose (decided 2026-08-21) as the app's only source of
# per-machine timing data. Whisper's own progress output is the only thing this
# path prints, so everything before its first batch (decode, VAD, language
# detect) and everything after its last (alignment) is a blind spot — these
# timers name each phase inside it. They are what showed the second unit of work
# in a job costing far more per second of audio than the first. Cost is one
# log.info per phase per chunk, written to backend/logs/<date>/app-*.log and
# never surfaced in the UI; if it ever gets noisy, drop these to DEBUG rather
# than removing them.
@contextlib.contextmanager
def _perf(label):
    t0 = time.monotonic()
    try:
        yield
    finally:
        log.info('PERF: %s took %.1fs', label, time.monotonic() - t0)


# CTranslate2's caching allocator keeps growing until the card is full. Measured
# 2026-08-26 on a 12 GB card: during decode the whisper model held everything —
# nvidia-smi read 12281/12282 MiB used, 0 MiB free, 100% utilisation at 58 W of
# 220 — and every torch stage around it (VAD, alignment, pyannote) then ran out
# of host memory through WDDM paging instead of VRAM. The same 498s chunk took
# 7.1s in one job and 841.7s in the next with no setting changed. Windows pages
# rather than raising OOM, so the retry below never sees it; bounding the cache
# is what leaves the torch stages a working set. CT2 still allocates whatever a
# batch genuinely needs — this caps only what it holds on to between batches.
CT2_ALLOCATOR_CONFIG = '4,3,10,1073741824'  # bin_growth,min_bin,max_bin,max_cached_bytes


def cap_ct2_allocator():
    """Bound the CTranslate2 CUDA cache unless the machine already set a policy.

    Read by CT2 when it creates the allocator, i.e. on the first model load, so
    this has to run before whisperx.load_model().
    """
    existing = os.environ.get('CT2_CUDA_CACHING_ALLOCATOR_CONFIG')
    if existing:
        log.info('CT2 caching allocator left as configured: %s', existing)
        return
    os.environ['CT2_CUDA_CACHING_ALLOCATOR_CONFIG'] = CT2_ALLOCATOR_CONFIG
    log.info('CT2 caching allocator capped at %s', CT2_ALLOCATOR_CONFIG)


def _is_out_of_memory(exc, device):
    """Whether ``exc`` is the GPU running out of memory.

    'out of memory' is the test this path has always used on CUDA, and also
    what torch says on Metal ('MPS backend out of memory'). What CTranslate2's
    Metal backend raises on a full GPU was never seen in testing, so on Metal
    the usual allocation-failure wordings count too.
    """
    msg = str(exc).lower()
    if 'out of memory' in msg:
        return True
    return device == 'mps' and any(
        k in msg for k in ('insufficient memory', 'failed to allocate', 'could not allocate'))


def free_vram_gb():
    """Free VRAM in GB, or None off CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.mem_get_info()[0] / (1024 ** 3)
    except Exception:      # noqa: BLE001 — diagnostics must never break a job
        return None


def parse_nvidia_smi_free_gb(output, uuid):
    """Free GB of the card whose UUID ends with `uuid`, from `uuid,used,total` CSV in MiB."""
    want = str(uuid).strip().lower()
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != 3 or not want or not parts[0].lower().endswith(want):
            continue
        try:
            return (float(parts[2]) - float(parts[1])) / 1024
        except ValueError:
            return None
    return None


def driver_free_vram_gb():
    """Free VRAM as the driver counts it, or None when nvidia-smi cannot say.

    Under Windows WDDM, torch's mem_get_info reports this process's budget, not
    what the card has left: on 2026-11-18 it read 7.1 GB free while nvidia-smi
    showed 7081 of 12282 MiB used — 5.1 GB. The card is matched by UUID, so
    CUDA_VISIBLE_DEVICES reordering cannot point this at another GPU.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        uuid = torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
        kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=uuid,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5, check=True, **kwargs).stdout
    except Exception:      # noqa: BLE001 — diagnostics must never break a job
        return None
    return parse_nvidia_smi_free_gb(out, uuid)


#: CTranslate2 gained a Metal backend in 4.9 (released 2026-11-10).
CT2_METAL_MIN_VERSION = (4, 9)
_ct2_metal = None


def ct2_metal_available():
    """True when CTranslate2 can run Whisper on this Mac's GPU (Metal).

    A version check and a device count — no model load, cached for the life of
    the worker, never raises. A Mac where Metal fails anyway (an older macOS,
    a chip the backend does not take) is caught when the model loads, in
    WhisperXEngine.load(), and drops to the CPU there.
    """
    global _ct2_metal
    if _ct2_metal is not None:
        return _ct2_metal
    ok = False
    if sys.platform == 'darwin':
        try:
            import ctranslate2
            version = tuple(int(p) for p in ctranslate2.__version__.split('.')[:2])
            count = getattr(ctranslate2, 'get_mps_device_count', None)
            ok = version >= CT2_METAL_MIN_VERSION and bool(count and count() > 0)
        except Exception:  # noqa: BLE001 — a probe must never break a job
            ok = False
    _ct2_metal = ok
    return ok


def detect_mac_device():
    """(device, compute_type) for whisperx on Apple Silicon.

    float16 only: CTranslate2 resolves int8_float16 on Metal to float16 itself
    ("native INT8 is slower" on current Apple GPUs), and bfloat16 decoded at
    half the speed (mac-bench, 2026-11-21). Without Metal it is the CPU at
    int8 — the same model, about 2.5x slower on an M4 Pro.
    """
    if ct2_metal_available():
        log.info('Using Metal (CTranslate2), compute_type=float16')
        return 'mps', 'float16'
    log.info('No Metal backend in this CTranslate2 — using CPU with int8.')
    return 'cpu', 'int8'


def detect_torch_device():
    """Return (device, compute_type) for CTranslate2/whisperx on this machine."""
    if sys.platform == 'darwin':
        return detect_mac_device()
    try:
        import torch
        if torch.cuda.is_available():
            compute_type = 'float16'
            vram = torch.cuda.get_device_properties(0)
            total_gb = (getattr(vram, 'total_memory', 0) or
                        getattr(vram, 'total_mem', 0)) / (1024 ** 3)
            if total_gb < 6:
                compute_type = 'int8'
            log.info('Using CUDA — %.0f GB VRAM, compute_type=%s',
                     total_gb, compute_type)
            return 'cuda', compute_type
        # cuda.is_available() is False for CPU-only PyTorch (+cpu wheels) even if the
        # NVIDIA driver and CUDA toolkit are installed system-wide.
        hint = ''
        try:
            ver = torch.__version__ or ''
            if '+cpu' in ver:
                hint = (
                    f' Installed PyTorch is CPU-only ({ver}); for GPU, reinstall torch with CUDA '
                    'from pytorch.org (e.g. cu128 matching your driver).'
                )
            elif '+' in ver and 'cu' in ver.split('+', 1)[1]:
                hint = (
                    f' PyTorch is CUDA-enabled ({ver}) but cuda.is_available() is false — '
                    'check NVIDIA driver, GPU visibility, and reboot after driver updates.'
                )
        except Exception:
            pass
        log.info('No CUDA in this PyTorch build — using CPU with int8.%s', hint)
    except ImportError:
        pass
    return 'cpu', 'int8'


def align_model_candidates(language_code: str):
    """Return ordered list of explicit align model ids to try for this language."""
    lang = (language_code or '').strip().lower()
    per_lang_env = os.environ.get(f'PINE_ALIGN_MODEL_{lang.upper()}', '').strip()
    if per_lang_env:
        return [m.strip() for m in per_lang_env.split(',') if m.strip()]
    global_env = os.environ.get('PINE_ALIGN_MODEL', '').strip()
    if global_env:
        return [m.strip() for m in global_env.split(',') if m.strip()]
    return ALIGN_MODEL_CANDIDATES.get(lang, [])


def probe_language_whisperx(pipeline, audio_np):
    """Return (language_code, confidence, options_for_ui) using first ~30s logic from whisperx."""
    import numpy as np
    from whisperx.audio import N_SAMPLES, log_mel_spectrogram

    model = getattr(pipeline, 'model', None)
    if model is None or not getattr(model.model, 'is_multilingual', True):
        return 'en', 1.0, [{'code': 'en', 'probability': 1.0}]

    audio_np = np.asarray(audio_np, dtype=np.float32).flatten()
    if audio_np.shape[0] < 4000:
        return 'en', 0.0, []

    model_n_mels = model.feat_kwargs.get('feature_size')
    segment = log_mel_spectrogram(
        audio_np[:N_SAMPLES] if audio_np.shape[0] >= N_SAMPLES else audio_np,
        n_mels=model_n_mels if model_n_mels is not None else 80,
        padding=0 if audio_np.shape[0] >= N_SAMPLES else N_SAMPLES - audio_np.shape[0],
    )
    encoder_output = model.encode(segment)
    results = model.model.detect_language(encoder_output)
    language_token, language_probability = results[0][0]
    code = str(language_token[2:-2])
    conf = float(language_probability)
    return code, conf, [{'code': code, 'probability': conf}]


class WhisperXEngine(EngineAdapter):
    id = 'whisperx'
    capabilities = EngineCapabilities(
        word_timestamps=True,
        diarization='external',
        streaming=False,
        multilingual=True,
    )

    def __init__(self, env=None):
        super().__init__(env)
        self._model = None
        self._model_dir = None
        self._device = None
        self._compute_type = None
        # Wav2Vec2 align model (reused across recordings; cleared on lang/device change).
        # On CUDA it waits in RAM between alignments — see _park_align_model.
        self._cached_align_model = None
        self._cached_align_metadata = None
        self._cached_align_language = None
        self._cached_align_device = None
        # Remember failing align model combinations to avoid repeated HF retries on each chunk/job.
        self._failed_align_keys = set()  # {(language_code, device)}

    @property
    def _hf_offline(self):
        return bool(getattr(self.env, 'hf_offline', False))

    def load(self):
        model_dir = self.env.model_dir
        if self._model is not None and self._model_dir == model_dir:
            return
        self._model = None
        if self._device is None:
            self._device, self._compute_type = detect_torch_device()
        if self._device == 'cuda':
            cap_ct2_allocator()
        if sys.platform == 'darwin':
            # Before whisperx, not after: whisperx imports pyannote for its
            # VAD, and pyannote looks for torchcodec once, at import. On a Mac
            # the real torchcodec fails to load (no LC_RPATH), so pyannote was
            # left with no decoder of its own and logged 28 lines saying so on
            # every start. Diarizer.load installs the same ffmpeg shim, but
            # since whisperx runs on the Mac that comes too late.
            compat.stub_torchcodec()
        import whisperx
        log.info('Loading WhisperX model from %s ...', model_dir)
        try:
            self._model = whisperx.load_model(
                model_dir,
                self._device,
                compute_type=self._compute_type,
            )
        except Exception as exc:
            if self._device != 'mps':
                raise
            # CTranslate2 reported Metal but the model would not come up on it
            # (an older macOS, a chip the backend does not take). Same model on
            # the CPU: slower, same transcript.
            log.warning('WhisperX on Metal failed to load (%s) — using CPU with int8', exc)
            self._device, self._compute_type = 'cpu', 'int8'
            self._model = whisperx.load_model(
                model_dir,
                self._device,
                compute_type=self._compute_type,
            )
        self._model_dir = model_dir
        log.info('WhisperX model loaded.')

    def _ct2_model(self):
        """The CTranslate2 Whisper under whisperx's pipeline, or None."""
        # FasterWhisperPipeline.model -> faster_whisper.WhisperModel.model
        return getattr(getattr(self._model, 'model', None), 'model', None)

    def _ensure_resident(self):
        """Put Whisper back on the GPU if ``release()`` took it off."""
        ct2 = self._ct2_model()
        if ct2 is not None and not getattr(ct2, 'model_is_loaded', True):
            with _perf('model reload'):
                ct2.load_model()

    def release(self):
        """Give unified memory back to the Mac between recordings (Metal only).

        The worker lives between jobs, and on Metal it used to sit on 15 GB of
        a 24 GB Mac while idle (mac-bench, 2026-11-21): Whisper 3.5 GB, the
        rest alignment on MPS (since moved to the CPU, see MPS_ALIGN_DEVICE)
        and torch's MPS cache. Whisper comes off the GPU here — 3.5 -> 0.6 GB,
        0.1s — and goes back in ``_ensure_resident`` on the next job (~1s).
        Nothing changes on CUDA or the CPU.
        """
        if self._device != 'mps':
            return
        self._park_align_model('release align park')
        ct2 = self._ct2_model()
        if ct2 is not None and getattr(ct2, 'model_is_loaded', False):
            try:
                with _perf('model unload'):
                    ct2.unload_model()
            except Exception as exc:  # noqa: BLE001 — keep the model rather than fail
                log.warning('Could not take Whisper off Metal: %s', exc)
        gc.collect()
        import torch
        torch.mps.empty_cache()

    def probe_language(self, audio_path, probe_audio):
        self._ensure_resident()
        code, conf, options = probe_language_whisperx(self._model, probe_audio)
        return LanguageProbe(
            code=code,
            confidence=conf,
            options=options,
            uncertain=conf < LANG_CONFIDENCE_MIN,
        )

    # ── alignment ──

    def _load_align_cache_first(self, whisperx, kwargs):
        """Load an align model from disk; go online only when it is not there.

        After onboarding the worker runs with Hugging Face offline. Align
        models are per language and fetched on first use, so offline mode used
        to be lifted around every load — a cached model too, which leaves the
        hub client free to ask huggingface.co about it mid-transcription. Now
        the load tries the local cache with offline mode still on, and lifts it
        only when that fails: the first recording in a new language.
        """
        if not self._hf_offline or compat.hf_network_allowed():
            return whisperx.load_align_model(**kwargs)
        try:
            return whisperx.load_align_model(**kwargs)
        except Exception as exc:
            log.info('Align model %s is not on disk (%s) — fetching it',
                     kwargs.get('model_name') or kwargs.get('language_code'),
                     str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__)
        with compat.allow_hf_network_for_align(self._hf_offline):
            return whisperx.load_align_model(**kwargs)

    def _get_align_model(self, whisperx, language_code, device):
        """Load or return cached alignment model for language_code + device."""
        key = (language_code, device)
        if key in self._failed_align_keys:
            raise RuntimeError(
                f'Alignment model previously failed for language={language_code} device={device}')
        if (self._cached_align_model is not None
                and self._cached_align_language == language_code
                and self._cached_align_device == device):
            if device in ('cuda', 'mps'):
                self._cached_align_model.to(device)  # back from RAM, see _park_align_model
            return self._cached_align_model, self._cached_align_metadata
        if self._cached_align_model is not None:
            try:
                del self._cached_align_model
            except Exception:
                pass
            self._cached_align_model = None
            self._cached_align_metadata = None
            gc.collect()
        candidates = align_model_candidates(language_code)
        attempts = candidates if candidates else [None]
        errors = []
        model_a = None
        metadata = None
        for model_name in attempts:
            try:
                kwargs = {'language_code': language_code, 'device': device}
                if model_name:
                    kwargs['model_name'] = model_name
                model_a, metadata = self._load_align_cache_first(whisperx, kwargs)
                if model_name:
                    log.info('Loaded align model override for %s: %s',
                             language_code, model_name)
                break
            except Exception as exc:
                errors.append(exc)
                if model_name:
                    log.warning(
                        'Align model failed for lang=%s model=%s: %s',
                        language_code, model_name, exc)
                else:
                    log.warning(
                        'Default align model failed for lang=%s: %s',
                        language_code, exc)
                continue

        if model_a is None or metadata is None:
            self._failed_align_keys.add(key)
            raise errors[-1] if errors else RuntimeError(
                f'Failed to load align model for language={language_code}')
        self._cached_align_model = model_a
        self._cached_align_metadata = metadata
        self._cached_align_language = language_code
        self._cached_align_device = device
        return model_a, metadata

    def _park_align_model(self, label='align park'):
        """Move the cached align model to RAM until the next alignment needs it.

        Left on the card it shares 12 GB with Whisper, pyannote and the Windows
        desktop. Measured 2026-11-18 on an 88-min recording: chunk 1, decoded
        before the align model existed, took 101s; chunks 2 and 3, decoded with
        its ~1.4 GB resident, took 851s and 836s — 10-12 min of that before the
        first batch — while the worker spilled into shared memory through WDDM.
        Re-run with parking under the same pressure (chunk 1 at 7.1 GB free):
        134s and 86s, transcript identical word for word. Each move costs
        0.5-0.7s. Dropping the model instead would mean a
        reload per chunk, each one reopening the Hugging Face network window
        and risking a transient failure that disables alignment for the rest
        of the worker's life (_failed_align_keys).

        On Metal the align model lives on the CPU anyway (MPS_ALIGN_DEVICE);
        what this buys there is torch's MPS cache — VAD and the rest — handed
        back between chunks.
        """
        if self._device not in ('cuda', 'mps') or self._cached_align_model is None:
            return
        with _perf(label):
            self._cached_align_model.to('cpu')
            gc.collect()
            import torch
            if self._device == 'cuda':
                torch.cuda.empty_cache()
            else:
                torch.mps.empty_cache()

    # On a Mac, wav2vec2 aligns on the CPU while Whisper decodes on Metal.
    # Measured 2026-11-21 on one 30-min chunk of the 88-min recording (M4 Pro,
    # 24 GB): on MPS it took 29s but drove torch's Metal allocations to 17.6 GB
    # and kept 12.5 GB of them after the model was dropped and the cache
    # emptied; a second pass took 48s. Through PINE the second chunk's
    # alignment then ran out of memory ('other allocations: 25 GiB'). On the
    # CPU the same chunk took 50s for the same 4010 words — about 40s more
    # per 88 minutes, against a GPU left to Whisper and pyannote (mac-bench).
    MPS_ALIGN_DEVICE = 'cpu'

    def _align(self, whisperx, segments, language, audio, label=''):
        """wav2vec2 alignment: on the engine's device, or the CPU beside Metal.

        ``label`` prefixes the PERF lines ('chunk 2 ' -> 'chunk 2 align').
        Failures propagate, for the caller to turn alignment off as before.
        """
        device = self.MPS_ALIGN_DEVICE if self._device == 'mps' else self._device
        with _perf(f'{label}align model'):
            model_a, metadata = self._get_align_model(whisperx, language, device)
        with _perf(f'{label}align'):
            return whisperx.align(segments, model_a, metadata, audio,
                                  device, return_char_alignments=False)

    # ── decoding ──

    # Decode working set on top of the loaded model, measured 2026-11-18 on one
    # 30-min chunk (large-v3, float16, ru, 12 GB card): batch 4 — 1.6 GB in
    # 42.7s, 8 — 3.5 GB in 33.3s, 16 — 5.9 GB in 30.2s. Each threshold adds 1 GB
    # for the desktop, which draws on the same card. Missing by even a little
    # is not 10% slower but 3-25x: the 88-min recording took 299s at batch 8
    # and 700s at 16 under the same load, the text differing in 16 of 11868
    # words either way.
    BATCH_SIZE_BY_FREE_GB = ((7.0, 16), (4.5, 8), (2.6, 4))

    # Metal, measured 2026-11-21 (M4 Pro, 24 GB, the 88-min recording): batch 16
    # decoded it in 658s, 32 in 671s — the GPU is full at 16, and the text
    # differed in 2 of 11.8k words. The GPU draws on the Mac's own RAM, so a
    # machine of 16 GB or less starts at 8. That one is not measured; the
    # halving in _transcribe_with_oom_retry is what stands behind it.
    MPS_BATCH_SIZE = 16
    MPS_BATCH_SIZE_SMALL_RAM = 8
    MPS_SMALL_RAM_GB = 16.5

    def _mps_batch_size(self):
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:  # noqa: BLE001 — diagnostics must never break a job
            ram_gb = None
        small = ram_gb is not None and ram_gb <= self.MPS_SMALL_RAM_GB
        size = self.MPS_BATCH_SIZE_SMALL_RAM if small else self.MPS_BATCH_SIZE
        log.info('Decoding on Metal with batch_size=%d (%s GB RAM)', size,
                 '?' if ram_gb is None else f'{ram_gb:.0f}')
        return size

    def _pick_batch_size(self):
        """Start from the batch the free VRAM actually supports.

        On Windows an overcommit does not raise OOM, it pages, and the halving
        path below never runs — so the batch has to fit up front, judged by
        what the driver reports rather than torch's inflated budget.
        """
        if self._device == 'mps':
            return self._mps_batch_size()
        if self._device != 'cuda':
            return 4
        free_gb, source = driver_free_vram_gb(), 'driver'
        if free_gb is None:
            free_gb, source = free_vram_gb(), 'torch'
        if free_gb is None:
            return 16
        for needed, size in self.BATCH_SIZE_BY_FREE_GB:
            if free_gb >= needed:
                log.info('Decoding with batch_size=%d (%.1f GB VRAM free per %s)',
                         size, free_gb, source)
                return size
        log.warning('Only %.1f GB VRAM free (per %s) — decoding with batch_size=2',
                    free_gb, source)
        return 2

    def _transcribe_with_oom_retry(self, audio, language=None):
        """Transcribe with automatic batch_size halving on CUDA or Metal OOM."""
        self._ensure_resident()
        batch_size = self._pick_batch_size()
        min_batch = 2

        while batch_size >= min_batch:
            tx_kw = {'batch_size': batch_size, 'print_progress': True}
            if language:
                tx_kw['language'] = language
            try:
                return self._model.transcribe(audio, **tx_kw)
            except RuntimeError as exc:
                if (_is_out_of_memory(exc, self._device)
                        and self._device in ('cuda', 'mps')
                        and batch_size > min_batch):
                    log.warning(
                        '%s OOM with batch_size=%d — halving and retrying',
                        'CUDA' if self._device == 'cuda' else 'Metal', batch_size)
                    import torch
                    if self._device == 'cuda':
                        torch.cuda.empty_cache()
                    else:
                        torch.mps.empty_cache()
                    gc.collect()
                    batch_size = batch_size // 2
                else:
                    raise

        # Final attempt with min_batch — let any error propagate
        tx_kw = {'batch_size': min_batch, 'print_progress': True}
        if language:
            tx_kw['language'] = language
        return self._model.transcribe(audio, **tx_kw)

    def _reread_unpunctuated(self, segments, audio, language, ctx=None, carry=''):
        """Decode once more every window Whisper wrote without punctuation.

        ``segments`` are whisperx's raw ones, before alignment, timed against
        ``audio`` — so the window is cut exactly as Whisper first heard it, and
        alignment afterwards times the words of whichever text stands. Each
        re-read is prompted with the transcript's own punctuated text just
        before it (``carry`` covers the start of a chunk) and replaces the
        original only when ``accept_reread`` agrees. A failed re-read keeps
        the original; the model's options are restored whatever happens.
        """
        todo = [i for i, s in enumerate(segments) if is_unpunctuated(s.get('text'))]
        if not todo or self._model is None:
            return segments
        if ctx is not None:
            ctx.on_status(stage='transcribing',
                          message=f'Restoring punctuation ({len(todo)} passages)…')
        import dataclasses
        base = self._model.options
        restored = 0
        try:
            for i in todo:
                if ctx is not None:
                    ctx.check_cancel()
                seg = segments[i]
                t0, t1 = float(seg.get('start', 0) or 0), float(seg.get('end', 0) or 0)
                a, b = max(0, int(t0 * _SAMPLE_RATE)), min(len(audio), int(t1 * _SAMPLE_RATE))
                if b - a < _SAMPLE_RATE:
                    continue
                self._model.options = dataclasses.replace(
                    base, initial_prompt=prompt_before(segments, i, carry) or None)
                kw = {'batch_size': 1}
                if language:
                    kw['language'] = language
                try:
                    out = self._model.transcribe(audio[a:b], **kw)
                except Exception as exc:  # noqa: BLE001 — the original text stands
                    log.warning('Re-reading %.1f-%.1fs failed: %s', t0, t1, exc)
                    continue
                new = ' '.join(str(s.get('text', '') or '').strip()
                               for s in out.get('segments', [])).strip()
                if accept_reread(seg.get('text'), new):
                    seg['text'] = ' ' + new
                    restored += 1
                else:
                    log.info('Re-read of %.1f-%.1fs kept the original '
                             '(no better, or too different)', t0, t1)
        finally:
            self._model.options = base
        log.info('Re-read %d window(s) Whisper left without punctuation: %d restored',
                 len(todo), restored)
        return segments

    def _transcribe_chunked(self, whisperx, audio_path, total_duration,
                            language, ctx: TranscribeContext):
        """Transcribe + align long audio in chunks to limit RAM usage.

        Returns (result_dict, detected_language).
        """
        step = CHUNK_SIZE_SEC - CHUNK_OVERLAP_SEC
        chunk_starts = []
        offset = 0
        while offset < total_duration:
            chunk_starts.append(offset)
            offset += step
        total_chunks = len(chunk_starts)

        all_segments = []
        detected_lang = language or 'en'
        chunked_start = time.monotonic()
        align_enabled = True
        carry = ''

        for i, offset in enumerate(chunk_starts):
            duration = min(CHUNK_SIZE_SEC, total_duration - offset)

            if i == 0:
                msg = f'Chunk 1/{total_chunks} — starting…'
                pct = 0
                eta = None
            else:
                elapsed = time.monotonic() - chunked_start
                avg_secs = elapsed / i
                remaining = avg_secs * (total_chunks - i)
                pct = round(i / total_chunks * 100)
                eta = round(remaining)
                msg = (f'Chunk {i + 1}/{total_chunks} ({pct}%) — '
                       f'~{fmt_elapsed(remaining)} remaining')

            ctx.on_status(stage='transcribing', message=msg, percent=pct, eta_secs=eta)
            free_gb = free_vram_gb()
            log.info('Chunk %d/%d  offset=%.0fs  duration=%.0fs%s',
                     i + 1, total_chunks, offset, duration,
                     '' if free_gb is None else f'  vram_free={free_gb:.1f}GB')

            with _perf(f'chunk {i + 1} decode'):
                chunk_audio = load_audio_range(audio_path, offset, duration)

            with _perf(f'chunk {i + 1} whisper'):
                result = self._transcribe_with_oom_retry(chunk_audio, language=language)

            lang = result.get('language', language or 'en')
            if i == 0:
                detected_lang = language or lang

            with _perf(f'chunk {i + 1} punctuation re-read'):
                self._reread_unpunctuated(result.get('segments', []), chunk_audio,
                                          language or lang, ctx, carry)
            # What the next chunk's first window is prompted with: this chunk's
            # text up to where the next one starts.
            carry = tail_text(result.get('segments', []), until=step)

            # Align (cached align model, reused across chunks and recordings)
            if align_enabled:
                try:
                    result = self._align(whisperx, result['segments'], detected_lang,
                                         chunk_audio, f'chunk {i + 1} ')
                except Exception as exc:
                    align_enabled = False
                    log.warning(
                        'Alignment failed for chunk %d (lang=%s): %s - '
                        'skipping alignment for remaining chunks',
                        i, detected_lang, exc)
                finally:
                    self._park_align_model(f'chunk {i + 1} align park')

            # Ownership boundaries — each chunk keeps segments whose midpoint
            # falls inside its exclusive zone, eliminating overlap duplicates.
            own_start = offset + (CHUNK_OVERLAP_SEC / 2 if i > 0 else 0)
            own_end = (offset + duration
                       - (CHUNK_OVERLAP_SEC / 2 if i < total_chunks - 1 else 0))

            for seg in result.get('segments', []):
                seg['start'] = seg.get('start', 0) + offset
                seg['end'] = seg.get('end', 0) + offset
                if 'words' in seg:
                    for w in seg['words']:
                        w['start'] = w.get('start', 0) + offset
                        w['end'] = w.get('end', 0) + offset
                mid = (seg['start'] + seg['end']) / 2
                if own_start <= mid < own_end:
                    all_segments.append(seg)

            del chunk_audio
            gc.collect()
            if self._device == 'cuda':
                import torch
                torch.cuda.empty_cache()
            elif self._device == 'mps':
                import torch
                torch.mps.empty_cache()
            ctx.check_cancel()

        all_segments.sort(key=lambda s: s.get('start', 0))
        return {'segments': all_segments}, detected_lang

    def transcribe(self, audio_path, total_duration, language, ctx: TranscribeContext) -> TranscribeOutput:
        import whisperx

        use_chunks = total_duration >= CHUNK_THRESHOLD_SEC and total_duration > 0
        audio = None

        if use_chunks:
            log.info('Using chunked processing (%.0f min)', total_duration / 60)
            result, detected_lang = self._transcribe_chunked(
                whisperx, audio_path, total_duration, language, ctx)
        else:
            with _perf('decode'):
                audio = whisperx.load_audio(audio_path)
            total_duration = audio.shape[0] / 16000
            device_note = ' (CPU - may be slow)' if self._device == 'cpu' else ''
            ctx.on_status(stage='transcribing',
                          message=f'Transcribing audio...{device_note}')

            _heartbeat_stop = threading.Event()
            _transcribe_start = time.monotonic()

            def _heartbeat():
                while not _heartbeat_stop.wait(15):
                    elapsed = time.monotonic() - _transcribe_start
                    ctx.on_status(
                        stage='transcribing',
                        message=f'Transcribing... {fmt_elapsed(elapsed)} elapsed{device_note}',
                    )

            threading.Thread(target=_heartbeat, daemon=True).start()
            try:
                with _perf('whisper'):
                    result = self._transcribe_with_oom_retry(audio, language=language)
            finally:
                _heartbeat_stop.set()

            detected_lang = language or result.get('language', 'en')
            with _perf('punctuation re-read'):
                self._reread_unpunctuated(result.get('segments', []), audio,
                                          detected_lang, ctx)
            ctx.on_status(stage='aligning', message='Aligning word timestamps...')
            try:
                result = self._align(whisperx, result['segments'], detected_lang, audio)
            except Exception as exc:
                log.warning('Alignment failed for lang=%s: %s - skipping', detected_lang, exc)
            finally:
                self._park_align_model()

        return TranscribeOutput(
            segments=result.get('segments', []),
            language=detected_lang,
            duration_seconds=total_duration,
            audio=audio,
        )
