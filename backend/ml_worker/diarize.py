"""Speaker diarization stage (pyannote), shared by every STT engine.

``whisperx.diarize.DiarizationPipeline`` finds the speakers and
``whisperx.assign_word_speakers`` puts them on the words, on every platform.
It consumes the unified segment format and returns it with ``speaker`` filled.

The work splits in two: ``compute`` needs only the audio, ``assign`` needs only
the transcript plus what ``compute`` returned. ``run`` does both back to back;
the pipeline calls them separately so the expensive half can overlap the STT
stage it does not actually depend on.
"""

import logging
import os
import sys
import time

from . import compat
from .engines.base import _noop_check_cancel, _noop_status

log = logging.getLogger(__name__)

DEFAULT_DIARIZATION_MODEL = 'pyannote/speaker-diarization-community-1'


def speaker_kwargs(num_speakers=None):
    """Speaker-count constraint passed to the pyannote diarization pipeline.

    When the recording has an explicit speaker count, pin pyannote to
    exactly that many speakers (best quality when the count is known —
    e.g. 2 for a 1-on-1 interview, which avoids a spurious third voice).
    Otherwise fall back to the env min/max hints, which default to 2..4.
    That range is a constraint, not a guess: PINE only ever runs on
    interview recordings with two to four participants. Material outside
    that range — a monologue, or a group of five and up — needs an explicit
    num_speakers on the recording, or PINE_MIN_SPEAKERS / PINE_MAX_SPEAKERS
    for a whole batch.
    """
    try:
        n = int(num_speakers) if num_speakers else 0
    except (TypeError, ValueError):
        n = 0
    if n > 0:
        return {'num_speakers': n}
    _min = int(os.environ.get('PINE_MIN_SPEAKERS', '2'))
    _max = int(os.environ.get('PINE_MAX_SPEAKERS', '4'))
    return {'min_speakers': _min, 'max_speakers': _max}


def detect_diarize_device():
    """Preferred torch device for pyannote on this machine."""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if (sys.platform == 'darwin'
                and getattr(torch.backends.mps, 'is_available', lambda: False)()):
            return os.environ.get('PINE_DIARIZE_DEVICE', 'mps')
    except ImportError:
        pass
    return 'cpu'


def annotation_to_dataframe(annotation):
    """Convert pyannote Annotation to a DataFrame for whisperx.assign_word_speakers."""
    import pandas as pd
    rows = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        rows.append({
            'segment': turn,
            'label': None,
            'speaker': speaker,
            'start': turn.start,
            'end': turn.end,
        })
    return pd.DataFrame(rows)


def normalize_diarize_audio_input(diarize_input):
    """Return numpy float32 mono waveform for DiarizationPipeline, or str path."""
    import numpy as np
    import torch

    if isinstance(diarize_input, dict) and 'waveform' in diarize_input:
        wf = diarize_input['waveform']
        if isinstance(wf, torch.Tensor):
            wf = wf.squeeze().detach().cpu().numpy()
        arr = np.asarray(wf, dtype=np.float32).flatten()
        return arr
    if isinstance(diarize_input, torch.Tensor):
        return diarize_input.squeeze().detach().cpu().numpy().astype(np.float32).flatten()
    if isinstance(diarize_input, np.ndarray):
        return np.asarray(diarize_input, dtype=np.float32).flatten()
    return diarize_input


def instantiate_whisperx_pipeline(model_name, hf_token, device):
    """Build DiarizationPipeline; WhisperX versions differ (token vs use_auth_token)."""
    import inspect

    import torch
    from whisperx.diarize import DiarizationPipeline

    if isinstance(device, str):
        device = torch.device(device)
    params = inspect.signature(DiarizationPipeline.__init__).parameters
    names = set(params)
    kwargs = {'device': device}
    if 'model_name' in names:
        kwargs['model_name'] = model_name
    elif 'model_config' in names:
        kwargs['model_config'] = model_name
    else:
        kwargs['model_name'] = model_name
    auth = hf_token or None
    if 'token' in names:
        kwargs['token'] = auth
    elif 'use_auth_token' in names:
        kwargs['use_auth_token'] = auth
    elif 'hf_token' in names:
        kwargs['hf_token'] = auth
    if 'cache_dir' in names:
        kwargs['cache_dir'] = None
    return DiarizationPipeline(**kwargs)


class Diarizer:
    """Loads the pyannote pipeline once and labels speakers per recording."""

    def __init__(self, env=None, device=None):
        self.env = env
        self.pipeline = None
        self.device = device

    # ── loading ──

    def load(self):
        if self.pipeline is not None:
            return
        compat.patch_torchaudio_for_pyannote()
        compat.patch_hf_hub_legacy_use_auth_token()
        compat.patch_hf_hub_is_offline_mode()
        import warnings
        # ``message`` is matched against the start of the warning text, and
        # pyannote's begins with a newline — so the old bare 'torchcodec'
        # pattern never matched and the notice reached every log.
        warnings.filterwarnings('ignore', message=r'(?s).*torchcodec')
        if sys.platform == 'darwin':
            compat.stub_torchcodec()
        if self.device is None:
            self.device = detect_diarize_device()

        # pyannote resolves sub-models with hf_hub_download(..., cache_dir=PYANNOTE_CACHE).
        # Must be set before whisperx/pyannote import so CACHE_DIR in pyannote is correct.
        os.makedirs(self.env.pyannote_cache, exist_ok=True)
        os.environ['PYANNOTE_CACHE'] = self.env.pyannote_cache

        diarize_dir = self.env.diarize_dir
        config_file = os.path.join(diarize_dir, 'config.yaml')

        log.info('Loading diarization pipeline from %s ...', diarize_dir)

        import torch

        model_name = (config_file if os.path.isfile(config_file)
                      else DEFAULT_DIARIZATION_MODEL)

        def _make(device):
            return instantiate_whisperx_pipeline(
                model_name, self.env.hf_token, device)

        try:
            self.pipeline = _make(torch.device(self.device))
            log.info('Diarization pipeline on %s', self.device)
        except Exception as exc:
            if self.device != 'cpu':
                log.warning('Diarization init on %s failed (%s) — retrying on CPU',
                            self.device, exc)
                self.device = 'cpu'
                self.pipeline = _make(torch.device('cpu'))
            else:
                raise
        log.info('Diarization pipeline loaded.')

    # ── running ──

    def run(self, diarize_input, segments, recording_id,
            audio_path=None, total_duration=0, num_speakers=None,
            on_status=_noop_status):
        """Diarize and label ``segments`` in one pass. Returns the segments."""
        result = self.compute(
            diarize_input, recording_id, audio_path=audio_path,
            total_duration=total_duration, num_speakers=num_speakers,
            on_status=on_status)
        return self.assign(result, segments)

    def compute(self, diarize_input, recording_id, audio_path=None,
                total_duration=0, num_speakers=None,
                on_status=_noop_status, check_cancel=_noop_check_cancel):
        """Work out who spoke when — the half of the job that needs only audio.

        Kept separate from ``assign`` so the pipeline can start diarizing while
        the engine is still transcribing: pyannote never reads the transcript,
        only the speaker *assignment* does. Returns an opaque handle for
        ``assign``, or None when diarization produced nothing usable.

        ``audio_path``, ``total_duration`` and ``check_cancel`` are part of the
        stage interface the pipeline calls; whisperx's pipeline takes the whole
        recording in one call and needs none of them.
        """
        return self._compute_whisperx(
            diarize_input, num_speakers=num_speakers, on_status=on_status)

    def assign(self, result, segments):
        """Attach the speaker labels from ``compute`` to transcript segments."""
        if result is None:
            return segments

        t0 = time.monotonic()
        import whisperx
        fill_nearest = os.environ.get('PINE_DIARIZE_FILL_NEAREST', '').strip() == '1'
        transcript_result = whisperx.assign_word_speakers(
            result,
            {'segments': segments},
            fill_nearest=fill_nearest,
        )
        segments = transcript_result.get('segments', segments)
        log.info('PERF: speaker assignment took %.1fs', time.monotonic() - t0)
        return segments

    def _compute_whisperx(self, diarize_input, num_speakers=None,
                          on_status=_noop_status):
        """whisperx's DiarizationPipeline, as a DataFrame."""
        import pandas as pd

        t0 = time.monotonic()
        diarize_input = normalize_diarize_audio_input(diarize_input)
        log.info('PERF: diarize audio prep took %.1fs', time.monotonic() - t0)
        _spk_kwargs = speaker_kwargs(num_speakers)

        t1 = time.monotonic()
        try:
            diarize_segments = self.pipeline(diarize_input, **_spk_kwargs)
        except Exception as exc:
            if self.device != 'cpu':
                import torch
                log.warning(
                    'Diarization on %s failed: %s — retrying on CPU',
                    self.device, exc)
                on_status(stage='diarizing',
                          message='Speaker ID failed on GPU, retrying on CPU…')
                self.pipeline.model.to(torch.device('cpu'))
                self.device = 'cpu'
                diarize_segments = self.pipeline(diarize_input, **_spk_kwargs)
            else:
                raise
        log.info('PERF: diarize pipeline (%s) took %.1fs',
                 self.device, time.monotonic() - t1)

        if sys.platform == 'darwin':
            compat.clear_torchcodec_decode_cache()

        if isinstance(diarize_segments, tuple):
            diarize_segments = diarize_segments[0]

        for _attr in ('exclusive_speaker_diarization', 'speaker_diarization'):
            _v = getattr(diarize_segments, _attr, None)
            if _v is not None:
                diarize_segments = annotation_to_dataframe(_v)
                break
        else:
            if hasattr(diarize_segments, 'itertracks') and not isinstance(
                    diarize_segments, pd.DataFrame):
                diarize_segments = annotation_to_dataframe(diarize_segments)

        return diarize_segments
