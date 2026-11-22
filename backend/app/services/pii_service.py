"""PII detection and redaction using GLiNER."""

import importlib.util
import logging
import os
import warnings

from ..extensions import db
from ..models.ml_model import MLModel
from ..models.setting import Setting

log = logging.getLogger(__name__)

_gliner_model = None

# Entity types GLiNER should detect
PII_LABELS = [
    'person', 'email', 'phone number', 'address',
    'credit card number', 'social security number',
    'date of birth', 'organization', 'location',
]

# Common words GLiNER falsely flags as PII (pronouns, articles, etc.)
_PII_STOPWORDS = {
    'i', 'we', 'he', 'she', 'it', 'they', 'me', 'my', 'our', 'us',
    'you', 'your', 'the', 'a', 'an', 'this', 'that',
}


class PIIError(RuntimeError):
    """Raised when PII removal cannot proceed."""


def is_available():
    """Check if GLiNER model is downloaded, weights are present, and package is importable."""
    if importlib.util.find_spec('gliner') is None:
        return False
    m = db.session.get(MLModel, 'gliner-pii')
    if not m or m.status not in ('downloaded', 'ready'):
        return False
    model_path = m.path or ''
    if not model_path or not os.path.isdir(model_path):
        return False
    # Check for any weight file — names vary across model versions and formats
    weight_exts = ('.safetensors', '.bin', '.pt', '.onnx')
    has_weights = any(
        f.endswith(weight_exts)
        for f in os.listdir(model_path)
        if os.path.isfile(os.path.join(model_path, f))
    )
    if not has_weights:
        # DB says downloaded but weights are missing — reset so re-download is offered
        log.warning('GLiNER weights missing from %s; resetting status to not_downloaded', model_path)
        m.status = 'not_downloaded'
        m.downloaded_bytes = 0
        db.session.commit()
        return False
    return True


def _load_model(app):
    """Lazy-load the GLiNER model. Raises PIIError on failure."""
    global _gliner_model
    if _gliner_model is not None:
        return _gliner_model

    with app.app_context():
        if not is_available():
            raise PIIError(
                'GLiNER PII model is not available. '
                'Install the gliner package and download the model via Settings.'
            )
        models_path = Setting.get('models_path', app.config['DEFAULT_MODELS_PATH'])

    model_dir = os.path.join(models_path, 'gliner-pii')

    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise PIIError(
            'The gliner Python package is not installed. '
            'Re-run model setup from Settings to install it.'
        ) from exc

    # Force fully-offline loading so GLiNER + transformers never hit HuggingFace.
    # GLiNER doesn't propagate local_files_only to its internal tokenizer init.
    # We patch at three levels to be thorough:
    #   1. huggingface_hub module constant (checked by hub API calls)
    #   2. HF_HUB_OFFLINE env var (checked by some code paths)
    #   3. socket.getaddrinfo (ultimate fallback — blocks any leaked requests)
    import socket as _sock
    _real_getaddrinfo = _sock.getaddrinfo

    def _blocked_getaddrinfo(*args, **kwargs):
        host = args[0] if args else ''
        if isinstance(host, str) and 'huggingface' in host:
            raise OSError(f'Network blocked for offline model loading: {host}')
        return _real_getaddrinfo(*args, **kwargs)

    import huggingface_hub.constants as _hf_const
    _prev_const = getattr(_hf_const, 'HF_HUB_OFFLINE', False)
    _prev_env = os.environ.get('HF_HUB_OFFLINE')

    _hf_const.HF_HUB_OFFLINE = True
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    _sock.getaddrinfo = _blocked_getaddrinfo
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*sentencepiece.*byte fallback.*')
            warnings.filterwarnings('ignore', message='.*incorrect regex pattern.*')
            warnings.filterwarnings('ignore', message='.*truncate to max_length.*no maximum length.*')
            _gliner_model = GLiNER.from_pretrained(model_dir, local_files_only=True)
        log.info('GLiNER PII model loaded from %s', model_dir)
        return _gliner_model
    except OSError as exc:
        if 'Network blocked' in str(exc):
            raise PIIError(
                'GLiNER tried to fetch files from HuggingFace but network was blocked. '
                'Tokenizer files may be missing — reinstall the PII model from Settings.'
            ) from exc
        raise PIIError(f'Could not load GLiNER model: {exc}') from exc
    except Exception as exc:
        raise PIIError(f'Could not load GLiNER model: {exc}') from exc
    finally:
        _sock.getaddrinfo = _real_getaddrinfo
        _hf_const.HF_HUB_OFFLINE = _prev_const
        if _prev_env is None:
            os.environ.pop('HF_HUB_OFFLINE', None)
        else:
            os.environ['HF_HUB_OFFLINE'] = _prev_env
        os.environ.pop('TRANSFORMERS_OFFLINE', None)


def detect_pii(text, threshold=0.6):
    """Detect PII entities in text. Returns list of {start, end, label, text}.

    Raises PIIError if the detector fails. It used to log and return [],
    which a caller cannot tell from "nothing found" — so an export asked to
    remove PII went out unredacted, without a word to the user.
    """
    if _gliner_model is None:
        return []
    try:
        entities = _gliner_model.predict_entities(text, PII_LABELS, threshold=threshold)
        result = []
        for e in entities:
            start = e.get('start')
            end = e.get('end')
            txt = e.get('text', '')
            label = e.get('label', 'PII')
            # Skip common pronouns/articles that GLiNER misclassifies
            stripped = txt.strip()
            if len(stripped) < 2 or stripped.lower() in _PII_STOPWORDS:
                continue
            if start is not None and end is not None:
                result.append({'start': start, 'end': end, 'label': label, 'text': txt})
            elif txt:
                # Fallback: some GLiNER versions return text only; find spans
                idx = 0
                while True:
                    pos = text.find(txt, idx)
                    if pos < 0:
                        break
                    result.append({'start': pos, 'end': pos + len(txt), 'label': label, 'text': txt})
                    idx = pos + 1
        return result
    except Exception as exc:
        log.warning('PII detection failed: %s', exc)
        raise PIIError(f'PII detection failed: {exc}') from exc


def _unchanged_offset(offset, side='start'):
    return offset


def redact_with_offsets(text, entities):
    """Redact *entities* in *text*; return ``(new_text, where)``.

    ``where(offset, side)`` says where an offset into the old text lands in the
    new one: unchanged before an entity, shifted by the change in length after
    it, and — for an offset inside an entity — the marker's start
    (``side='start'``) or end (``side='end'``), so an anchor that covered part
    of a name now covers its marker. Tags and comments are stored as such
    offsets; without this they slid off their words after every redaction.
    Overlapping entities become one marker.
    """
    spans = []
    usable = (e for e in entities or [] if e.get('start') is not None and e.get('end') is not None)
    for ent in sorted(usable, key=lambda e: (e['start'], -e['end'])):
        s = max(0, min(int(ent['start']), len(text)))
        e = max(0, min(int(ent['end']), len(text)))
        if e <= s:
            continue
        if spans and s < spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], e)
            continue
        spans.append([s, e, ent.get('label', 'PII')])

    parts, pos, edits = [], 0, []
    for s, e, label in spans:
        marker = f'[{label} redacted]'
        parts += [text[pos:s], marker]
        edits.append((s, e, len(marker)))
        pos = e
    parts.append(text[pos:])

    def where(offset, side='start'):
        delta = 0
        for s, e, n in edits:
            if offset <= s:
                break
            if offset < e:
                return s + delta + (n if side == 'end' else 0)
            delta += n - (e - s)
        return offset + delta

    return ''.join(parts), where


def redact_text(text, entities=None, replacement='[REDACTED]'):
    """Replace PII entities in text with a redaction marker."""
    if entities is None:
        entities = detect_pii(text)
    if not entities:
        return text
    return redact_with_offsets(text, entities)[0]


def redact_segments_with_offsets(segments, threshold=0.6):
    """Redact PII from transcript segments; return ``(segments, offset_maps)``.

    ``offset_maps[i]`` is segment i's ``where`` from ``redact_with_offsets``
    (the identity where nothing was found), for moving annotation anchors.
    """
    if _gliner_model is None:
        raise PIIError('GLiNER model is not loaded. Call _load_model() first.')
    redacted, maps = [], []
    for seg in segments:
        new_seg = dict(seg)
        where = _unchanged_offset
        text = new_seg.get('text', '')
        if text:
            entities = detect_pii(text, threshold=threshold)
            if entities:
                new_seg['text'], where = redact_with_offsets(text, entities)
        redacted.append(new_seg)
        maps.append(where)
    return redacted, maps


def redact_segments(segments, threshold=0.6):
    """Redact PII from a list of transcript segments. Returns new list."""
    return redact_segments_with_offsets(segments, threshold=threshold)[0]
