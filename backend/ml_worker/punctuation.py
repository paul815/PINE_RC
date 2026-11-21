"""Find Whisper windows that came out without punctuation, and judge a re-read.

Whisper large-v3, decoding in batches, now and then writes a whole 30-second
window in lowercase with no punctuation at all, while the windows around it
are fine. The engine decodes such a window once more on its own, with the
transcript's preceding text as ``initial_prompt`` — a punctuated prompt pulls
the decoder back into punctuating — and keeps the new text only when it passes
``accept_reread``. Pure functions here; the decoding lives in the engine.
"""

import difflib
import re

from .constants import (
    REREAD_MAX_LENGTH_CHANGE,
    REREAD_MIN_SIMILARITY,
    REREAD_PROMPT_CHARS,
    UNPUNCTUATED_MAX_MARK_RATIO,
    UNPUNCTUATED_MIN_WORDS,
)

_MARK = re.compile(r'[.,?!;:…]["»”\')\]]*$')


def is_unpunctuated(text):
    """True for a long text with almost no punctuation marks."""
    words = str(text or '').split()
    if len(words) < UNPUNCTUATED_MIN_WORDS:
        return False
    marks = sum(1 for w in words if _MARK.search(w))
    return marks / len(words) < UNPUNCTUATED_MAX_MARK_RATIO


def _norm_words(text):
    out = []
    for w in str(text or '').lower().replace('ё', 'е').split():
        w = re.sub(r"[^\w']", '', w)
        if w:
            out.append(w)
    return out


def word_similarity(a, b):
    """0..1 agreement of two texts' words, ignoring case and punctuation."""
    wa, wb = _norm_words(a), _norm_words(b)
    if not wa and not wb:
        return 1.0
    return difflib.SequenceMatcher(None, wa, wb, autojunk=False).ratio()


def accept_reread(old, new):
    """Keep a re-read only if it is punctuated and still says the same thing."""
    if not new or is_unpunctuated(new):
        return False
    n_old, n_new = len(_norm_words(old)), len(_norm_words(new))
    if n_old == 0 or abs(n_new - n_old) / n_old > REREAD_MAX_LENGTH_CHANGE:
        return False
    return word_similarity(old, new) >= REREAD_MIN_SIMILARITY


def prompt_before(segments, index, carry=''):
    """The punctuated text right before ``segments[index]``, for initial_prompt.

    Walks back over earlier segments, skipping any that are themselves
    unpunctuated (they would teach the wrong style), and falls back on
    ``carry`` — the tail of the previous chunk — when the chunk has none.
    Trimmed to REREAD_PROMPT_CHARS at a word boundary.
    """
    parts = []
    length = 0
    for j in range(index - 1, -1, -1):
        text = str(segments[j].get('text', '') or '').strip()
        if not text or is_unpunctuated(text):
            continue
        parts.insert(0, text)
        length += len(text) + 1
        if length >= REREAD_PROMPT_CHARS:
            break
    if length < REREAD_PROMPT_CHARS and carry:
        parts.insert(0, carry.strip())
    return _trim_left(' '.join(parts), REREAD_PROMPT_CHARS)


def tail_text(segments, until=None):
    """Punctuated text at the end of ``segments`` (ending by ``until`` seconds)."""
    picked = [s for s in segments if until is None or float(s.get('end', 0) or 0) <= until]
    return prompt_before(picked, len(picked))


def _trim_left(text, limit):
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[-limit:]
    space = cut.find(' ')
    return cut[space + 1:] if 0 <= space < len(cut) - 1 else cut
