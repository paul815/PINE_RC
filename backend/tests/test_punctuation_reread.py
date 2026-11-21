"""Re-reading windows Whisper wrote without punctuation.

The detection and acceptance rules are pure functions; the engine test runs
WhisperXEngine._reread_unpunctuated against a fake whisperx pipeline.
"""
import dataclasses
import types

import pytest

from ml_worker.punctuation import (
    accept_reread,
    is_unpunctuated,
    prompt_before,
    tail_text,
    word_similarity,
)

# From the IELTS test recording, 06:12, as Whisper first wrote it.
COLLAPSED = ('and so on so yeah hyundai is definitely a place i would uh recommend '
             'for the visitors and also i actually recommended to my friends um which '
             'i met back in uh germany uh they actually knew about hyundai')
# The same window re-read with the preceding text as prompt.
REREAD = ('And so on. So yeah, Haeundae is definitely a place I would recommend for '
          'the visitors. And also, I actually recommend it to my friends, which I met '
          'back in Germany. They actually knew about Haeundae')


def test_a_long_text_without_marks_is_unpunctuated():
    assert is_unpunctuated(COLLAPSED)


def test_normal_text_and_short_text_are_not():
    assert not is_unpunctuated(REREAD)
    assert not is_unpunctuated('yes i think so')            # too short to tell
    assert not is_unpunctuated('')


def test_a_punctuated_re_read_of_the_same_words_is_accepted():
    assert word_similarity(COLLAPSED, REREAD) > 0.8
    assert accept_reread(COLLAPSED, REREAD)


def test_a_re_read_that_is_still_unpunctuated_is_rejected():
    assert not accept_reread(COLLAPSED, COLLAPSED.replace('uh ', ''))


def test_a_re_read_that_says_something_else_is_rejected():
    other = ('Thank you for watching. Please subscribe to the channel, and see you '
             'next time. Bye, everyone. Thanks again for being here, okay? Good.')
    assert not accept_reread(COLLAPSED, other)


def test_a_re_read_that_loses_a_third_of_the_words_is_rejected():
    words = REREAD.split()
    assert not accept_reread(COLLAPSED, ' '.join(words[: len(words) // 2]))


def test_empty_re_read_is_rejected():
    assert not accept_reread(COLLAPSED, '')


def test_prompt_is_the_punctuated_text_before_and_skips_collapsed_windows():
    segs = [{'text': ' It is located in Busan.'},
            {'text': ' You can search it on Google Maps.'},
            {'text': COLLAPSED},
            {'text': COLLAPSED}]
    assert prompt_before(segs, 3) == 'It is located in Busan. You can search it on Google Maps.'


def test_prompt_falls_back_on_the_previous_chunk():
    segs = [{'text': COLLAPSED}]
    assert prompt_before(segs, 0, carry='Earlier text.') == 'Earlier text.'
    assert prompt_before(segs, 0) == ''


def test_prompt_is_trimmed_at_a_word_boundary():
    from ml_worker.constants import REREAD_PROMPT_CHARS
    segs = [{'text': ' ' + 'Word, ' * 200}, {'text': COLLAPSED}]
    p = prompt_before(segs, 1)
    assert len(p) <= REREAD_PROMPT_CHARS
    assert p.startswith('Word,')


def test_tail_text_stops_at_the_next_chunk():
    segs = [{'text': ' First.', 'end': 10.0}, {'text': ' Second.', 'end': 1790.0}]
    assert tail_text(segs, until=1770.0) == 'First.'
    assert tail_text(segs) == 'First. Second.'


# ── the engine ──

@dataclasses.dataclass
class _Opts:
    initial_prompt: str | None = None
    beam_size: int = 5


class _FakePipeline:
    """Stands in for whisperx's FasterWhisperPipeline."""

    def __init__(self, answer):
        self.options = _Opts()
        self.answer = answer
        self.calls = []

    def transcribe(self, audio, batch_size=None, language=None):
        self.calls.append({'n': len(audio), 'prompt': self.options.initial_prompt,
                           'language': language, 'batch_size': batch_size})
        if isinstance(self.answer, Exception):
            raise self.answer
        return {'segments': [{'text': ' ' + self.answer, 'start': 0.0, 'end': 1.0}]}


def _engine(pipeline):
    from ml_worker.engines.whisperx_engine import WhisperXEngine
    eng = WhisperXEngine(env=types.SimpleNamespace(hf_offline=False))
    eng._model = pipeline
    return eng


def _segments():
    return [
        {'text': ' I would recommend Haeundae.', 'start': 0.0, 'end': 3.0},
        {'text': ' ' + COLLAPSED, 'start': 3.0, 'end': 20.0},
        {'text': ' Visiting. Yes.', 'start': 20.0, 'end': 22.0},
    ]


def test_engine_replaces_an_unpunctuated_window_with_its_re_read():
    import numpy as np
    fake = _FakePipeline(REREAD)
    segs = _segments()
    _engine(fake)._reread_unpunctuated(segs, np.zeros(16000 * 22, np.float32), 'en')
    assert segs[1]['text'] == ' ' + REREAD
    assert segs[0]['text'] == ' I would recommend Haeundae.'          # untouched
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call['n'] == 16000 * 17                                    # exactly 3-20 s
    assert call['prompt'] == 'I would recommend Haeundae.'
    assert call['language'] == 'en'
    assert fake.options.initial_prompt is None                        # restored


def test_engine_keeps_the_original_when_the_re_read_fails():
    import numpy as np
    fake = _FakePipeline(RuntimeError('decoder fell over'))
    segs = _segments()
    _engine(fake)._reread_unpunctuated(segs, np.zeros(16000 * 22, np.float32), 'en')
    assert segs[1]['text'] == ' ' + COLLAPSED
    assert fake.options.initial_prompt is None


def test_engine_keeps_the_original_when_the_re_read_is_no_better():
    import numpy as np
    fake = _FakePipeline(COLLAPSED)
    segs = _segments()
    _engine(fake)._reread_unpunctuated(segs, np.zeros(16000 * 22, np.float32), 'en')
    assert segs[1]['text'] == ' ' + COLLAPSED


def test_engine_does_nothing_when_every_window_is_punctuated():
    import numpy as np
    fake = _FakePipeline(REREAD)
    segs = [_segments()[0], _segments()[2]]
    _engine(fake)._reread_unpunctuated(segs, np.zeros(16000 * 22, np.float32), 'en')
    assert fake.calls == []


def test_engine_stops_when_the_job_is_cancelled():
    import numpy as np

    from ml_worker.errors import TranscriptionCancelled

    fake = _FakePipeline(REREAD)

    def cancel():
        raise TranscriptionCancelled(1)

    ctx = types.SimpleNamespace(on_status=lambda **kw: None, check_cancel=cancel)
    with pytest.raises(TranscriptionCancelled):
        _engine(fake)._reread_unpunctuated(_segments(), np.zeros(16000 * 22, np.float32), 'en', ctx)
    assert fake.calls == []
    assert fake.options.initial_prompt is None
