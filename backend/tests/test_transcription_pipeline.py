"""Tier 2 tests: Mocked transcription pipeline — speaker assignment, chunking,
cancellation, requeue, and status flow.

The ML pipeline lives in the Flask-free ``ml_worker`` package; queue/cancel/
status orchestration stays in ``app.services.transcription``. Tests below
target whichever side owns the logic.
"""

import os
from collections import namedtuple
from unittest.mock import MagicMock, patch

import pytest


def _module_available(name):
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# Heavy ML packages are installed dynamically by model_manager.py during
# onboarding, not pinned in requirements.txt. A bare checkout (and CI) has no
# torch, so the tests that exercise real tensor code are skipped rather than
# failed — the mocked tests above them still run everywhere.
# Requires both: these exercise real tensor code and its numpy interop, and a
# torch present without numpy is a half-finished install rather than a usable
# one — checking only torch lets such a state through and the tests then fail
# on the numpy import instead of skipping.
needs_torch = pytest.mark.skipif(
    not (_module_available('torch') and _module_available('numpy')),
    reason='requires the torch/numpy stack, installed during onboarding rather than from requirements.txt',
)

needs_numpy = pytest.mark.skipif(
    not _module_available('numpy'),
    reason='requires numpy, pulled in with the ML stack during onboarding',
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal mock of pyannote Segment (named tuple with start/end)
PyannoteSegment = namedtuple('PyannoteSegment', ['start', 'end'])


class FakeDiarization:
    """Mimics pyannote.core.Annotation enough for assign_speakers_simple."""

    def __init__(self, turns):
        """turns: list of (start, end, speaker_label)"""
        self._turns = turns

    def itertracks(self, yield_label=False):
        for start, end, speaker in self._turns:
            seg = PyannoteSegment(start, end)
            if yield_label:
                yield seg, None, speaker
            else:
                yield seg, None


def _make_diarizer(engine_kind='mlx', device='cpu', pipeline=None):
    """Create a Diarizer without loading any ML models."""
    from ml_worker.diarize import Diarizer
    d = Diarizer(env=None, engine_kind=engine_kind, device=device)
    d.pipeline = pipeline
    return d


# ---------------------------------------------------------------------------
# 0. MLX segment text sync (Mac-native UI contract)
# ---------------------------------------------------------------------------

class TestSynchronizeMlxSegmentsForUi:
    def test_joins_trimmed_words_into_segment_text(self):
        from ml_worker.engines.mlx_engine import synchronize_segments_for_ui
        segs = [{
            'start': 0.0,
            'end': 1.0,
            'text': 'wrong',
            'words': [
                {'word': ' Hello', 'start': 0.0, 'end': 0.5},
                {'word': ' world', 'start': 0.5, 'end': 1.0},
            ],
        }]
        synchronize_segments_for_ui(segs)
        assert segs[0]['text'] == 'Hello world'
        assert segs[0]['words'][0]['word'] == 'Hello'
        assert segs[0]['words'][1]['word'] == 'world'

    def test_keeps_a_hyphenated_word_whole(self):
        """Whisper splits "как-то" into two tokens and marks the join by *not*
        putting a space on the second. Joining on spaces printed "как -то" 89
        times in one 38-minute interview."""
        from ml_worker.engines.mlx_engine import synchronize_segments_for_ui
        segs = [{
            'words': [
                {'word': ' во', 'start': 0.0, 'end': 0.3},
                {'word': '-первых', 'start': 0.3, 'end': 0.8},
                {'word': ' как', 'start': 0.9, 'end': 1.2},
                {'word': '-то', 'start': 1.2, 'end': 1.5},
            ],
        }]
        synchronize_segments_for_ui(segs)
        assert segs[0]['text'] == 'во-первых как-то'
        # The UI locates each word inside the text with indexOf, so every word it
        # is handed still has to be findable there.
        for w in segs[0]['words']:
            assert w['word'] in segs[0]['text']

    def test_falls_back_to_spaces_for_words_that_arrive_trimmed(self):
        """Nothing left to read the spacing from, so the old join is the only
        sane answer — which is also what makes running this twice a no-op."""
        from ml_worker.engines.mlx_engine import synchronize_segments_for_ui
        segs = [{'words': [{'word': 'Hello'}, {'word': 'world'}]}]
        synchronize_segments_for_ui(segs)
        assert segs[0]['text'] == 'Hello world'

    def test_is_idempotent(self):
        from ml_worker.engines.mlx_engine import synchronize_segments_for_ui
        segs = [{'words': [{'word': ' как'}, {'word': '-то'}, {'word': ' вот'}]}]
        synchronize_segments_for_ui(segs)
        once = segs[0]['text']
        synchronize_segments_for_ui(segs)
        assert segs[0]['text'] == once == 'как-то вот'

    def test_an_abbreviation_comes_back_as_one_word(self):
        """Whisper opens a word on the full stop, so "т.д." arrives in two pieces,
        and the saved transcript, rebuilt from the words, printed "т .д."."""
        from ml_worker.engines.mlx_engine import synchronize_segments_for_ui
        from ml_worker.pipeline import clean_transcript_segments
        segs = [{
            'start': 0.0, 'end': 2.0, 'speaker': 'A',
            'words': [
                {'word': ' и', 'start': 0.0, 'end': 0.2, 'probability': 0.9},
                {'word': ' т', 'start': 0.2, 'end': 0.4, 'probability': 0.9},
                {'word': '.д.', 'start': 0.4, 'end': 0.7, 'probability': 0.6},
                {'word': ' сайт', 'start': 0.8, 'end': 1.2, 'probability': 0.9},
                {'word': ' ya', 'start': 1.2, 'end': 1.5, 'probability': 0.9},
                {'word': '.ru', 'start': 1.5, 'end': 2.0, 'probability': 0.9},
            ],
        }]
        synchronize_segments_for_ui(segs)

        assert segs[0]['text'] == 'и т.д. сайт ya.ru'
        assert [w['word'] for w in segs[0]['words']] == ['и', 'т.д.', 'сайт', 'ya.ru']
        merged = segs[0]['words'][1]
        assert (merged['start'], merged['end'], merged['probability']) == (0.2, 0.7, 0.6)
        assert clean_transcript_segments(segs)[0]['text'] == 'и т.д. сайт ya.ru'

    def test_a_full_stop_on_its_own_is_not_a_word_tail(self):
        from ml_worker.engines.mlx_engine import synchronize_segments_for_ui
        segs = [{'words': [{'word': ' ну'}, {'word': '...'}, {'word': ' вот'}]}]
        synchronize_segments_for_ui(segs)
        assert [w['word'] for w in segs[0]['words']] == ['ну', '...', 'вот']


# ---------------------------------------------------------------------------
# 0b. Keeping non-speech away from mlx-whisper (Mac-native; whisperx has a VAD)
# ---------------------------------------------------------------------------

def _tone(secs, freq=425.0, sample_rate=16000, on=None, off=None):
    """A dial tone, optionally rung ``on`` seconds every ``on + off``."""
    import numpy as np
    t = np.arange(int(secs * sample_rate)) / sample_rate
    wave = 0.3 * np.sin(2 * np.pi * freq * t)
    if on is not None:
        wave = wave * ((t % (on + off)) < on)
    return wave.astype(np.float32)


def _voice(secs, sample_rate=16000, seed=0):
    """A 120 Hz harmonic stack with vibrato, breath noise and pauses.

    Not speech, but it has what this code reads speech by: a stack of harmonics
    where a tone has one line, and loud/quiet structure for the energy gate.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    t = np.arange(int(secs * sample_rate)) / sample_rate
    f0 = 120 + 8 * np.sin(2 * np.pi * 3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    wave = sum((1.0 / h) * np.sin(h * phase) for h in range(1, 40))
    wave = wave * 0.2 * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
    wave = wave * (np.sin(2 * np.pi * t / 7.0) > -0.7)      # pauses between turns
    return (wave + rng.normal(0, 0.002, t.size)).astype(np.float32)


def _telephone(wave, sample_rate=16000):
    """Band-limit to 300-3400 Hz, as a phone line does before Whisper ever sees it."""
    import numpy as np
    spectrum = np.fft.rfft(wave)
    freqs = np.fft.rfftfreq(wave.size, 1.0 / sample_rate)
    spectrum[(freqs < 300) | (freqs > 3400)] = 0
    return np.fft.irfft(spectrum, wave.size).astype(np.float32)


class TestSpectralFlatness:
    def test_a_tone_and_a_voice_land_orders_of_magnitude_apart(self):
        from ml_worker.constants import MLX_VAD_MIN_FLATNESS
        from ml_worker.engines.mlx_engine import spectral_flatness
        assert spectral_flatness(_tone(5)) < MLX_VAD_MIN_FLATNESS / 100
        assert spectral_flatness(_voice(5)) > MLX_VAD_MIN_FLATNESS * 10

    def test_a_phone_line_does_not_collapse_the_measure(self):
        """The reason flatness is read inside a band. Across the whole spectrum
        the empty bins above 3.4 kHz dominate the geometric mean and drag a voice
        down past any threshold that would separate it from a tone."""
        from ml_worker.constants import MLX_VAD_MIN_FLATNESS
        from ml_worker.engines.mlx_engine import spectral_flatness
        assert spectral_flatness(_telephone(_tone(5))) < MLX_VAD_MIN_FLATNESS
        assert spectral_flatness(_telephone(_voice(5))) > MLX_VAD_MIN_FLATNESS * 10

    def test_too_little_audio_is_not_a_tone(self):
        import numpy as np
        from ml_worker.engines.mlx_engine import spectral_flatness
        assert spectral_flatness(np.zeros(64, dtype=np.float32)) == 1.0


class TestGateSpeechForMlx:
    def test_ringback_is_cut_off_the_front_and_speech_survives(self):
        """The failure this exists for: a call that opens on ringback, which the
        energy gate passes as 'loud' and Whisper writes down as 'Звук колокола.'"""
        import numpy as np
        from ml_worker.engines.mlx_engine import gate_speech
        audio = np.concatenate([
            np.zeros(5 * 16000, dtype=np.float32),
            _tone(20, on=1.0, off=4.0),
            np.zeros(5 * 16000, dtype=np.float32),
            _voice(60),
        ])
        gated, splices = gate_speech(audio)
        assert splices is not None, 'expected the ringback to be cut'
        # Everything kept comes from after the tone ends at 30s.
        assert min(original for _, _, original in splices) >= 25.0
        assert len(gated) < len(audio)

    def test_audio_with_nothing_to_cut_is_passed_through_untouched(self):
        from ml_worker.engines.mlx_engine import gate_speech
        audio = _voice(60)
        gated, splices = gate_speech(audio)
        assert splices is None
        assert gated is audio

    def test_the_pauses_of_a_conversation_are_not_worth_a_seam(self):
        """What this gate is not for. An 88-minute interview came back cut at 18
        places for 29s in total — half a percent — and the pauses it took were the
        ones a window boundary could have landed in. Each seam is also a place
        ``remap`` has to carry words across. Only dead air earns one."""
        import numpy as np
        from ml_worker.engines.mlx_engine import gate_speech
        audio = np.concatenate([x for _ in range(6) for x in
                                (_voice(30), np.zeros(3 * 16000, dtype=np.float32))])
        gated, splices = gate_speech(audio)

        assert splices is None, 'three-second pauses are not dead air'
        assert gated is audio

    def test_a_long_dead_stretch_is_still_cut(self):
        """And what it is for: the same take with one real stretch of nothing."""
        import numpy as np
        from ml_worker.engines.mlx_engine import gate_speech
        audio = np.concatenate([_voice(30),
                                np.zeros(3 * 16000, dtype=np.float32),
                                _voice(30),
                                np.zeros(40 * 16000, dtype=np.float32),
                                _voice(30)])
        gated, splices = gate_speech(audio)

        assert splices is not None
        assert len(splices) == 2, 'expected the 40s of nothing cut and nothing else'
        assert 30 < (len(audio) - len(gated)) / 16000 < 45

    def test_a_gate_that_would_swallow_the_recording_is_ignored(self):
        """The dead-man's switch. Whatever the thresholds decide, dropping half
        the recording is a misread, and the old behaviour is the safe one."""
        import numpy as np
        from ml_worker.engines.mlx_engine import gate_speech
        # A tone throughout: read as speech by energy, as a tone by flatness,
        # leaving nothing at all — so the audio has to go through as it is.
        audio = np.concatenate([np.zeros(2 * 16000, dtype=np.float32), _tone(40)])
        gated, splices = gate_speech(audio)
        assert splices is None
        assert gated is audio

    def test_silence_is_not_mistaken_for_a_recording_to_gate(self):
        import numpy as np
        from ml_worker.engines.mlx_engine import gate_speech
        audio = np.zeros(30 * 16000, dtype=np.float32)
        gated, splices = gate_speech(audio)
        assert splices is None
        assert gated is audio


class _FakeMlx:
    """Stands in for mlx_whisper: records each call's arguments, replies to order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def transcribe(self, path, **kw):
        self.calls.append(kw)
        if self.replies:
            return self.replies.pop(0)
        return {'segments': [], 'language': 'ru'}


def _said(text, start=0.0, end=2.0):
    return {'start': start, 'end': end, 'text': text}


class TestTrimTail:
    """Where one window ends and the next begins. Looking for a silence to cut in
    was the first answer and it does not survive the gate running before it: with
    the pauses compacted to 0.2s, 43 of 44 window seams on an 88-minute interview
    fell at exactly 120.000s, most of them mid-word. So the seam follows the model
    instead — the sentence still going when the window ran out is read again."""

    def test_the_unfinished_last_segment_is_read_again_next_window(self):
        from ml_worker.engines.mlx_engine import trim_tail
        segments = [_said('Первое предложение.', 0.0, 60.0),
                    _said('Второе предло', 60.0, 120.0)]
        kept, resume = trim_tail(segments, start=0.0, end=120.0, total=600.0)

        assert [seg['text'] for seg in kept] == ['Первое предложение.']
        assert resume == 60.0, 'the next window has to start where the cut sentence did'

    def test_the_last_window_of_a_take_keeps_everything(self):
        """Nothing was cut off there: the recording ends."""
        from ml_worker.engines.mlx_engine import trim_tail
        segments = [_said('Первое.', 0.0, 60.0), _said('И последнее.', 60.0, 120.0)]
        kept, resume = trim_tail(segments, start=480.0, end=600.0, total=600.0)

        assert kept == segments
        assert resume == 600.0

    def test_a_window_holding_one_long_segment_is_not_read_again(self):
        """Resuming at its start would re-read the whole window for no progress."""
        from ml_worker.engines.mlx_engine import trim_tail
        segments = [_said('Совсем короткое.', 0.0, 2.0),
                    _said('Длинный ответ без единой паузы…', 2.0, 120.0)]
        kept, resume = trim_tail(segments, start=0.0, end=120.0, total=600.0)

        assert kept == segments
        assert resume == 120.0

    def test_a_single_segment_window_advances_by_its_length(self):
        from ml_worker.engines.mlx_engine import trim_tail
        segments = [_said('Один сегмент.', 0.0, 120.0)]
        kept, resume = trim_tail(segments, start=120.0, end=240.0, total=600.0)

        assert kept == segments
        assert resume == 240.0


class TestSentenceRateCollapse:
    """The second way a window goes wrong. A runaway repeats itself and is caught
    by ``is_runaway``; this one keeps transcribing but stops punctuating, because
    the prompt it was handed broke off mid-sentence and it wrote in that style.
    Five stretches of one 88-minute interview, up to three windows long, came back
    that way."""

    def test_a_window_far_under_the_recording_is_flagged(self):
        from ml_worker.engines.mlx_engine import is_collapsed
        rates = [12.0, 13.0, 11.0]
        assert is_collapsed([_said('слово ' * 200)], rates)

    def test_a_window_in_line_with_the_recording_is_not(self):
        from ml_worker.engines.mlx_engine import is_collapsed, sentence_rate
        rates = [12.0, 13.0, 11.0]
        ordinary = [_said('Одно предложение. И второе. И третье.')]
        assert sentence_rate(ordinary) > 12.0
        assert not is_collapsed(ordinary, rates)

    def test_nothing_is_judged_before_there_is_a_median(self):
        from ml_worker.engines.mlx_engine import is_collapsed
        assert not is_collapsed([_said('слово ' * 200)], [12.0, 13.0])

    def test_a_recording_that_punctuates_sparsely_sets_its_own_median(self):
        """No number in the code says what normal is — the recording does."""
        from ml_worker.engines.mlx_engine import is_collapsed
        sparse = [2.0, 2.2, 1.9, 2.1]
        assert not is_collapsed([_said('слово ' * 100 + '.')], sparse)

    def test_an_empty_window_is_not_a_collapse(self):
        from ml_worker.engines.mlx_engine import is_collapsed, sentence_rate
        assert sentence_rate([]) is None
        assert not is_collapsed([], [12.0, 13.0, 11.0])

    def test_only_a_mark_that_ends_a_sentence_counts(self):
        from ml_worker.engines.mlx_engine import count_sentence_endings
        assert count_sentence_endings('Так. Нет? Понял!') == 3
        assert count_sentence_endings('Он спросил: «а зачем?» — Затем.') == 2
        assert count_sentence_endings('Сайты, чаты и т.д. и всё прочее') == 0
        assert count_sentence_endings('вот… значит так') == 0
        assert count_sentence_endings('Было в 2020. 2021 был хуже.') == 2

    def test_a_loop_of_full_stops_does_not_pass_for_punctuation(self):
        """The window the user saw: ninety seconds with no punctuation, then
        Whisper going round "и т.д." — which, counted mark by mark, read as five
        times better punctuated than the rest of the interview."""
        from ml_worker.engines.mlx_engine import is_collapsed
        # What healthy windows of a Russian interview read at.
        rates = [20.0, 21.0, 19.0]
        flat = _said('на телеграме потому что мы все-таки одно время до туда сложили '
                     'большое количество материала но после апреля поняли что')
        loop = _said('и .д. .д. т .д. и т .д. .д. и т т .д. .д. и и т .д. и т .д. и')
        assert is_collapsed([flat, loop], rates)


class TestRunawayDetector:
    """What tells a looping window from a person repeating themselves."""

    def test_a_line_repeated_past_the_limit_is_a_runaway(self):
        from ml_worker.engines.mlx_engine import is_runaway
        assert is_runaway([_said('Звук колокола.')] * 4)

    def test_someone_saying_da_three_times_is_not(self):
        from ml_worker.engines.mlx_engine import is_runaway
        assert not is_runaway([_said('Да.'), _said('да'), _said('Да!')])

    def test_a_window_that_reads_its_prompt_back_is_a_runaway(self):
        """The first step of the failure: the window stops hearing the audio and
        starts copying what it was handed."""
        from ml_worker.engines.mlx_engine import is_runaway
        assert is_runaway([_said('Звук колокола.'), _said('Звук колокола')],
                          prompt='…а потом звук колокола.')

    def test_a_segment_going_round_a_few_words_is_a_runaway(self):
        """No two lines match, so the repeat count never sees it."""
        from ml_worker.engines.mlx_engine import is_runaway
        assert is_runaway([
            _said('всю мою школу туда пока была идея сделать там 10-20 раз'),
            _said('и т.д. т.д. и т т.д. и т.д. и и т.д. т.д. и т.д.'),
            _said('т.д. и т.д. и .д. и т.д. и т.д.'),
        ])

    def test_someone_repeating_themselves_is_not(self):
        """The least varied 16 words in ~33,000 words of real transcripts."""
        from ml_worker.engines.mlx_engine import is_runaway
        assert not is_runaway([_said(
            'kind of to kind of drill that in yeah definitely yeah definitely '
            'yeah definitely yeah definitely')])
        assert not is_runaway([_said(
            'по поводу и чемпионата мира в России и чемпионата мира в Катаре '
            'и чемпионата мира в')])

    def test_one_short_answer_inside_a_long_prompt_is_not(self):
        from ml_worker.engines.mlx_engine import is_runaway
        assert not is_runaway([_said('Да.')],
                              prompt='Да, мы понимаем, что люди адаптируются.')

    def test_the_tail_handed_on_is_cut_on_a_word(self):
        from ml_worker.engines.mlx_engine import prompt_tail
        assert prompt_tail('раз два три четыре пять.', limit=11) == 'пять.'
        assert prompt_tail('раз два.', limit=10) == 'раз два.'

    def test_the_tail_stops_at_the_last_finished_sentence(self):
        """What the window heard last is half a sentence, and a prompt that breaks
        off teaches the next window to break off too."""
        from ml_worker.engines.mlx_engine import prompt_tail
        assert (prompt_tail('Мы ушли в создание нового сайта. Блокировок там')
                == 'Мы ушли в создание нового сайта.')
        assert (prompt_tail('Он спросил: «а зачем?» И я не ответил')
                == 'Он спросил: «а зачем?»')

    def test_a_window_with_no_finished_sentence_hands_on_nothing(self):
        from ml_worker.engines.mlx_engine import prompt_tail
        assert prompt_tail('и тогда он говорит что всё это') == ''


class TestAlienWords:
    """Words the decoder sampled rather than read. One Mac interview had ten of
    them in otherwise clean Russian; the same recording on Windows had none."""

    @pytest.mark.parametrize('word', [
        'terugивает', 'Никонаisme', 'песço', 'Actinghesiaọнальных'])
    def test_two_alphabets_in_one_word(self, word):
        from ml_worker.engines.mlx_engine import is_alien_word
        assert is_alien_word(word, 'ru')
        assert is_alien_word(word, 'en'), 'no language runs them together'

    @pytest.mark.parametrize('word', ['ọn', 'ọn坐', 'generatedți'])
    def test_a_third_alphabet_in_a_cyrillic_language(self, word):
        from ml_worker.engines.mlx_engine import is_alien_word
        assert is_alien_word(word, 'ru')

    @pytest.mark.parametrize('word', [
        'YouTube', 'FIFA.', 'Profi.ru', 'Prime', '«Динамо».', 'YouTube-канал',
        'SMS-ка', "iPhone'ы", 'Citroën', '2019', 'как-то', '—'])
    def test_what_russian_speech_is_full_of_passes(self, word):
        from ml_worker.engines.mlx_engine import is_alien_word
        assert not is_alien_word(word, 'ru')

    def test_other_languages_keep_their_own_letters(self):
        from ml_worker.engines.mlx_engine import is_alien_word
        assert not is_alien_word('Łódź', 'pl')
        assert not is_alien_word('mulțumesc', 'ro')

    def test_striking_them_out_leaves_the_sentence(self):
        from ml_worker.engines.mlx_engine import alien_words, strip_alien_words
        segments = [
            {'start': 0.0, 'end': 3.0, 'text': ' Кто я такой? ọn Сказать',
             'words': [{'word': ' Кто'}, {'word': ' я'}, {'word': ' такой?'},
                       {'word': ' ọn'}, {'word': ' Сказать'}]},
            {'start': 3.0, 'end': 4.0, 'text': ' jedem terugивает'},
        ]
        assert alien_words(segments, 'ru') == ['ọn', 'terugивает']

        kept = strip_alien_words(segments, 'ru')
        assert kept[0]['text'] == ' Кто я такой? Сказать'
        assert [w['word'] for w in kept[0]['words']] == [
            ' Кто', ' я', ' такой?', ' Сказать']
        assert kept[1]['text'] == 'jedem'

    def test_a_segment_left_empty_is_dropped(self):
        from ml_worker.engines.mlx_engine import strip_alien_words
        assert strip_alien_words([{'text': ' ọn坐'}], 'ru') == []


def _varied(stem, count):
    """``count`` different words, so the loop detector reads them as speech."""
    return ' '.join(f'{stem}{i}' for i in range(count))


def _words(text, start, step=0.5):
    """Word timings for ``text``, one word every ``step`` seconds from ``start``."""
    return [{'word': f' {w}', 'start': start + i * step,
             'end': start + i * step + step * 0.8}
            for i, w in enumerate(text.split())]


class TestFindHoles:
    """Speech the gate heard and no word covers. On the Mac transcript of one
    interview, three answers of 7 to 19 seconds were simply not there."""

    def test_a_skipped_passage_is_found(self):
        from ml_worker.engines.mlx_engine import find_holes
        segments = [{'start': 0.0, 'end': 10.0, 'words': _words('a ' * 20, 0.0)},
                    {'start': 25.0, 'end': 30.0, 'words': _words('b ' * 10, 25.0)}]
        holes = find_holes(segments, [(0.0, 30.0)], until=30.0)
        assert len(holes) == 1
        lo, hi = holes[0]
        assert lo == pytest.approx(9.9) and hi == 25.0

    def test_a_word_stretched_across_it_does_not_hide_it(self):
        """The word before a skip is often timed to end after it."""
        from ml_worker.engines.mlx_engine import find_holes
        segments = [{'start': 0.0, 'end': 20.0,
                     'words': [{'word': ' генерал', 'start': 1.0, 'end': 19.0},
                               {'word': ' КГБ,', 'start': 19.0, 'end': 20.0}]}]
        holes = find_holes(segments, [(0.0, 20.0)], until=20.0)
        assert holes == [(3.0, 19.0)]

    def test_a_pause_is_not_a_hole(self):
        from ml_worker.engines.mlx_engine import find_holes
        segments = [{'start': 0.0, 'end': 10.0, 'words': _words('a ' * 20, 0.0)},
                    {'start': 14.0, 'end': 20.0, 'words': _words('b ' * 12, 14.0)}]
        assert find_holes(segments, [(0.0, 20.0)], until=20.0) == []

    def test_quiet_the_gate_did_not_call_speech_is_not_a_hole(self):
        from ml_worker.engines.mlx_engine import find_holes
        segments = [{'start': 0.0, 'end': 5.0, 'words': _words('a ' * 10, 0.0)}]
        assert find_holes(segments, [(0.0, 5.0)], until=60.0) == []

    def test_nothing_past_the_part_being_kept(self):
        """The next window reads that part again anyway."""
        from ml_worker.engines.mlx_engine import find_holes
        segments = [{'start': 0.0, 'end': 5.0, 'words': _words('a ' * 10, 0.0)}]
        assert find_holes(segments, [(0.0, 60.0)], until=10.0) == []

    def test_a_segment_without_word_timings_covers_its_span(self):
        from ml_worker.engines.mlx_engine import find_holes
        assert find_holes([_said('Одно длинное.', 0.0, 30.0)], [(0.0, 30.0)],
                          until=30.0) == []

    def test_only_the_words_inside_are_kept(self):
        from ml_worker.engines.mlx_engine import words_inside
        segments = [{'start': 0.0, 'end': 4.0, 'text': ' раз два три четыре',
                     'words': _words('раз два три четыре', 0.0, step=1.0)}]
        kept = words_inside(segments, 1.0, 3.0)
        assert kept[0]['text'] == ' два три'
        assert kept[0]['start'] == 1.0 and kept[0]['end'] == pytest.approx(2.8)


class TestDecodeWindows:
    """The chain PINE holds in place of the one mlx-whisper would hold itself."""

    def _engine(self):
        from ml_worker.engines import mlx_engine
        engine = mlx_engine.MlxWhisperEngine(env=None)
        engine._model_path = 'fake-model'
        return engine

    def _run(self, monkeypatch, replies, windows, window_sec=1.0, rates=None):
        import sys

        import numpy as np
        from ml_worker.engines import mlx_engine
        fake = _FakeMlx(replies)
        monkeypatch.setitem(sys.modules, 'mlx_whisper', fake)
        monkeypatch.setattr(mlx_engine, 'MLX_PROMPT_WINDOW_SEC', window_sec)
        samples = np.zeros(int(windows * window_sec * 16000), dtype=np.float32)
        segments, lang = self._engine()._decode_windows(samples, rates=rates)
        return fake, segments, lang

    def test_each_window_is_seeded_with_the_last_one(self, monkeypatch):
        fake, segments, lang = self._run(
            monkeypatch,
            [{'segments': [_said(' Мы ушли в создание нового сайта.')], 'language': 'ru'},
             {'segments': [_said(' Блокировок там нет.')], 'language': 'ru'}],
            windows=2)

        assert lang == 'ru'
        assert fake.calls[0].get('initial_prompt') is None, 'nothing to carry yet'
        assert fake.calls[0]['condition_on_previous_text'] is True
        assert fake.calls[1]['initial_prompt'] == 'Мы ушли в создание нового сайта.'
        # Settled on the first window rather than re-detected on every one.
        assert fake.calls[1]['language'] == 'ru'
        assert [seg['start'] for seg in segments] == [0.0, 1.0]

    def test_a_looping_window_is_decoded_again_with_nothing_carried_in(self, monkeypatch):
        fake, segments, _ = self._run(
            monkeypatch,
            [{'segments': [_said(' Звук колокола.')], 'language': 'ru'},
             {'segments': [_said('Звук колокола.')] * 5, 'language': 'ru'},
             {'segments': [_said(' Люди адаптируются.')], 'language': 'ru'},
             {'segments': [_said(' Сайт работает без VPN.')], 'language': 'ru'}],
            windows=3)

        assert len(fake.calls) == 4, 'expected the middle window to be decoded twice'
        assert fake.calls[2].get('initial_prompt') is None
        assert fake.calls[2]['condition_on_previous_text'] is False
        # And the text that set the loop off is not handed to the next window.
        assert fake.calls[3].get('initial_prompt') is None
        assert [seg['text'].strip() for seg in segments] == [
            'Звук колокола.', 'Люди адаптируются.', 'Сайт работает без VPN.']

    def test_a_window_that_comes_back_empty_keeps_the_prompt(self, monkeypatch):
        fake, _, _ = self._run(
            monkeypatch,
            [{'segments': [_said(' Люди адаптируются.')], 'language': 'ru'},
             {'segments': [], 'language': 'ru'},
             {'segments': [_said(' Сайт работает.')], 'language': 'ru'}],
            windows=3)

        assert fake.calls[2]['initial_prompt'] == 'Люди адаптируются.'

    def test_a_window_that_stops_punctuating_is_decoded_again(self, monkeypatch):
        """The failure this was written for. The window is still transcribing, so
        ``is_runaway`` sees nothing wrong; what it has stopped doing is ending
        sentences, which is what the prompt it was handed did too."""
        ordinary = {'segments': [_said(' Одно. Второе. Третье. Четвёртое.')],
                    'language': 'ru'}
        fake, segments, _ = self._run(
            monkeypatch,
            [ordinary, ordinary, ordinary,
             {'segments': [_said(' и тогда он говорит что всё это было совсем '
                                 'не так как теперь рассказывают')],
              'language': 'ru'},
             {'segments': [_said(' И тогда он говорит. Что всё это было. Не так.')],
              'language': 'ru'},
             ordinary],
            windows=5)

        assert len(fake.calls) == 6, 'expected the fourth window to be decoded twice'
        assert fake.calls[3]['initial_prompt'] == 'Одно. Второе. Третье. Четвёртое.'
        # Read again without one, and with the memory it keeps of its own output.
        assert fake.calls[4].get('initial_prompt') is None
        assert fake.calls[4]['condition_on_previous_text'] is True
        # The second reading recovered, so its tail is fit to carry after all.
        assert (fake.calls[5]['initial_prompt']
                == 'И тогда он говорит. Что всё это было. Не так.')
        assert 'не так как теперь рассказывают' not in ' '.join(
            seg['text'] for seg in segments)

    _ORDINARY = ' Одно. Второе. Третье. Четвёртое.'
    _FLAT = ' и тогда он говорит что всё это было совсем не так как теперь рассказывают'
    _MENDED = ' И тогда он говорит. Что всё это было. Не так.'

    @staticmethod
    def _reply(text=None):
        return {'segments': [_said(text)] if text else [], 'language': 'ru'}

    def test_a_window_still_flat_without_its_prompt_is_read_piece_by_piece(
            self, monkeypatch):
        """mlx-whisper carries each 30s piece of a window into the next, so one
        piece that loses the punctuation hands the loss on inside the window
        whatever prompt the window was given."""
        r = self._reply
        fake, segments, _ = self._run(
            monkeypatch,
            [r(self._ORDINARY), r(self._ORDINARY), r(self._ORDINARY),
             r(self._FLAT), r(self._FLAT), r(self._MENDED), r(self._ORDINARY)],
            windows=5)

        assert len(fake.calls) == 7, 'expected the fourth window read three times'
        assert fake.calls[4].get('initial_prompt') is None
        assert fake.calls[4]['condition_on_previous_text'] is True
        assert fake.calls[5].get('initial_prompt') is None
        assert fake.calls[5]['condition_on_previous_text'] is False
        assert fake.calls[6]['initial_prompt'] == self._MENDED.strip()
        assert self._FLAT.strip() not in [seg['text'].strip() for seg in segments]

    def test_a_window_that_does_not_recover_hands_on_the_prompt_it_got(
            self, monkeypatch):
        """Dropping the prompt there sent the next window in with none — and
        with no prompt the check never ran again, so the rest of an interview
        came back unpunctuated and unread."""
        r = self._reply
        best = ' и тогда он говорит что всё это было совсем не так. Как теперь'
        fake, segments, _ = self._run(
            monkeypatch,
            [r(self._ORDINARY), r(self._ORDINARY), r(self._ORDINARY),
             r(self._FLAT), r(self._FLAT), r(best), r(self._ORDINARY)],
            windows=5)

        assert len(fake.calls) == 7
        assert fake.calls[6]['initial_prompt'] == self._ORDINARY.strip()
        # The reading that ended the most sentences is the one kept.
        assert [seg['text'].strip() for seg in segments][3] == best.strip()

    def test_a_window_with_no_prompt_is_still_judged(self, monkeypatch):
        """After a runaway there is no prompt to take away, so the window is read
        again the one way that differs: piece by piece."""
        r = self._reply
        fake, segments, _ = self._run(
            monkeypatch,
            [r(self._ORDINARY), r(self._ORDINARY), r(self._ORDINARY),
             {'segments': [_said('Звук колокола.')] * 5, 'language': 'ru'},
             r(self._ORDINARY),
             r(self._FLAT), r(self._MENDED)],
            windows=5)

        assert len(fake.calls) == 7
        assert fake.calls[5].get('initial_prompt') is None
        assert fake.calls[6].get('initial_prompt') is None
        assert fake.calls[6]['condition_on_previous_text'] is False
        assert [seg['text'].strip() for seg in segments][4] == self._MENDED.strip()

    def test_an_empty_reading_does_not_replace_a_flat_one(self, monkeypatch):
        """Words without punctuation are worth more than no words."""
        r = self._reply
        fake, segments, _ = self._run(
            monkeypatch,
            [r(self._ORDINARY), r(self._ORDINARY), r(self._ORDINARY),
             r(self._FLAT), r(), r()],
            windows=4)

        assert len(fake.calls) == 6
        assert [seg['text'].strip() for seg in segments][3] == self._FLAT.strip()

    def test_a_take_is_judged_against_the_rates_it_is_handed(self, monkeypatch):
        """Chunks share the recording's rates. Each chunk starting its own left
        the first three windows of every chunk unjudged."""
        rates = [125.0, 125.0, 125.0]
        fake, segments, _ = self._run(
            monkeypatch, [self._reply(self._FLAT), self._reply(self._MENDED)],
            windows=1, rates=rates)

        assert len(fake.calls) == 2, 'the first window of the take was not judged'
        assert fake.calls[1]['condition_on_previous_text'] is False
        assert segments[0]['text'].strip() == self._MENDED.strip()
        assert len(rates) == 4, 'the take adds its windows for the next one'

    _LOOP = ' и т.д. т.д. и т т.д. и т.д. и и т.д. т.д. и т.д. т.д. и т.д. и т.д. и'

    def test_a_window_that_trails_off_into_a_loop_is_read_again(self, monkeypatch):
        """The interview the user sent: unpunctuated speech, then Whisper going
        round "и т.д." until its fallback gave up."""
        r = self._reply
        fake, segments, _ = self._run(
            monkeypatch,
            [r(self._ORDINARY), r(self._ORDINARY), r(self._ORDINARY),
             {'segments': [_said(self._FLAT), _said(self._LOOP)], 'language': 'ru'},
             r(self._MENDED), r(self._ORDINARY)],
            windows=5)

        assert len(fake.calls) == 6
        assert fake.calls[4]['condition_on_previous_text'] is False
        # Nothing of the loop is handed on, and nothing of it is kept.
        assert fake.calls[5].get('initial_prompt') is None
        assert not any('т.д. т.д.' in seg['text'] for seg in segments)

    def test_a_reading_that_loops_is_no_recovery(self, monkeypatch):
        """Its full stops would have passed it for well punctuated."""
        r = self._reply
        fake, segments, _ = self._run(
            monkeypatch,
            [r(self._ORDINARY), r(self._ORDINARY), r(self._ORDINARY),
             r(self._FLAT),
             {'segments': [_said(self._FLAT), _said(self._LOOP)], 'language': 'ru'},
             r(self._MENDED), r(self._ORDINARY)],
            windows=5)

        assert len(fake.calls) == 7, 'the looping reading should not end the retries'
        assert [seg['text'].strip() for seg in segments][3] == self._MENDED.strip()

    def test_the_fallback_never_climbs_to_sampling(self, monkeypatch):
        """mlx-whisper's own ladder runs to 1.0, where it writes in any alphabet."""
        from ml_worker.constants import MLX_TEMPERATURES
        fake, _, _ = self._run(monkeypatch, [self._reply(self._ORDINARY)], windows=1)

        assert fake.calls[0]['temperature'] == MLX_TEMPERATURES
        assert max(MLX_TEMPERATURES) < 0.5

    def test_a_window_with_alien_words_is_read_again(self, monkeypatch):
        r = self._reply
        fake, segments, _ = self._run(
            monkeypatch,
            [r(' Кто я такой? ọn Сказать, что я выдвигаю, неправильно.'),
             r(' Кто я такой? Сказать, что я выдвигаю кандидата, неправильно.'),
             r(self._ORDINARY)],
            windows=2)

        assert len(fake.calls) == 3
        assert fake.calls[1].get('initial_prompt') is None
        assert fake.calls[1]['condition_on_previous_text'] is False
        assert segments[0]['text'].strip() == (
            'Кто я такой? Сказать, что я выдвигаю кандидата, неправильно.')
        # The clean reading is fit to carry.
        assert fake.calls[2]['initial_prompt'] == segments[0]['text'].strip()

    def test_alien_words_a_second_reading_keeps_are_struck_out(self, monkeypatch):
        r = self._reply
        fake, segments, _ = self._run(
            monkeypatch,
            [r(' Только меньше считаете jedem ọn坐 се'),
             r(' Только ọn坐 меньше terugивает'),
             r(self._ORDINARY)],
            windows=2)

        assert len(fake.calls) == 3
        assert segments[0]['text'] == 'Только меньше считаете jedem се'

    def test_a_looping_window_is_not_read_a_third_time_for_its_letters(
            self, monkeypatch):
        """It was already read with nothing carried in."""
        fake, segments, _ = self._run(
            monkeypatch,
            [{'segments': [_said('Звук колокола.')] * 5, 'language': 'ru'},
             self._reply(' Люди ọn адаптируются.')],
            windows=1)

        assert len(fake.calls) == 2
        assert segments[0]['text'] == 'Люди адаптируются.'

    def _run_voice(self, monkeypatch, replies, secs):
        import sys
        fake = _FakeMlx(replies)
        monkeypatch.setitem(sys.modules, 'mlx_whisper', fake)
        segments, _ = self._engine()._decode_windows(_voice(secs), language='ru')
        return fake, segments

    def test_speech_the_decoder_skipped_is_read_on_its_own(self, monkeypatch):
        """What happened to 19 seconds of an answer on the Mac: the text runs
        to "у которого я брал", and picks up again at the next question."""
        # Varied words: twenty of one word in a row is a loop, not speech.
        first, second = _varied('раз', 20), _varied('два', 30)
        before = {'start': 0.0, 'end': 10.0, 'text': first,
                  'words': _words(first, 0.0)}
        after = {'start': 25.0, 'end': 40.0, 'text': second,
                 'words': _words(second, 25.0)}
        # The hole runs 9.9-25s and is read from 8.9s, a second either side.
        hole = {'start': 0.0, 'end': 17.0, 'text': ' лишнее нужное тоже лишнее',
                'words': [{'word': ' лишнее', 'start': 0.1, 'end': 0.6},
                          {'word': ' нужное', 'start': 3.0, 'end': 3.5},
                          {'word': ' тоже', 'start': 12.0, 'end': 12.5},
                          {'word': ' лишнее', 'start': 16.5, 'end': 16.9}]}
        fake, segments = self._run_voice(
            monkeypatch,
            [{'segments': [before, after], 'language': 'ru'},
             {'segments': [hole], 'language': 'ru'}],
            secs=40)

        assert len(fake.calls) == 2, 'the hole should be read once, on its own'
        assert fake.calls[1].get('initial_prompt') is None
        assert fake.calls[1]['condition_on_previous_text'] is False
        texts = [seg['text'].strip() for seg in segments]
        assert texts[1] == 'нужное тоже', 'only the words inside the hole'
        assert segments[1]['start'] == pytest.approx(8.9 + 3.0, abs=0.05)
        assert [seg['start'] for seg in segments] == sorted(
            seg['start'] for seg in segments)

    def test_speech_every_word_covers_costs_nothing(self, monkeypatch):
        text = _varied('слово', 80)
        covered = {'start': 0.0, 'end': 40.0, 'text': text,
                   'words': _words(text, 0.0)}
        fake, segments = self._run_voice(
            monkeypatch, [{'segments': [covered], 'language': 'ru'}], secs=40)

        assert len(fake.calls) == 1
        assert len(segments) == 1


class TestMlxLanguageProbeWindow:
    """Where the language probe listens. Reading the first 30 s of a phone call
    means reading ringback, which is how a Russian interview came back as English
    at p=0.29 — and stopped the pipeline to ask the user about it."""

    def _engine_over(self, monkeypatch, audio):
        from ml_worker.engines import mlx_engine
        monkeypatch.setattr(mlx_engine, 'load_audio_range',
                            lambda path, offset, duration: audio)
        return mlx_engine.MlxWhisperEngine(env=None)

    def test_skips_the_ringback_and_listens_to_the_speech(self, monkeypatch):
        import numpy as np
        from ml_worker.constants import MLX_VAD_MIN_FLATNESS
        from ml_worker.engines.mlx_engine import spectral_flatness
        audio = np.concatenate([_tone(30, on=1.0, off=4.0), _voice(60)])
        engine = self._engine_over(monkeypatch, audio)

        window = engine.probe_window('call.m4a', 30.0, total_duration=90.0)

        assert len(window) == 30 * 16000
        assert spectral_flatness(window) > MLX_VAD_MIN_FLATNESS * 10, \
            'the probe is still listening to the tone'

    def test_falls_back_to_the_head_when_it_finds_no_speech(self, monkeypatch):
        import numpy as np
        audio = np.zeros(90 * 16000, dtype=np.float32)
        engine = self._engine_over(monkeypatch, audio)

        window = engine.probe_window('silence.wav', 30.0, total_duration=90.0)

        assert len(window) == 30 * 16000

    def test_speech_near_the_end_still_gets_a_full_window(self, monkeypatch):
        """Pulled back from the end rather than handed the two seconds left."""
        import numpy as np
        audio = np.concatenate([_tone(70, on=1.0, off=4.0), _voice(20)])
        engine = self._engine_over(monkeypatch, audio)

        window = engine.probe_window('call.m4a', 30.0, total_duration=90.0)

        assert len(window) == 30 * 16000


class TestCleanTranscriptSegments:
    def test_drops_empty_and_zero_length_artifacts(self):
        from ml_worker.pipeline import clean_transcript_segments
        segs = [
            {
                'start': 1.0,
                'end': 2.0,
                'text': ' Hello world ',
                'speaker': 'A',
                'words': [
                    {'word': ' Hello', 'start': 1.0, 'end': 1.4},
                    {'word': 'world ', 'start': 1.4, 'end': 2.0},
                ],
            },
            {
                'start': 2.0,
                'end': 2.0,
                'text': '',
                'speaker': '',
                'words': [],
            },
            {
                'start': 3.0,
                'end': 4.0,
                'text': 'Broken',
                'speaker': 'B',
                'words': [
                    {'word': 'Broken', 'start': 3.5, 'end': 3.5},
                ],
            },
        ]
        out = clean_transcript_segments(segs)
        assert out == [{
            'start': 1.0,
            'end': 2.0,
            'text': 'Hello world',
            'speaker': 'A',
            'words': [
                {'word': 'Hello', 'start': 1.0, 'end': 1.4},
                {'word': 'world', 'start': 1.4, 'end': 2.0},
            ],
        }]

    @staticmethod
    def _seg(text, start, end, speaker='Participant 1'):
        words = text.split()
        step = (end - start) / len(words)
        return {
            'start': start, 'end': end, 'text': text, 'speaker': speaker,
            'words': [{'word': w, 'start': start + i * step,
                       'end': start + (i + 1) * step}
                      for i, w in enumerate(words)],
        }

    def test_drops_the_credits_whisper_wrote_over_silence(self):
        # The three found in one real whisperx interview.
        from ml_worker.pipeline import clean_transcript_segments
        segs = [
            self._seg('Продолжение следует.', 308.962, 329.515),
            self._seg('Продолжение следует...', 794.759, 802.015),
            self._seg('Субтитры создавал DimaTorzok', 1988.092, 1989.307,
                      speaker=''),
        ]
        assert clean_transcript_segments(segs) == []

    def test_a_credit_is_dropped_whoever_it_is_attributed_to(self):
        from ml_worker.pipeline import clean_transcript_segments
        segs = [self._seg('Редактор субтитров А.Семкин', 10.0, 11.0)]
        assert clean_transcript_segments(segs) == []

    def test_keeps_a_common_phrase_said_at_normal_pace(self):
        from ml_worker.pipeline import clean_transcript_segments
        segs = [self._seg('Продолжение следует.', 10.0, 11.0)]
        assert len(clean_transcript_segments(segs)) == 1

    def test_keeps_a_phrase_quoted_inside_a_longer_answer(self):
        from ml_worker.pipeline import clean_transcript_segments
        segs = [
            self._seg('А в конце было написано субтитры создавал кто-то', 10.0, 14.0),
            self._seg('Ну и продолжение следует, как говорится', 20.0, 50.0),
        ]
        assert len(clean_transcript_segments(segs)) == 2

    def test_keeps_ordinary_speech_with_no_speaker(self):
        from ml_worker.pipeline import clean_transcript_segments
        segs = [self._seg('Да, я согласен.', 10.0, 11.0, speaker='')]
        assert len(clean_transcript_segments(segs)) == 1


# ---------------------------------------------------------------------------
# 1. assign_speakers_simple
# ---------------------------------------------------------------------------

class TestJoinWords:
    """Whisper spells "как-то" as two word tokens and tells them apart by the
    leading space on the first. The mlx engine strips that space so the recording
    UI can find each word in the segment text, and the plain join that used to
    rebuild the text here then printed "как -то" into the saved transcript."""

    def test_a_hyphenated_word_survives_the_join(self):
        from ml_worker.pipeline import join_words
        assert join_words(['как', '-то']) == 'как-то'

    def test_an_ordinary_word_list_joins_as_it_always_did(self):
        from ml_worker.pipeline import join_words
        assert join_words(['мы', 'ушли', 'в', 'новый', 'сайт.']) == 'мы ушли в новый сайт.'

    def test_a_dash_standing_on_its_own_stays_its_own_word(self):
        from ml_worker.pipeline import join_words
        assert join_words(['и', '-', 'да']) == 'и - да'

    def test_the_saved_transcript_spells_it_as_one_word(self):
        from ml_worker.pipeline import clean_transcript_segments
        segs = [{
            'start': 0.0, 'end': 1.0, 'speaker': 'A', 'text': 'как-то',
            'words': [{'word': 'как', 'start': 0.0, 'end': 0.5},
                      {'word': '-то', 'start': 0.5, 'end': 1.0}],
        }]
        assert clean_transcript_segments(segs)[0]['text'] == 'как-то'


class TestAssignSpeakersSimple:
    """Tests for ml_worker.diarize.assign_speakers_simple."""

    def test_basic_speaker_assignment(self):
        from ml_worker.diarize import assign_speakers_simple
        diarization = FakeDiarization([
            (0.0, 5.0, 'SPEAKER_00'),
            (5.0, 10.0, 'SPEAKER_01'),
        ])
        segments = [
            {'start': 0.5, 'end': 4.5, 'text': 'Hello there',
             'words': [{'start': 0.5, 'end': 2.0}, {'start': 2.0, 'end': 4.5}]},
            {'start': 5.5, 'end': 9.0, 'text': 'Hi back',
             'words': [{'start': 5.5, 'end': 7.0}, {'start': 7.0, 'end': 9.0}]},
        ]
        result = assign_speakers_simple(diarization, segments)
        assert result[0]['speaker'] == 'SPEAKER_00'
        assert result[1]['speaker'] == 'SPEAKER_01'

    def test_a_segment_no_turn_covers_goes_to_the_nearest_speaker(self):
        """Left blank, it exported as a bare "Speaker" between two named people."""
        from ml_worker.diarize import assign_speakers_simple
        diarization = FakeDiarization([
            (0.0, 2.0, 'SPEAKER_00'),
            (20.0, 30.0, 'SPEAKER_01'),
        ])
        segments = [
            {'start': 10.0, 'end': 15.0, 'text': 'Between turns',
             'words': [{'start': 10.0, 'end': 15.0}]},
        ]
        result = assign_speakers_simple(diarization, segments)
        assert result[0]['speaker'] == 'SPEAKER_01', '5s away beats 8s away'

    def test_a_one_word_answer_goes_to_the_short_turn_it_was(self):
        """"Нет." is about as long as the turns the noise filter drops."""
        from ml_worker.diarize import assign_speakers_simple
        diarization = FakeDiarization([
            (0.0, 5.0, 'MOD'),
            (5.1, 5.35, 'RESP'),
            (7.0, 12.0, 'MOD'),
        ])
        segments = [
            {'start': 5.1, 'end': 5.4, 'text': 'Нет.',
             'words': [{'start': 5.1, 'end': 5.4}]},
        ]
        result = assign_speakers_simple(diarization, segments)
        assert result[0]['speaker'] == 'RESP'

    def test_majority_vote_across_words(self):
        from ml_worker.diarize import assign_speakers_simple
        # Speaker A covers 0-6s, Speaker B covers 6-10s
        diarization = FakeDiarization([
            (0.0, 6.0, 'A'),
            (6.0, 10.0, 'B'),
        ])
        # Segment spans both, but more words fall in A's range
        segments = [
            {'start': 0.0, 'end': 10.0, 'text': 'Long sentence',
             'words': [
                 {'start': 0.0, 'end': 2.0},   # A
                 {'start': 2.0, 'end': 4.0},   # A
                 {'start': 4.0, 'end': 5.5},   # A
                 {'start': 7.0, 'end': 9.0},   # B
             ]},
        ]
        result = assign_speakers_simple(diarization, segments)
        assert result[0]['speaker'] == 'A'  # majority

    def test_three_speakers(self):
        from ml_worker.diarize import assign_speakers_simple
        diarization = FakeDiarization([
            (0.0, 3.0, 'S1'),
            (3.0, 6.0, 'S2'),
            (6.0, 9.0, 'S3'),
        ])
        segments = [
            {'start': 0.5, 'end': 2.5, 'text': 'One', 'words': [{'start': 0.5, 'end': 2.5}]},
            {'start': 3.5, 'end': 5.5, 'text': 'Two', 'words': [{'start': 3.5, 'end': 5.5}]},
            {'start': 6.5, 'end': 8.5, 'text': 'Three', 'words': [{'start': 6.5, 'end': 8.5}]},
        ]
        result = assign_speakers_simple(diarization, segments)
        assert result[0]['speaker'] == 'S1'
        assert result[1]['speaker'] == 'S2'
        assert result[2]['speaker'] == 'S3'

    def test_no_words_uses_segment_times(self):
        """Segments without words should use segment start/end for speaker lookup."""
        from ml_worker.diarize import assign_speakers_simple
        diarization = FakeDiarization([(0.0, 5.0, 'SOLO')])
        segments = [{'start': 1.0, 'end': 3.0, 'text': 'No words key'}]
        result = assign_speakers_simple(diarization, segments)
        assert result[0]['speaker'] == 'SOLO'

    def test_empty_diarization(self):
        from ml_worker.diarize import assign_speakers_simple
        diarization = FakeDiarization([])
        segments = [
            {'start': 0.0, 'end': 5.0, 'text': 'Hello',
             'words': [{'start': 0.0, 'end': 5.0}]},
        ]
        result = assign_speakers_simple(diarization, segments)
        # Empty diarization → speaker key either absent or empty
        assert result[0].get('speaker', '') == ''

    # Host and guest talk in turn; a third label holds 5s of the recording,
    # between two of the guest's turns.
    _TURNS = [(0.0, 50.0, 'HOST'), (50.0, 100.0, 'GUEST'),
              (100.0, 105.0, 'THIRD'), (105.0, 150.0, 'GUEST'),
              (150.0, 200.0, 'HOST')]
    # The host and the guest 0.66 alike, as on the interview this was measured on.
    _HOST, _GUEST = [1.0, 0.0, 0.0], [0.66, 0.7513, 0.0]

    @staticmethod
    def _third_voice_segment():
        return [{'start': 101.0, 'end': 104.0, 'text': 'Здесь масса причин.',
                 'words': [{'start': 101.0, 'end': 104.0}]}]

    def test_a_third_voice_is_kept_apart(self):
        """An advert read by someone else, 0.35 alike the nearest main speaker,
        was written under the guest's name."""
        from ml_worker.diarize import NativeDiarization, assign_speakers_simple
        third = [0.2, 0.3, 0.93]
        diarization = NativeDiarization(
            FakeDiarization(self._TURNS),
            {'HOST': self._HOST, 'GUEST': self._GUEST, 'THIRD': third})
        result = assign_speakers_simple(diarization, self._third_voice_segment())
        assert result[0]['speaker'] == 'THIRD'

    def test_a_split_off_voice_goes_to_whoever_it_sounds_like(self):
        """Not to whoever talks around it: that is the guest, the voice is the host's."""
        from ml_worker.diarize import NativeDiarization, assign_speakers_simple
        like_host = [0.95, 0.1, 0.29]
        diarization = NativeDiarization(
            FakeDiarization(self._TURNS),
            {'HOST': self._HOST, 'GUEST': self._GUEST, 'THIRD': like_host})
        result = assign_speakers_simple(diarization, self._third_voice_segment())
        assert result[0]['speaker'] == 'HOST'

    def test_without_voices_a_minor_speaker_goes_to_its_neighbours(self):
        from ml_worker.diarize import assign_speakers_simple
        result = assign_speakers_simple(FakeDiarization(self._TURNS),
                                        self._third_voice_segment())
        assert result[0]['speaker'] == 'GUEST'

    def test_voices_are_read_in_the_order_of_the_labels(self):
        from ml_worker.diarize import speaker_centroids

        class Labels:
            def labels(self):
                return ['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02']

        class Output:
            speaker_diarization = Labels()
            speaker_embeddings = [[3.0, 4.0], [0.0, 2.0], [0.0, 0.0]]

        centroids = speaker_centroids(Output())
        assert centroids['SPEAKER_00'] == pytest.approx([0.6, 0.8])
        assert centroids['SPEAKER_01'] == pytest.approx([0.0, 1.0])
        assert 'SPEAKER_02' not in centroids, 'a zero row pads, it is no voice'

    def test_an_output_without_voices_has_no_centroids(self):
        from ml_worker.diarize import speaker_centroids
        assert speaker_centroids(FakeDiarization([(0.0, 1.0, 'A')])) is None


# ---------------------------------------------------------------------------
# 2. map_speakers
# ---------------------------------------------------------------------------

class TestMapSpeakers:
    """Tests for ml_worker.pipeline.map_speakers."""

    def test_basic_renaming(self):
        from ml_worker.pipeline import map_speakers
        segments = [
            {'speaker': 'SPEAKER_00', 'text': 'Hi'},
            {'speaker': 'SPEAKER_01', 'text': 'Hello'},
            {'speaker': 'SPEAKER_00', 'text': 'Question'},
        ]
        mapping = map_speakers(segments)
        assert mapping['SPEAKER_00'] == 'Moderator'
        assert mapping['SPEAKER_01'] == 'Participant 1'
        # Verify segments were updated in-place
        assert segments[0]['speaker'] == 'Moderator'
        assert segments[1]['speaker'] == 'Participant 1'
        assert segments[2]['speaker'] == 'Moderator'

    def test_empty_segments(self):
        from ml_worker.pipeline import map_speakers
        mapping = map_speakers([])
        assert mapping == {}

    def test_no_speaker_key(self):
        from ml_worker.pipeline import map_speakers
        segments = [{'text': 'No speaker'}]
        mapping = map_speakers(segments)
        assert mapping == {}

    def test_many_speakers(self):
        """More than 6 speakers should get generic 'Speaker N' labels."""
        from ml_worker.pipeline import map_speakers
        segments = [{'speaker': f'SPEAKER_{i:02d}', 'text': f's{i}'} for i in range(8)]
        mapping = map_speakers(segments)
        assert mapping['SPEAKER_00'] == 'Moderator'
        assert mapping['SPEAKER_05'] == 'Participant 5'
        assert mapping['SPEAKER_06'] == 'Speaker 7'
        assert mapping['SPEAKER_07'] == 'Speaker 8'


# ---------------------------------------------------------------------------
# 3. Chunk decision constants (re-exported for the API layer and tests)
# ---------------------------------------------------------------------------

class TestChunkConstants:
    """Verify chunking threshold and overlap constants."""

    def test_chunk_threshold(self):
        from app.services.transcription import CHUNK_THRESHOLD_SEC
        assert CHUNK_THRESHOLD_SEC == 1800  # 30 min

    def test_chunk_size(self):
        from app.services.transcription import CHUNK_SIZE_SEC
        assert CHUNK_SIZE_SEC == 1800

    def test_chunk_overlap(self):
        from app.services.transcription import CHUNK_OVERLAP_SEC
        assert CHUNK_OVERLAP_SEC == 30


class TestAlignModelFailureCaching:
    """Ensure failed align loads are memoized to avoid repeated retries."""

    @staticmethod
    def _make_engine():
        from ml_worker.engines.whisperx_engine import WhisperXEngine
        return WhisperXEngine(env=None)

    def test_failed_align_model_is_not_retried(self):
        eng = self._make_engine()

        class FakeWhisperX:
            calls = 0

            @staticmethod
            def load_align_model(language_code, device, model_name=None):
                FakeWhisperX.calls += 1
                raise RuntimeError('align download failed')

        with pytest.raises(RuntimeError, match='align download failed'):
            eng._get_align_model(FakeWhisperX, 'ru', 'cpu')

        with pytest.raises(RuntimeError, match='previously failed'):
            eng._get_align_model(FakeWhisperX, 'ru', 'cpu')

        # First attempt exhausts both configured Russian candidates; second attempt is blocked by cache.
        assert FakeWhisperX.calls == 2

    def test_align_model_fallback_tries_next_candidate(self):
        eng = self._make_engine()

        class FakeWhisperX:
            calls = []

            @staticmethod
            def load_align_model(language_code, device, model_name=None):
                FakeWhisperX.calls.append(model_name)
                if model_name == 'anton-l/wav2vec2-large-xlsr-53-russian':
                    raise RuntimeError('first model failed')
                return object(), {'language': language_code, 'type': 'huggingface', 'dictionary': {}}

        model, metadata = eng._get_align_model(FakeWhisperX, 'ru', 'cpu')
        assert model is not None
        assert metadata['language'] == 'ru'
        assert FakeWhisperX.calls == [
            'anton-l/wav2vec2-large-xlsr-53-russian',
            'jonatasgrosman/wav2vec2-large-xlsr-53-russian',
        ]
        assert ('ru', 'cpu') not in eng._failed_align_keys


class TestAlignModelParking:
    """The cached align model waits in RAM while Whisper decodes.

    Measured 2026-11-18 on a 12 GB card: an 88-min recording's first chunk
    decoded in 101s, and the next two — with the ~1.4 GB align model still on
    the card — took 851s and 836s while Windows paged VRAM to system memory.
    """

    class FakeAlignModel:
        def __init__(self, device):
            self.device = device
            self.moves = []

        def to(self, device):
            self.device = str(device)
            self.moves.append(self.device)
            return self

    @classmethod
    def _harness(cls, device):
        """Engine on `device` plus a fake whisperx that records where the align model sits."""
        from ml_worker.engines.whisperx_engine import WhisperXEngine

        eng = WhisperXEngine(env=None)
        eng._device = device
        seen = {'decode': [], 'align': []}

        def where():
            model = eng._cached_align_model
            return None if model is None else model.device

        class FakeWhisperModel:
            @staticmethod
            def transcribe(audio, **kwargs):
                seen['decode'].append(where())
                return {'segments': [{'start': 1.0, 'end': 2.0, 'text': 'привет'}],
                        'language': 'ru'}

        eng._model = FakeWhisperModel()

        class FakeAudio:
            shape = (16000 * 60,)

        class FakeWhisperX:
            @staticmethod
            def load_audio(path):
                return FakeAudio()

            @staticmethod
            def load_align_model(language_code, device, model_name=None):
                return cls.FakeAlignModel(device), {'language': language_code}

            @staticmethod
            def align(segments, model, metadata, audio, device, return_char_alignments=False):
                seen['align'].append(model.device)
                return {'segments': segments}

        fake_torch = MagicMock()
        patches = [
            patch.dict('sys.modules', {'whisperx': FakeWhisperX, 'torch': fake_torch}),
            patch('ml_worker.engines.whisperx_engine.free_vram_gb', return_value=8.0),
            patch('ml_worker.engines.whisperx_engine.load_audio_range',
                  return_value=FakeAudio()),
        ]
        return eng, seen, patches

    def test_chunked_decode_runs_with_align_model_in_ram(self):
        from ml_worker.engines.base import TranscribeContext

        eng, seen, patches = self._harness('cuda')
        with patches[0], patches[1], patches[2]:
            eng.transcribe('rec.wav', 2000, None, TranscribeContext())

        # Chunk 1 decodes before the align model exists; chunk 2 must not share the card with it.
        assert seen['decode'] == [None, 'cpu']
        assert seen['align'] == ['cuda', 'cuda']
        # Parked again afterwards, so diarization and the next job get the memory too.
        assert eng._cached_align_model.device == 'cpu'

    def test_next_recording_decodes_with_align_model_in_ram(self):
        from ml_worker.engines.base import TranscribeContext

        eng, seen, patches = self._harness('cuda')
        with patches[0], patches[1], patches[2]:
            eng.transcribe('first.wav', 60, None, TranscribeContext())
            eng.transcribe('second.wav', 60, None, TranscribeContext())

        assert seen['decode'] == [None, 'cpu']
        assert seen['align'] == ['cuda', 'cuda']
        assert eng._cached_align_model.device == 'cpu'

    def test_cpu_engine_never_moves_the_align_model(self):
        from ml_worker.engines.base import TranscribeContext

        eng, seen, patches = self._harness('cpu')
        with patches[0], patches[1], patches[2]:
            eng.transcribe('rec.wav', 2000, None, TranscribeContext())

        assert seen['align'] == ['cpu', 'cpu']
        assert eng._cached_align_model.moves == []


# ---------------------------------------------------------------------------
# 4. Cancellation
# ---------------------------------------------------------------------------

class TestCancellation:
    """Tests for cancellation signal machinery."""

    def test_check_cancel_raises_when_set(self):
        from app.services.transcription import (
            TranscriptionCancelled,
            _check_cancel,
            _register_cancel,
            _unregister_cancel,
        )
        ev = _register_cancel(99990)
        ev.set()
        with pytest.raises(TranscriptionCancelled):
            _check_cancel(99990)
        _unregister_cancel(99990)

    def test_check_cancel_noop_when_not_set(self):
        from app.services.transcription import (
            _check_cancel,
            _register_cancel,
            _unregister_cancel,
        )
        _register_cancel(99991)
        # Should NOT raise
        _check_cancel(99991)
        _unregister_cancel(99991)

    def test_cancel_transcription_returns_true_when_active(self):
        from app.services.transcription import (
            _register_cancel,
            _unregister_cancel,
            cancel_transcription,
        )
        _register_cancel(99992)
        assert cancel_transcription(99992) is True
        _unregister_cancel(99992)

    def test_cancel_transcription_returns_false_when_inactive(self):
        from app.services.transcription import cancel_transcription
        assert cancel_transcription(99993) is False

    def test_cancel_transcription_wakes_language_waiter(self):
        import threading

        from app.services.transcription import (
            _language_waiters,
            _language_waiters_lock,
            cancel_transcription,
        )
        rid = 99994
        ev = threading.Event()
        with _language_waiters_lock:
            _language_waiters[rid] = {'event': ev, 'language': None, 'cancelled': False}
        assert cancel_transcription(rid) is True
        with _language_waiters_lock:
            st = _language_waiters.pop(rid, None)
        assert st['cancelled'] is True
        assert ev.is_set()


# ---------------------------------------------------------------------------
# 5. Requeue interrupted
# ---------------------------------------------------------------------------

class TestRequeueInterrupted:
    """Tests for requeue_interrupted() on startup."""

    def test_stuck_recordings_requeued(self, app):
        with app.app_context():
            from app.extensions import db
            from app.models.project import Project
            from app.models.recording import Recording

            proj = Project(name='Requeue Test', folder_name='rq_test')
            db.session.add(proj)
            db.session.flush()

            # Create recordings stuck in different states
            stuck1 = Recording(project_id=proj.id, original_name='s1.mp3',
                               stored_name='s1.mp3', transcription_status='transcribing')
            stuck2 = Recording(project_id=proj.id, original_name='s2.mp3',
                               stored_name='s2.mp3', transcription_status='transcribing')
            stuck3 = Recording(project_id=proj.id, original_name='s3.mp3',
                               stored_name='s3.mp3', transcription_status='awaiting_language')
            ok = Recording(project_id=proj.id, original_name='ok.mp3',
                           stored_name='ok.mp3', transcription_status='transcribed')
            db.session.add_all([stuck1, stuck2, stuck3, ok])
            db.session.commit()
            s1id, s2id, s3id, okid = stuck1.id, stuck2.id, stuck3.id, ok.id

        with patch('app.services.transcription.enqueue') as mock_enqueue:
            from app.services.transcription import requeue_interrupted
            requeue_interrupted(app)

        # Stuck recordings should be set to pending and enqueued
        with app.app_context():
            from app.models.recording import Recording as Rec
            assert db.session.get(Rec, s1id).transcription_status == 'pending'
            assert db.session.get(Rec, s2id).transcription_status == 'pending'
            assert db.session.get(Rec, s3id).transcription_status == 'pending'
            assert db.session.get(Rec, okid).transcription_status == 'transcribed'
            assert mock_enqueue.call_count == 3

    def test_no_stuck_recordings(self, app):
        """No stuck recordings → no enqueue calls."""
        with patch('app.services.transcription.enqueue') as mock_enqueue:
            from app.services.transcription import requeue_interrupted
            requeue_interrupted(app)
            assert mock_enqueue.call_count == 0


# ---------------------------------------------------------------------------
# 6. Diarization MPS-to-CPU fallback
# ---------------------------------------------------------------------------

class TestDiarizeCPUFallback:
    """Tests for Diarizer.run MPS-to-CPU fallback retry (Mac native pyannote path)."""

    @needs_torch
    def test_mps_fallback_to_cpu(self):
        """If diarization fails on MPS, retry on CPU and assign speakers."""
        annotation = FakeDiarization([
            (0.0, 5.0, 'SPEAKER_00'),
            (5.0, 10.0, 'SPEAKER_01'),
        ])

        class FakePipeline:
            def __init__(self):
                self.model = MagicMock()
                self.to = MagicMock(return_value=self)
                self._n = 0

            def __call__(self, audio_input, **kwargs):
                self._n += 1
                if self._n == 1:
                    raise RuntimeError('MPS op not supported')
                return annotation

        d = _make_diarizer(engine_kind='mlx', device='mps', pipeline=FakePipeline())

        segments = [
            {'start': 0.5, 'end': 4.5, 'text': 'Hello',
             'words': [{'start': 0.5, 'end': 4.5}]},
            {'start': 5.5, 'end': 9.0, 'text': 'Hi',
             'words': [{'start': 5.5, 'end': 9.0}]},
        ]

        dummy_audio = {"waveform": __import__('torch').zeros(1, 16000), "sample_rate": 16000}
        result = d.run(dummy_audio, segments, 1)

        assert d.device == 'cpu'
        assert d.pipeline.to.call_count == 1
        assert result[0]['speaker'] == 'SPEAKER_00'
        assert result[1]['speaker'] == 'SPEAKER_01'

    @needs_torch
    def test_cpu_failure_no_retry(self):
        """If already on CPU, failure should propagate (no infinite retry)."""

        class FailPipe:
            model = MagicMock()

            def to(self, device):
                return self

            def __call__(self, audio_input, **kwargs):
                raise RuntimeError('Diarization broken')

        d = _make_diarizer(engine_kind='mlx', device='cpu', pipeline=FailPipe())

        dummy_audio = {"waveform": __import__('torch').zeros(1, 16000), "sample_rate": 16000}
        with pytest.raises(RuntimeError, match='Diarization broken'):
            d.run(dummy_audio, [], 1)

    @needs_torch
    def test_mps_success_no_fallback(self):
        """If MPS works, no fallback to CPU should happen."""
        annotation = FakeDiarization([
            (0.0, 5.0, 'SPEAKER_00'),
        ])

        class OkPipe:
            def __init__(self):
                self.model = MagicMock()
                self.to = MagicMock(return_value=self)

            def __call__(self, audio_input, **kwargs):
                return annotation

        d = _make_diarizer(engine_kind='mlx', device='mps', pipeline=OkPipe())

        segments = [
            {'start': 0.5, 'end': 4.5, 'text': 'Hello',
             'words': [{'start': 0.5, 'end': 4.5}]},
        ]

        dummy_audio = {"waveform": __import__('torch').zeros(1, 16000), "sample_rate": 16000}
        # Device is faked as 'mps' on a non-Mac host; stub the MPS-only cache clear.
        with patch('torch.mps.empty_cache'):
            result = d.run(dummy_audio, segments, 1)

        assert d.device == 'mps'
        d.pipeline.to.assert_not_called()
        assert result[0]['speaker'] == 'SPEAKER_00'


class TestStubTorchcodec:
    """Tests for compat.stub_torchcodec sys.modules injection."""

    def _clean(self):
        import sys
        for key in list(sys.modules):
            if key == 'torchcodec' or key.startswith('torchcodec.'):
                del sys.modules[key]

    def test_stub_replaces_real_module(self):
        from ml_worker import compat
        self._clean()
        compat.stub_torchcodec()
        import sys
        assert getattr(sys.modules.get('torchcodec'), '_pine_stub', False)
        assert getattr(sys.modules.get('torchcodec.decoders'), '_pine_stub', False)
        assert getattr(sys.modules.get('torchcodec._core.ops'), '_pine_stub', False)
        self._clean()

    def test_stub_allows_importlib_find_spec(self):
        """Python 3.13+ raises ValueError if torchcodec is in sys.modules but __spec__ is None
        (transformers checks this at import time)."""
        import importlib.util

        from ml_worker import compat

        self._clean()
        compat.stub_torchcodec()
        try:
            spec = importlib.util.find_spec('torchcodec')
        finally:
            self._clean()
        assert spec is not None
        assert spec.name == 'torchcodec'

    def test_stub_exposes_the_name_pyannote_imports(self):
        """pyannote.audio.core.io does ``from torchcodec import AudioSamples``.

        When that fails it warns that decoding is broken and falls back — which
        is what our own shim used to look like from the outside.
        """
        from ml_worker import compat
        self._clean()
        compat.stub_torchcodec()
        try:
            from torchcodec import AudioSamples
            from torchcodec.decoders import AudioDecoder
            fields = AudioSamples._fields
        finally:
            self._clean()
        assert fields == ('data', 'pts_seconds', 'sample_rate')
        assert AudioDecoder is not None

    def test_stub_idempotent(self):
        from ml_worker import compat
        self._clean()
        compat.stub_torchcodec()
        compat.stub_torchcodec()  # second call — no error
        import sys
        assert getattr(sys.modules.get('torchcodec'), '_pine_stub', False)
        self._clean()

    def test_stub_clears_broken_partial_import(self):
        import sys
        import types

        from ml_worker import compat
        self._clean()
        broken = types.ModuleType('torchcodec')
        sys.modules['torchcodec'] = broken
        compat.stub_torchcodec()
        assert getattr(sys.modules.get('torchcodec'), '_pine_stub', False), \
            "stub should have replaced the broken module"

    def test_stub_skips_when_real_works(self):
        """When real torchcodec loads AND decodes successfully, the shim should NOT be installed."""
        import sys
        import types

        from ml_worker import compat
        self._clean()

        # Plant a fake "real" torchcodec whose AudioDecoder accepts a file path
        # (the smoke test in stub_torchcodec instantiates AudioDecoder)
        fake = types.ModuleType('torchcodec')
        fake_dec = types.ModuleType('torchcodec.decoders')
        fake_dec.AudioDecoder = type('AudioDecoder', (), {
            '__init__': lambda self, *a, **kw: None,
        })
        fake.decoders = fake_dec
        sys.modules['torchcodec'] = fake
        sys.modules['torchcodec.decoders'] = fake_dec

        compat.stub_torchcodec()

        assert not getattr(sys.modules.get('torchcodec'), '_pine_stub', False), \
            "real torchcodec should not be replaced by the shim"
        self._clean()

    @needs_torch
    def test_audio_decoder_loads_wav(self, tmp_path):
        """The shim AudioDecoder should load audio via ffmpeg."""
        import struct
        import wave

        from ml_worker import compat
        self._clean()
        compat.stub_torchcodec()

        # Create a 1-second 16kHz mono WAV with the stdlib wave module
        wav_path = str(tmp_path / 'test.wav')
        n_frames = 16000
        with wave.open(wav_path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack(f'<{n_frames}h', *([0] * n_frames)))

        try:
            from torchcodec.decoders import AudioDecoder
            dec = AudioDecoder(wav_path)
        except (RuntimeError, FileNotFoundError, OSError):
            pytest.skip('ffmpeg not available on this platform')

        # metadata
        assert dec.metadata.sample_rate == 16000
        assert dec.metadata.num_channels == 1
        assert dec.metadata.num_frames == n_frames

        # get_all_samples
        samples = dec.get_all_samples()
        assert samples.data.shape[0] == 1       # mono
        assert samples.data.shape[1] == n_frames
        assert samples.sample_rate == 16000

        # get_samples_played_in_range (first 0.5s)
        chunk = dec.get_samples_played_in_range(0.0, 0.5)
        assert chunk.data.shape == (1, 8000)
        assert chunk.sample_rate == 16000
        self._clean()


# ---------------------------------------------------------------------------
# DiarizeOutput unwrap (pyannote 4.x)
# ---------------------------------------------------------------------------

class TestDiarizeOutputUnwrap:
    """Ensure the native Diarizer.run unwraps DiarizeOutput-style pyannote results."""

    @needs_torch
    def test_unwraps_diarize_output(self):
        """Pipeline returning DiarizeOutput should feed assign_speakers_simple."""
        from dataclasses import dataclass

        annotation = FakeDiarization([
            (0.0, 5.0, 'SPEAKER_00'),
            (5.0, 10.0, 'SPEAKER_01'),
        ])

        @dataclass
        class DiarizeOutput:
            speaker_diarization: object
            exclusive_speaker_diarization: object = None
            speaker_embeddings: object = None

        class Pipe:
            model = MagicMock()

            def __call__(self, x, **kwargs):
                return DiarizeOutput(speaker_diarization=annotation)

        d = _make_diarizer(engine_kind='mlx', device='cpu', pipeline=Pipe())

        segments = [
            {'start': 1.0, 'end': 4.0, 'text': 'Hello',
             'words': [{'start': 1.0, 'end': 2.0, 'word': 'Hello'}]},
            {'start': 6.0, 'end': 9.0, 'text': 'World',
             'words': [{'start': 6.0, 'end': 7.0, 'word': 'World'}]},
        ]

        import numpy as np
        diarize_input = np.zeros(16000 * 10, dtype=np.float32)

        result = d.run(diarize_input, segments, 'test-rec')
        assert result[0]['speaker'] == 'SPEAKER_00'
        assert result[1]['speaker'] == 'SPEAKER_01'

    @needs_torch
    def test_the_voices_travel_with_the_labels(self):
        """What each label sounds like reaches the step that merges speakers."""
        from dataclasses import dataclass

        class Labelled(FakeDiarization):
            def labels(self):
                return sorted({spk for _s, _e, spk in self._turns})

        annotation = Labelled([(0.0, 5.0, 'SPEAKER_00'), (5.0, 10.0, 'SPEAKER_01')])

        @dataclass
        class DiarizeOutput:
            speaker_diarization: object
            exclusive_speaker_diarization: object = None
            speaker_embeddings: object = None

        class Pipe:
            model = MagicMock()

            def __call__(self, x, **kwargs):
                return DiarizeOutput(speaker_diarization=annotation,
                                     exclusive_speaker_diarization=annotation,
                                     speaker_embeddings=[[2.0, 0.0], [0.0, 3.0]])

        import numpy as np
        d = _make_diarizer(engine_kind='mlx', device='cpu', pipeline=Pipe())
        result = d.compute(np.zeros(16000 * 10, dtype=np.float32), 'test-rec')

        assert result.centroids == {'SPEAKER_00': [1.0, 0.0],
                                    'SPEAKER_01': [0.0, 1.0]}
        assert list(result.itertracks(yield_label=True))[1][2] == 'SPEAKER_01'


class TestSpeakerCountThreading:
    """Per-recording speaker count is forwarded into the diarization pipeline."""

    def test_speaker_kwargs_exact_and_fallback(self):
        from ml_worker.diarize import speaker_kwargs
        assert speaker_kwargs(2) == {'num_speakers': 2}
        assert speaker_kwargs(1) == {'num_speakers': 1}
        assert speaker_kwargs(None) == {'min_speakers': 2, 'max_speakers': 4}
        assert speaker_kwargs(0) == {'min_speakers': 2, 'max_speakers': 4}

    def _capture_pipe_diarizer(self, captured):
        annotation = FakeDiarization([(0.0, 5.0, 'SPEAKER_00')])

        class CapturePipe:
            model = MagicMock()

            def __call__(self, audio_input, **kwargs):
                captured.update(kwargs)
                return annotation

        return _make_diarizer(engine_kind='mlx', device='cpu', pipeline=CapturePipe())

    @needs_torch
    def test_num_speakers_passed_to_pipeline(self):
        captured = {}
        d = self._capture_pipe_diarizer(captured)
        segments = [{'start': 0.5, 'end': 4.5, 'text': 'Hi',
                     'words': [{'start': 0.5, 'end': 4.5, 'word': 'Hi'}]}]
        dummy_audio = {"waveform": __import__('torch').zeros(1, 16000),
                       "sample_rate": 16000}
        d.run(dummy_audio, segments, 1, num_speakers=2)
        assert captured == {'num_speakers': 2}

    @needs_torch
    def test_no_count_falls_back_to_min_max(self):
        captured = {}
        d = self._capture_pipe_diarizer(captured)
        segments = [{'start': 0.5, 'end': 4.5, 'text': 'Hi',
                     'words': [{'start': 0.5, 'end': 4.5, 'word': 'Hi'}]}]
        dummy_audio = {"waveform": __import__('torch').zeros(1, 16000),
                       "sample_rate": 16000}
        d.run(dummy_audio, segments, 1)
        assert 'num_speakers' not in captured
        assert 'min_speakers' in captured and 'max_speakers' in captured


# ---------------------------------------------------------------------------
# Mac pipeline: MLX cache clearing
# ---------------------------------------------------------------------------

class TestClearMlxCache:
    """_clear_mlx_cache should be a safe no-op when mlx is not installed."""

    def test_no_mlx_installed(self):
        from app.services.transcription import _clear_mlx_cache
        # Should not raise even when mlx is not installed (Windows/Linux)
        _clear_mlx_cache()

    def test_mlx_cache_called(self):
        """When mlx is available, mx.clear_cache() should be called."""
        from app.services.transcription import _clear_mlx_cache
        mock_mx = MagicMock()
        with patch.dict('sys.modules', {'mlx': MagicMock(), 'mlx.core': mock_mx}):
            _clear_mlx_cache()
            # If mlx.core was importable, clear_cache would be called


# ---------------------------------------------------------------------------
# Mac pipeline: model choice
# ---------------------------------------------------------------------------

class TestMacModelChoice:
    """get_models_for_setup should select the quality model on Mac."""

    @patch('app.services.model_manager.IS_MAC', True)
    def test_model_on_mac(self):
        from app.services.model_manager import get_models_for_setup
        ids = get_models_for_setup(modules=[])
        assert 'mlx-whisper-large-v3' in ids

    @patch('app.services.model_manager.IS_MAC', True)
    def test_normalize_rewrites_whisperx_to_mlx_on_mac(self):
        """WhisperX is never installed on Mac, so the ID must not survive."""
        from app.services.model_manager import normalize_stt_model_id
        assert normalize_stt_model_id('whisperx-large-v3') == 'mlx-whisper-large-v3'
        assert normalize_stt_model_id('mlx-whisper-large-v3') == 'mlx-whisper-large-v3'

    @patch('app.services.model_manager.IS_MAC', False)
    def test_normalize_rewrites_mlx_to_whisperx_off_mac(self):
        from app.services.model_manager import normalize_stt_model_id
        assert normalize_stt_model_id('mlx-whisper-large-v3') == 'whisperx-large-v3'
        assert normalize_stt_model_id('whisperx-large-v3') == 'whisperx-large-v3'

    def test_unknown_model_falls_back_to_the_default(self):
        from app.services.model_manager import get_default_stt_model, normalize_stt_model_id
        assert normalize_stt_model_id('no-such-model') == get_default_stt_model()

    @patch('app.services.pip_installer.IS_MAC', True)
    @patch('app.services.pip_installer._is_package_installed')
    def test_check_ml_deps_never_asks_for_whisperx_on_mac(self, mock_is_installed):
        from app.services.pip_installer import check_ml_deps
        mock_is_installed.return_value = False
        deps = check_ml_deps()
        assert 'whisperx' not in deps
        assert 'mlx-whisper' in deps


class TestEngineSelection:
    def test_whisperx_large_v3_reuses_detected_cuda_defaults(self):
        from ml_worker.engines import select_engine_config

        engine, device, compute_type = select_engine_config(
            'whisperx-large-v3',
            prefer_mps=False,
            detected_device='cuda',
            detected_compute='float16',
        )

        assert engine == 'whisperx'
        assert device == 'cuda'
        assert compute_type == 'float16'

    def test_mlx_model_maps_to_mlx_engine(self):
        from ml_worker.engines import select_engine_config

        engine, device, compute_type = select_engine_config(
            'mlx-whisper-large-v3', prefer_mps=True)

        assert engine == 'mlx'
        assert device == 'cpu'
        assert compute_type is None

    def test_whisperx_on_mac_prefers_cpu_int8(self):
        from ml_worker.engines import select_engine_config

        engine, device, compute_type = select_engine_config(
            'whisperx-large-v3', prefer_mps=True,
            detected_device='cuda', detected_compute='float16')

        assert engine == 'whisperx'
        assert device == 'cpu'
        assert compute_type == 'int8'


# ---------------------------------------------------------------------------
# Mac pipeline: _write_wav helper
# ---------------------------------------------------------------------------

class TestParallelStages:
    """Diarization under the STT stage is opt-in on every platform. On a Mac,
    with pyannote on the CPU beside mlx-whisper, it had not finished 33 minutes
    in, and the job sat at 99% until the tab was closed."""

    @pytest.mark.parametrize('value, expected', [
        (None, 'False'), ('', 'False'), ('0', 'False'), ('1', 'True')])
    def test_only_asking_for_it_turns_it_on(self, value, expected):
        # A fresh interpreter rather than a reload: other modules hold what
        # they imported from ``constants``, and a reload would leave them stale.
        import subprocess
        import sys
        env = {k: v for k, v in os.environ.items() if k != 'PINE_PARALLEL_STAGES'}
        if value is not None:
            env['PINE_PARALLEL_STAGES'] = value
        backend = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(
            [sys.executable, '-c',
             'from ml_worker.constants import PARALLEL_STAGES; print(PARALLEL_STAGES)'],
            cwd=backend, env=env, capture_output=True, text=True, check=True)
        assert out.stdout.strip() == expected


class TestWriteWav:
    """Test the _write_wav helper produces valid WAV files."""

    @needs_numpy
    def test_write_and_read_wav(self, tmp_path):
        import numpy as np

        from app.services.transcription import _write_wav
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, 16000, dtype=np.float32))
        wav_path = str(tmp_path / 'test.wav')
        _write_wav(wav_path, audio)

        # Verify it's a valid WAV: check RIFF header
        with open(wav_path, 'rb') as f:
            header = f.read(4)
            assert header == b'RIFF'
            f.seek(8)
            assert f.read(4) == b'WAVE'


class TestModelPreflight:
    """A model that was never downloaded is a setup problem, caught before the job."""

    def _ready(self, app, model_id, models_path):
        from app.extensions import db
        from app.models.ml_model import MLModel
        os.makedirs(os.path.join(models_path, model_id), exist_ok=True)
        with app.app_context():
            row = db.session.get(MLModel, model_id)
            row.status = 'ready'
            db.session.commit()

    def test_missing_model_refuses_the_job(self, app):
        from app.services.transcription.job_runner import _preflight_stt_model

        with pytest.raises(RuntimeError) as exc:
            _preflight_stt_model(app)
        assert 'not installed' in str(exc.value)

    def test_installed_model_passes(self, app, temp_dir):
        from app.services.model_manager import get_default_stt_model
        from app.services.transcription.job_runner import _preflight_stt_model

        models_path = os.path.join(temp_dir, 'models')
        self._ready(app, get_default_stt_model(), models_path)

        assert _preflight_stt_model(app) == get_default_stt_model()


class TestHfHubOfflineShim:
    """compat.patch_hf_hub_is_offline_mode — what broke onboarding's warm-up.

    Fake modules rather than the real huggingface_hub: the shim has to behave
    the same whether or not the interpreter running the suite has the ML stack.
    """

    def _fake_hub(self, monkeypatch, *, with_helper=False, offline=False):
        import sys
        import types

        constants = types.ModuleType('huggingface_hub.constants')
        constants.HF_HUB_OFFLINE = offline
        hub = types.ModuleType('huggingface_hub')
        hub.constants = constants
        if with_helper:
            hub.is_offline_mode = lambda: 'the original'
        monkeypatch.setitem(sys.modules, 'huggingface_hub', hub)
        monkeypatch.setitem(sys.modules, 'huggingface_hub.constants', constants)
        return hub, constants

    def _unpatched(self, monkeypatch):
        from ml_worker import compat
        monkeypatch.setattr(compat, '_HF_HUB_OFFLINE_MODE_PATCHED', False)
        return compat

    def test_restores_the_helper_the_package_dropped(self, monkeypatch):
        compat = self._unpatched(monkeypatch)
        hub, _ = self._fake_hub(monkeypatch, offline=True)

        compat.patch_hf_hub_is_offline_mode()

        assert hub.is_offline_mode() is True

    def test_reads_the_flag_each_call_rather_than_snapshotting_it(self, monkeypatch):
        """apply_hf_offline flips the constant at runtime and must stay in charge."""
        compat = self._unpatched(monkeypatch)
        hub, constants = self._fake_hub(monkeypatch, offline=False)

        compat.patch_hf_hub_is_offline_mode()
        assert hub.is_offline_mode() is False

        constants.HF_HUB_OFFLINE = True
        assert hub.is_offline_mode() is True

    def test_leaves_a_real_helper_alone(self, monkeypatch):
        compat = self._unpatched(monkeypatch)
        hub, _ = self._fake_hub(monkeypatch, with_helper=True)

        compat.patch_hf_hub_is_offline_mode()

        assert hub.is_offline_mode() == 'the original'


class TestTrustedTorchLoad:
    """compat.trusted_torch_load — the scoped loan the Flask warm-up takes."""

    def _fake_torch(self, monkeypatch, calls):
        import sys
        import types

        torch = types.ModuleType('torch')

        def load(*args, **kwargs):
            calls.append(kwargs.get('weights_only', 'unset'))
            return 'checkpoint'

        torch.load = load
        monkeypatch.setitem(sys.modules, 'torch', torch)
        return torch, load

    def test_relaxes_the_loader_and_hands_it_back(self, monkeypatch):
        from ml_worker import compat
        calls = []
        torch, original = self._fake_torch(monkeypatch, calls)

        with compat.trusted_torch_load():
            torch.load('model.bin')
        torch.load('model.bin')

        assert calls == [False, 'unset']
        assert torch.load is original

    def test_hands_it_back_after_a_failed_load(self, monkeypatch):
        from ml_worker import compat
        torch, original = self._fake_torch(monkeypatch, [])

        with pytest.raises(RuntimeError):
            with compat.trusted_torch_load():
                raise RuntimeError('checkpoint is corrupt')

        assert torch.load is original

    def test_steps_aside_when_the_worker_already_patched_torch(self, monkeypatch):
        from ml_worker import compat
        calls = []
        torch, original = self._fake_torch(monkeypatch, calls)
        original._pine_trusted_checkpoint_wrap = True

        with compat.trusted_torch_load():
            assert torch.load is original

        assert torch.load is original

    def test_honours_the_strict_weights_only_override(self, monkeypatch):
        from ml_worker import compat
        calls = []
        torch, original = self._fake_torch(monkeypatch, calls)
        monkeypatch.setenv('PINE_TORCH_STRICT_WEIGHTS_ONLY', '1')

        with compat.trusted_torch_load():
            torch.load('model.bin')

        assert calls == ['unset']
        assert torch.load is original
