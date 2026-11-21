"""Splitting a Whisper segment where its words change speaker.

whisperx.assign_word_speakers labels every timed word, but a segment keeps a
single speaker: the one that overlaps it most. When two people talk inside one
segment, the second person's words used to be filed under the first.
"""
from ml_worker.pipeline import split_segments_at_speaker_changes as split


def _seg(words, speaker=None):
    """words: (text, start, end, speaker) tuples."""
    ws = []
    for text, start, end, spk in words:
        w = {'word': text, 'start': start, 'end': end}
        if spk is not None:
            w['speaker'] = spk
        ws.append(w)
    return {
        'start': ws[0]['start'],
        'end': ws[-1]['end'],
        'text': ' '.join(w['word'] for w in ws),
        'words': ws,
        'speaker': speaker or ws[0].get('speaker'),
    }


def test_single_speaker_segment_is_left_alone():
    seg = _seg([('Hello', 0.0, 0.4, 'A'), ('there.', 0.5, 0.9, 'A')])
    assert split([seg]) == [seg]


def _timed(text, speaker, t0):
    return [(w, t0 + i * 0.3, t0 + i * 0.3 + 0.2, speaker)
            for i, w in enumerate(text.split())]


def test_two_speakers_in_an_unpunctuated_stretch_become_two_segments():
    # The SWE interview at 60:08: Whisper dropped the punctuation, so the only
    # place to cut is where the speakers change.
    a = 'i would have rather just used the title and author of the book'
    b = 'yeah it is tricky because you do not have access to that library'
    seg = _seg(_timed(a, 'A', 0.0) + _timed(b, 'B', 10.0), speaker='B')
    out = split([seg])
    assert [(s['speaker'], s['text']) for s in out] == [('A', a), ('B', b)]
    assert out[0]['start'] == 0.0
    assert out[1]['start'] == 10.0
    assert out[1]['end'] == seg['end']


def test_a_short_unpunctuated_mix_is_not_cut():
    # Fewer words a side than a sentence cut needs: the segment stays whole.
    seg = _seg(_timed('I used the title', 'A', 0.0)
               + _timed('yeah it is tricky', 'B', 2.0), speaker='B')
    out = split([seg])
    assert out == [seg]
    assert out[0]['speaker'] == 'B'


def test_every_word_survives_the_split_in_order():
    words = [(f'w{i}', i * 0.5, i * 0.5 + 0.4, 'A' if i < 5 else 'B') for i in range(10)]
    seg = _seg(words)
    out = split([seg])
    assert [w['word'] for s in out for w in s['words']] == [f'w{i}' for i in range(10)]


def test_one_stray_word_is_jitter_not_a_turn():
    # A word straddling the change gets the other speaker; it is not a reply.
    seg = _seg([
        ('so', 0.0, 0.2, 'A'), ('we', 0.3, 0.4, 'A'), ('would', 0.5, 0.7, 'B'),
        ('store', 0.8, 1.0, 'A'), ('the', 1.1, 1.2, 'A'), ('id', 1.3, 1.5, 'A'),
    ], speaker='A')
    out = split([seg])
    assert len(out) == 1
    assert out[0]['speaker'] == 'A'


def test_a_one_word_sentence_from_the_other_speaker_is_kept():
    seg = _seg([
        ('Does', 0.0, 0.2, 'A'), ('that', 0.3, 0.4, 'A'), ('make', 0.5, 0.6, 'A'),
        ('sense?', 0.7, 1.0, 'A'), ('Yes.', 1.4, 1.7, 'B'),
    ], speaker='A')
    out = split([seg])
    assert [(s['speaker'], s['text']) for s in out] == [
        ('A', 'Does that make sense?'), ('B', 'Yes.')]


def test_a_sentence_goes_whole_to_whoever_said_most_of_it():
    # Gordon at 16:46: the diarizer gave "Вы его" to the interviewer and
    # "предложили, да?" back to the guest. A word-level cut put half a question
    # on each line; the sentence stays together instead.
    seg = _seg([
        ('Речь', 0.0, 0.3, 'G'), ('шла', 0.4, 0.6, 'G'), ('о', 0.7, 0.8, 'G'),
        ('нем…', 0.9, 1.2, 'G'),
        ('Вы', 1.5, 1.6, 'D'), ('его', 1.7, 1.8, 'D'), ('ведь', 1.9, 2.0, 'D'),
        ('предложили,', 2.1, 2.6, 'G'), ('да?', 2.7, 2.9, 'G'),
    ], speaker='G')
    out = split([seg])
    assert [(s['speaker'], s['text']) for s in out] == [
        ('G', 'Речь шла о нем…'), ('D', 'Вы его ведь предложили, да?')]


def test_a_dialogue_dash_does_not_vote():
    # Gordon at 68:25, as the diarizer labelled it: the dash sits on the
    # interviewer's side of the boundary, but it is punctuation, not speech.
    seg = _seg([
        ('Когда', 0.0, 0.3, 'D'), ('вы', 0.4, 0.5, 'D'), ('анонсировали', 0.6, 1.2, 'D'),
        ('интервью…', 1.3, 1.9, 'D'),
        ('—', 2.0, 2.1, 'D'), ('Можно', 2.2, 2.5, 'D'), ('сразу', 2.6, 2.9, 'G'),
        ('стоп?', 3.0, 3.3, 'G'),
    ], speaker='D')
    out = split([seg])
    assert [(s['speaker'], s['text']) for s in out] == [
        ('D', 'Когда вы анонсировали интервью…'), ('G', '— Можно сразу стоп?')]


def test_the_full_stop_says_where_the_turn_ends():
    # The diarizer flips one word late; the full stop says where the turn ends.
    seg = _seg([
        ('That', 0.0, 0.2, 'A'), ('is', 0.3, 0.4, 'A'), ('right.', 0.5, 0.8, 'A'),
        ('Well,', 1.0, 1.2, 'A'), ('not', 1.3, 1.4, 'B'), ('quite', 1.5, 1.7, 'B'),
        ('so.', 1.8, 2.0, 'B'),
    ], speaker='A')
    out = split([seg])
    assert [(s['speaker'], s['text']) for s in out] == [
        ('A', 'That is right.'), ('B', 'Well, not quite so.')]


def test_words_without_a_speaker_follow_their_neighbours():
    seg = _seg([
        ('We', 0.0, 0.2, 'A'), ('paid', 0.3, 0.5, 'A'), ('1990', 0.6, 0.9, None),
        ('dollars.', 1.0, 1.3, 'A'), ('Too', 1.6, 1.8, 'B'), ('much.', 1.9, 2.2, 'B'),
    ], speaker='A')
    out = split([seg])
    assert [(s['speaker'], s['text']) for s in out] == [
        ('A', 'We paid 1990 dollars.'), ('B', 'Too much.')]


def test_segment_without_word_speakers_is_unchanged():
    seg = _seg([('No', 0.0, 0.2, None), ('labels', 0.3, 0.6, None)], speaker='A')
    assert split([seg]) == [seg]


def test_segment_without_words_is_unchanged():
    seg = {'start': 0.0, 'end': 1.0, 'text': 'Hi.', 'speaker': 'A'}
    assert split([seg]) == [seg]


def test_three_turns_in_one_segment():
    seg = _seg([
        ('Is', 0.0, 0.1, 'A'), ('it', 0.2, 0.3, 'A'), ('expensive?', 0.4, 0.9, 'A'),
        ('Very.', 1.2, 1.5, 'B'),
        ('Why?', 1.8, 2.1, 'A'),
    ])
    out = split([seg])
    assert [(s['speaker'], s['text']) for s in out] == [
        ('A', 'Is it expensive?'), ('B', 'Very.'), ('A', 'Why?')]
