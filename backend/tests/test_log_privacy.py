"""Transcript text stays out of the logs.

Logs are kept for a week and attached to bug reports. A dropped hallucination
was logged with its text; the signatures that drop one can match a whole
segment around a credit, so what got logged could be what a respondent said.
The time range and length are enough to find it again in the recording.
"""

import logging


def test_a_dropped_hallucination_is_logged_without_its_text(caplog):
    from ml_worker.pipeline import clean_transcript_segments

    caplog.set_level(logging.DEBUG)
    segments = [
        {'start': 0.0, 'end': 4.0, 'text': 'Я переехала на Лесную улицу', 'speaker': 'SPEAKER_00'},
        {'start': 10.0, 'end': 12.0, 'text': 'Продолжение следует...', 'speaker': ''},
    ]

    kept = clean_transcript_segments(segments)

    assert [s['text'] for s in kept] == ['Я переехала на Лесную улицу']
    assert 'Dropped Whisper hallucination' in caplog.text
    assert 'Продолжение' not in caplog.text
    assert '10.0-12.0' in caplog.text
