"""Codes and comments stay on their words when "Remove PII" rewrites the text.

Redaction swaps a name for a marker of another length, but tag spans and
comments kept their offsets into the original text, so everything after a
redacted name slid sideways — a code could land on the wrong words or vanish.

The oracle here is simple: the redacted export must read exactly like the
plain export with the name replaced by its marker.
"""

import json
import os

import pytest

NAME = 'Anna Petrova'
MARKER = '[person redacted]'

SEGMENTS = [
    {'start': 0, 'end': 4, 'speaker': 'SPEAKER_00',
     'text': 'I met Anna Petrova yesterday and she was kind.'},
    {'start': 4, 'end': 8, 'speaker': 'SPEAKER_00',
     'text': 'Then Anna Petrova left, and we talked about the budget.'},
    {'start': 8, 'end': 9, 'speaker': 'SPEAKER_01', 'text': 'Thank you, Anna Petrova, for coming.'},
]
BLOCK_0 = SEGMENTS[0]['text'] + ' ' + SEGMENTS[1]['text']   # one speaker, one block
BLOCK_1 = SEGMENTS[2]['text']


class _FakeGliner:
    def predict_entities(self, text, labels, threshold=0.5):
        found, i = [], text.find(NAME)
        while i >= 0:
            found.append({'start': i, 'end': i + len(NAME), 'text': NAME, 'label': 'person'})
            i = text.find(NAME, i + 1)
        return found


@pytest.fixture
def export(app, monkeypatch):
    """Write a recording with the given annotations; return (plain, redacted) Markdown."""
    from app.extensions import db
    from app.models.project import Project
    from app.models.recording import Recording
    from app.models.setting import Setting
    from app.services import pii_service
    from app.services.export_service import export_recording_markdown

    monkeypatch.setattr(pii_service, '_gliner_model', _FakeGliner())
    monkeypatch.setattr(pii_service, '_load_model', lambda app: None)

    def run(tag_spans=(), comments=()):
        with app.app_context():
            root = app.config['DEFAULT_PROJECTS_PATH']
            Setting.set('projects_path', root)
            proj = Project(name='Offsets', folder_name='offsets')
            db.session.add(proj)
            db.session.flush()
            rec = Recording(project_id=proj.id, original_name='interview.mp3',
                            stored_name='interview.mp3', transcript_path='interview_transcript.json',
                            transcription_status='transcribed')
            db.session.add(rec)
            db.session.commit()
            pid, rid = proj.id, rec.id

        project_dir = os.path.join(root, 'offsets')
        os.makedirs(project_dir, exist_ok=True)
        with open(os.path.join(project_dir, 'interview_transcript.json'), 'w', encoding='utf-8') as fh:
            json.dump({'segments': SEGMENTS}, fh)
        with open(os.path.join(project_dir, 'interview_annotations.json'), 'w', encoding='utf-8') as fh:
            json.dump({'tag_spans': list(tag_spans), 'comments': list(comments)}, fh)
        with open(os.path.join(project_dir, 'project_tags.json'), 'w', encoding='utf-8') as fh:
            json.dump([{'id': 't1', 'name': 'coded', 'color': '#ff0000'}], fh)

        opts = {'include_comments': True, 'include_tags': True, 'include_project_details': False,
                'include_participant_details': False, 'include_prompt': False}
        plain, err1 = export_recording_markdown(app, pid, rid, {**opts, 'remove_pii': False})
        redacted, err2 = export_recording_markdown(app, pid, rid, {**opts, 'remove_pii': True})
        assert err1 is None and err2 is None
        return plain, redacted

    return run


def _body(markdown):
    return [line for line in markdown.splitlines() if not line.startswith('*Exported')]


def _assert_same_but_redacted(plain, redacted):
    assert NAME in plain, 'the fixture lost its name; the test proves nothing'
    assert _body(redacted) == [line.replace(NAME, MARKER) for line in _body(plain)]


def _seg(idx, phrase):
    start = SEGMENTS[idx]['text'].index(phrase)
    return start, start + len(phrase)


class TestTagsKeepTheirWords:
    def test_a_tag_after_the_name_in_the_same_segment(self, export):
        sc, ec = _seg(0, 'she was kind')
        plain, redacted = export(tag_spans=[
            {'id': 's1', 'tag_id': 't1', 'segment_idx': 0, 'start_char': sc, 'end_char': ec},
        ])

        assert '**she was kind** [coded]' in redacted
        _assert_same_but_redacted(plain, redacted)

    def test_a_tag_over_the_name_keeps_the_marker(self, export):
        sc, ec = _seg(0, 'Anna Petrova yesterday')
        plain, redacted = export(tag_spans=[
            {'id': 's1', 'tag_id': 't1', 'segment_idx': 0, 'start_char': sc, 'end_char': ec},
        ])

        assert f'**{MARKER} yesterday** [coded]' in redacted
        _assert_same_but_redacted(plain, redacted)

    def test_a_tag_anchored_in_the_merged_block(self, export):
        start = BLOCK_0.index('the budget')
        plain, redacted = export(tag_spans=[
            {'id': 's1', 'tag_id': 't1', 'segment_idx': 1, 'start_char': 0, 'end_char': 0,
             'merged_start': start, 'merged_end': start + len('the budget')},
        ])

        assert '**the budget** [coded]' in redacted
        _assert_same_but_redacted(plain, redacted)

    @pytest.mark.parametrize('end_key', ['end_merged_end', 'end_seg_end_char'])
    def test_a_tag_across_two_speaker_turns(self, export, end_key):
        """Starts after a name in one turn, ends after a name in the next."""
        start = BLOCK_0.index('the budget')
        end = BLOCK_1.index(NAME) + len(NAME) + 1          # through "Anna Petrova,"
        plain, redacted = export(tag_spans=[
            {'id': 's1', 'tag_id': 't1', 'segment_idx': 1, 'start_char': 0, 'end_char': 0,
             'merged_start': start, 'end_segment_idx': 2, end_key: end},
        ])

        assert '**the budget.** [coded]' in redacted
        assert f'**Thank you, {MARKER},** [coded]' in redacted
        _assert_same_but_redacted(plain, redacted)


class TestCommentsKeepTheirWords:
    def test_a_comment_after_the_name(self, export):
        sc, ec = _seg(1, 'we talked')
        plain, redacted = export(comments=[
            {'id': 'c1', 'segment_idx': 1, 'start_char': sc, 'end_char': ec, 'text': 'key moment'},
        ])

        assert '**we talked** [Comment: key moment]' in redacted
        _assert_same_but_redacted(plain, redacted)


class TestTheMapping:
    """pii_service.redact_with_offsets on its own."""

    def test_offsets_before_inside_and_after_an_entity(self):
        from app.services.pii_service import redact_with_offsets

        text = 'I met Anna Petrova yesterday'
        s = text.index(NAME)
        new, where = redact_with_offsets(text, [{'start': s, 'end': s + len(NAME), 'label': 'person'}])

        assert new == f'I met {MARKER} yesterday'
        assert where(2) == 2                                   # before
        assert where(s + 3, 'start') == s                      # inside → marker start
        assert where(s + 3, 'end') == s + len(MARKER)          # inside → marker end
        assert where(s + len(NAME)) == s + len(MARKER)         # right after
        assert new[where(text.index('yesterday')):].startswith('yesterday')

    def test_overlapping_entities_become_one_marker(self):
        from app.services.pii_service import redact_with_offsets

        new, _ = redact_with_offsets('Anna Petrova', [
            {'start': 0, 'end': 12, 'label': 'person'},
            {'start': 5, 'end': 12, 'label': 'person'},
        ])

        assert new == MARKER
