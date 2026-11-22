"""Export transcript to Markdown or ODT."""

import logging
import os
from datetime import datetime

from .file_utils import atomic_read_json, safe_child_path

# How PINE's own markdown reads (see _render_transcript_blocks and
# _build_project_header). Both prompts carry it, so a change to the export
# format has one place to update here.
_EXPORT_READING_GUIDE = """--- HOW TO READ THE EXPORT ---

- The export comes after these instructions. It starts with "# <project name>", followed by the project details, if included.
- Each transcript starts with a heading "# <interview name> (Transcript)", optionally followed by "## Segment" (segment name, with recruitment criteria in parentheses) and "## Additional details" (the researcher's notes about the participant).
- Each speaker turn starts with the speaker's name in bold on its own line (e.g. **Anna**, **SPEAKER_01**, **Speaker**), followed by what was said.
- A tagged passage looks like **passage** [Tag name]. Several tags on the same passage: **passage** [Tag A] [Tag B]. A tag inside a longer tagged passage: text **inner passage** [Inner tag] more text [Outer tag].
- A researcher comment on a passage looks like **passage** [Comment: comment text]. A comment on a whole turn is a line starting with > *Comment:*
- Tags and comments are the researcher's annotations, not words spoken in the interview. Comments are the researcher's interpretations: use them as hints, never as evidence of what a participant said or did.
- [REDACTED] marks removed personal information. Do not guess or reconstruct it.
- Key findings, Recommendations, Results and recommendations, and Further steps in the project details are the researcher's earlier conclusions. Treat them as claims to check against the transcripts, not as evidence.
- Transcripts come from automatic speech recognition, and speaker names from automatic speaker detection. Words may be misheard, and turns may be attributed to the wrong speaker."""

_EXPORT_SPEAKER_STEP = """   - Use the Interview Guide (if present) and the flow of questions and answers to tell the interviewer from the participant
   - Only the participant's words are evidence; the interviewer's questions, suggestions and paraphrases are not
   - If a turn is clearly attributed to the wrong speaker, rely on its content and mention the issue"""

_EXPORT_LANGUAGE_AND_QUOTE_RULES = """- Respond in the language of the user's request that accompanies this export; if there is no request, respond in the language of the interviews
- Keep quotes in their original language; if it differs from the response language, add a translation in brackets after the quote
- Quote only the spoken words: leave out the ** markers, tag names in [ ] and comments
- Quote the transcript as written, even where recognition is imperfect; do not build conclusions on words that look misheard, and mark such passages as unclear"""

DEFAULT_EXPORT_PROMPT = """You will receive a research project export that may include:
- Project details: Description, Research objective, Research questions, Hypotheses, Stakeholders, Methodology, Interview Guide, and the researcher's earlier Key findings, Recommendations, Results and recommendations, Further steps
- Multiple interview transcripts
- For each interview: the participant's user segment (name and recruitment criteria) and additional details
- Optional tags and comments added by the researcher

Your task is to produce a structured synthesis of the research.

""" + _EXPORT_READING_GUIDE + """

--- PROCESS ---

1. Understand the research context:
   - Extract and restate the Research objective and Research questions
   - Note hypotheses and the researcher's earlier conclusions, if present

2. Identify the speakers in each interview:
""" + _EXPORT_SPEAKER_STEP + """

3. Assess available data:
   - If tags are present → use them as a primary signal
   - If tags are missing → perform bottom-up thematic analysis across transcripts

4. Analyze interviews:
   - Identify patterns, recurring themes, and contradictions
   - Count how many interviews support each pattern
   - Highlight notable quotes where useful
   - Compare across user segments (if provided)

5. Synthesize insights:
   - Move from observations → patterns → insights (why it matters)
   - Answer each research question
   - Validate or challenge each hypothesis and earlier conclusion

6. Generate actionable output

--- OUTPUT FORMAT ---

## 1. Research Summary
- Objective
- Methodology (brief)
- Number of interviews analyzed and user segments covered

## 2. Answers to Research Questions
For each research question:
- Answer based on the data, or "insufficient data"
- Supporting evidence
- Strength of evidence (strong / moderate / weak)

## 3. Hypotheses (if provided)
For each hypothesis:
- Verdict: supported / refuted / mixed / insufficient data
- Supporting evidence

## 4. Key Themes
For each theme:
- Theme name
- Description
- Prevalence (e.g. "5 of 8 interviews")
- Supporting evidence (quotes or paraphrased patterns)
- Affected user segments

## 5. Insights
- Insight statement (clear, non-obvious, decision-relevant)
- Supporting reasoning
- Related themes and research questions

## 6. Opportunities / Recommendations
- Actionable product or research recommendations
- Prioritize (High / Medium / Low impact)

## 7. Segment Differences (if applicable)
- Key differences between user groups

## 8. Open Questions / Gaps
- What remains unclear
- Earlier researcher conclusions that the transcripts do not support
- What should be researched next

--- IMPORTANT RULES ---

""" + _EXPORT_LANGUAGE_AND_QUOTE_RULES + """
- After each quote, name its source: (<interview name>, <speaker>)
- Every observation, pattern and quote must come from the provided data; do NOT invent quotes, behaviors, or patterns
- Insights and recommendations are interpretations: they are welcome, but each must be traceable to specific evidence and clearly distinguishable from observations
- Clearly signal uncertainty or weak evidence
- State prevalence as counts ("3 of 7 interviews"), not as "many" or "most"; with only a few interviews, say that the findings are indicative
- Do not summarize interview-by-interview; focus on cross-interview synthesis

--- FINAL VALIDATION ---

- Re-check every quote included in the output
- Ensure each quote exists verbatim in the provided transcripts
- Remove or correct any quote that cannot be directly verified"""

DEFAULT_EXPORT_PROMPT_RECORDING = """You will receive a single interview export that may include:
- Project details: Description, Research objective, Research questions, Hypotheses, Stakeholders, Methodology, Interview Guide, and the researcher's earlier Key findings, Recommendations, Results and recommendations, Further steps
- The participant's user segment (name and recruitment criteria) and additional details
- A full transcript
- Optional tags and comments added by the researcher

Your task is to:
1) Analyze the interview
2) Extract structured insights
3) Connect findings to the research context

""" + _EXPORT_READING_GUIDE + """

--- PROCESS ---

1. Understand context:
   - Identify the Research objective and Research questions
   - Note the user segment and additional details (if provided)

2. Identify the speakers:
""" + _EXPORT_SPEAKER_STEP + """

3. Analyze the transcript:
   - If tags are present → organize findings around them
   - If no tags → perform open coding (identify themes from scratch)

4. Structure findings:
   - Key topics discussed
   - Pain points, needs, behaviors, motivations
   - Notable quotes

5. Interpret:
   - Translate observations into insights
   - Indicate strength of evidence (strong / moderate / weak)

6. Connect to broader research:
   - How this interview informs or challenges each research question and hypothesis
   - Whether it supports or contradicts the researcher's earlier conclusions

--- OUTPUT FORMAT ---

## 1. Interview Summary
- Participant segment (if known)
- Context of the conversation
- Key topics covered

## 2. Key Findings
For each finding:
- Finding statement
- Evidence (quote or paraphrase)

## 3. Themes
- Emerging themes from this interview

## 4. Insights
- Insight statement
- Why it matters
- Confidence level (strong / moderate / weak)

## 5. Pain Points & Needs
- Problems mentioned
- Underlying needs

## 6. Research Questions & Hypotheses
- For each research question: what this interview contributes, or "not covered"
- For each hypothesis: supports / contradicts / not addressed, with evidence

## 7. Signals for Synthesis
- What should be compared across other interviews
- New hypotheses to validate

--- IMPORTANT RULES ---

""" + _EXPORT_LANGUAGE_AND_QUOTE_RULES + """
- After each quote, name the speaker
- Every observation and quote must come from the provided transcript; do NOT invent quotes or behaviors
- Insights are interpretations: they are welcome, but each must be traceable to specific evidence
- Clearly separate observations from interpretations
- Do not generalize beyond this single interview
- Explicitly note uncertainty where evidence is limited

--- FINAL VALIDATION ---

- Re-check every quote included in the output
- Ensure each quote exists verbatim in the transcript
- Remove or correct any quote that cannot be directly verified"""

log = logging.getLogger(__name__)

from flask import current_app

from ..extensions import db
from ..models.project import Project
from ..models.recording import Recording
from ..models.segment import Segment
from ..models.setting import Setting
from .speaker_blocks import block_text_and_offsets, merge_speaker_blocks


def _projects_root(app):
    return Setting.get('projects_path', app.config['DEFAULT_PROJECTS_PATH'])


def _fmt_time(secs):
    if not secs or secs < 0:
        return '0:00'
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    if h > 0:
        return f'{h}:{m:02d}:{s:02d}'
    return f'{m}:{s:02d}'


def _segment_export_line(segment):
    """Line under ## Segment: segment name, or name (recruitment criteria) if description is set."""
    if not segment:
        return 'Unassigned'
    name = segment.name if segment.name is not None else ''
    desc = (segment.description or '').strip()
    if desc:
        desc_flat = ' '.join(desc.split())
        return f'{name} ({desc_flat})'
    return name


def _merge_consecutive_speakers(segments):
    """Speaker turns for export, in reading order. See ``speaker_blocks``.

    Returns list of dicts: speaker, start, end, text, indices (original segment
    indices). An unnamed speaker exports as ``Speaker`` rather than a blank.
    """
    merged = merge_speaker_blocks(segments)
    for block in merged:
        block['speaker'] = block['speaker'] or 'Speaker'
    return merged


def _inline_span_end_cap(span):
    """End offset within merged block text for cross-segment spans (tags or comments)."""
    if span.get('end_merged_end') is not None:
        return span['end_merged_end']
    v = span.get('end_seg_end_char')
    return v


def _inline_span_bounds(kind, span, text, indices, offset_by_idx):
    """Compute [sc, ec) offsets for a tag or inline comment within merged block text."""
    len_t = len(text)
    is_cross_block = span.get('end_segment_idx') is not None
    if is_cross_block:
        start_in_block = span.get('segment_idx') in indices
        end_in_block = span.get('end_segment_idx') in indices
        if start_in_block and end_in_block:
            base = offset_by_idx.get(span['segment_idx'], 0)
            if span.get('merged_start') is not None:
                sc = span['merged_start']
            else:
                sc = base + span.get('start_char', 0)
            eme = _inline_span_end_cap(span)
            ec = eme if eme is not None else len_t
        elif start_in_block:
            base = offset_by_idx.get(span['segment_idx'], 0)
            if span.get('merged_start') is not None:
                sc = span['merged_start']
            else:
                sc = base + span.get('start_char', 0)
            ec = len_t
        elif end_in_block:
            sc = 0
            eme = _inline_span_end_cap(span)
            # None = legacy / missing; 0 would be zero-width and was wrongly skipped with pos==0
            ec = len_t if eme is None else max(0, min(eme, len_t))
        else:
            sc = 0
            ec = len_t
    elif (
        span.get('merged_start') is not None
        and span.get('merged_end') is not None
        and span.get('segment_idx') in indices
    ):
        sc = span['merged_start']
        ec = span['merged_end']
    else:
        base = offset_by_idx.get(span.get('segment_idx', 0), 0)
        sc = base + span.get('start_char', 0)
        ec = base + span.get('end_char', 0)
    sc = max(0, min(sc, len_t))
    ec = max(sc, min(ec, len_t))
    return sc, ec


def _tag_display_name(span, tag_map):
    tid = span.get('tag_id') or ''
    meta = tag_map.get(tid) or {}
    return (meta.get('name') or tid) or ''


def _tags_strictly_inside(inner, outer):
    _, isc, iec = inner
    _, osc, oec = outer
    return osc <= isc and iec <= oec and (osc < isc or iec < oec)


def _tag_find_roots(tag_items):
    roots = []
    for t in tag_items:
        if not any(_tags_strictly_inside(t, o) for o in tag_items if o is not t):
            roots.append(t)
    return roots


def _tag_intervals_overlap(sc1, ec1, sc2, ec2):
    return not (ec1 <= sc2 or ec2 <= sc1)


def _tag_overlapping_components(tag_items):
    """Split tag_items into connected components by overlapping [sc, ec) intervals."""
    if not tag_items:
        return []
    n = len(tag_items)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            _, a, b = tag_items[i]
            _, c, d = tag_items[j]
            if _tag_intervals_overlap(a, b, c, d):
                union(i, j)
    buckets = {}
    for i in range(n):
        r = find(i)
        buckets.setdefault(r, []).append(tag_items[i])
    return list(buckets.values())


def _tag_direct_children(outer, candidates):
    inside = [t for t in candidates if _tags_strictly_inside(t, outer)]
    children = []
    for t in inside:
        if not any(_tags_strictly_inside(t, o) for o in inside if o is not t):
            children.append(t)
    return sorted(children, key=lambda x: (x[1], -(x[2] - x[1]), str(x[0].get('tag_id') or '')))


def _render_one_tag_item(item, text, tag_map):
    span, sc, ec = item
    slice_t = text[sc:ec]
    tt = slice_t if slice_t else (span.get('anchor_text') or '')
    nm = _tag_display_name(span, tag_map)
    return f'**{tt}** [{nm}]'


def _render_same_bounds_tags(roots, text, tag_map):
    span0, sc, ec = roots[0]
    slice_t = text[sc:ec]
    tt = slice_t if slice_t else (span0.get('anchor_text') or '')
    labels = [f"[{_tag_display_name(r[0], tag_map)}]" for r in sorted(roots, key=lambda x: str(x[0].get('tag_id') or ''))]
    return f'**{tt}** ' + ' '.join(labels)


def _render_disjoint_roots(roots, all_tags, text, tag_map):
    roots = sorted(roots, key=lambda x: (x[1], x[2]))
    parts = []
    p = roots[0][1]
    for root in roots:
        sc, ec = root[1], root[2]
        if p < sc:
            parts.append(text[p:sc])
        sub = [t for t in all_tags if sc <= t[1] and t[2] <= ec]
        parts.append(_render_nested_tag_markdown(sub, text, tag_map))
        p = max(p, ec)
    return ''.join(parts)


def _render_outer_with_children(outer, all_tags, text, tag_map):
    _, osc, oec = outer
    children = _tag_direct_children(outer, all_tags)
    if not children:
        return _render_one_tag_item(outer, text, tag_map)
    parts = []
    p = osc
    for ch in children:
        _, isc, iec = ch
        if p < isc:
            parts.append(text[p:isc])
        sub = [t for t in all_tags if isc <= t[1] and t[2] <= iec]
        parts.append(_render_nested_tag_markdown(sub, text, tag_map))
        p = iec
    if p < oec:
        parts.append(text[p:oec])
    parts.append(f' [{_tag_display_name(outer[0], tag_map)}]')
    return ''.join(parts)


def _render_nested_tag_markdown(tag_items, text, tag_map):
    """Nested: plain prefix, **inner** [Inner], suffix, [Outer]. Same bounds: **t** [A] [B]."""
    if not tag_items:
        return ''
    roots = _tag_find_roots(tag_items)
    roots.sort(key=lambda x: (x[1], -(x[2] - x[1]), str(x[0].get('tag_id') or '')))
    sc0, ec0 = roots[0][1], roots[0][2]
    if len(roots) >= 2 and all(r[1] == sc0 and r[2] == ec0 for r in roots):
        return _render_same_bounds_tags(roots, text, tag_map)
    if len(roots) > 1:
        disjoint = all(
            roots[i][2] <= roots[j][1] or roots[j][2] <= roots[i][1]
            for i in range(len(roots))
            for j in range(i + 1, len(roots))
        )
        if disjoint:
            return _render_disjoint_roots(roots, tag_items, text, tag_map)
    return _render_outer_with_children(roots[0], tag_items, text, tag_map)


def _render_transcript_blocks(segments, merged_blocks, tag_spans, comments, tag_map, include_tags, include_comments):
    """Render merged speaker blocks to markdown lines with tags and comments."""
    # Pre-index by segment_idx for O(1) per-block lookup (avoids O(M×T) scan)
    # Cross-block spans are indexed into every segment they cover
    tag_index: dict = {}
    for t in tag_spans:
        start_idx = t.get('segment_idx')
        end_idx = t.get('end_segment_idx', start_idx)
        if start_idx is not None:
            for idx in range(start_idx, (end_idx if end_idx is not None else start_idx) + 1):
                tag_index.setdefault(idx, []).append(t)
    comment_index: dict = {}
    for c in comments:
        start_idx = c.get('segment_idx')
        end_idx = c.get('end_segment_idx', start_idx)
        if start_idx is not None:
            for idx in range(start_idx, (end_idx if end_idx is not None else start_idx) + 1):
                comment_index.setdefault(idx, []).append(c)

    lines = []
    for block in merged_blocks:
        speaker = block['speaker'] or 'Speaker'
        indices = block['indices']
        # Merged text and the offset map tag spans are anchored into. Shared
        # with the renderer and with editing, so a tag saved on screen lands on
        # the same words here.
        text, offset_by_idx = block_text_and_offsets(segments, indices)
        # Collect inline spans: tags and positioned comments, sorted by position
        # Collect inline spans, deduplicating cross-block tags (indexed into multiple segments)
        seen_tags = set()
        seg_tags = []
        if include_tags:
            for idx in indices:
                for t in tag_index.get(idx, []):
                    tid = id(t)
                    if tid not in seen_tags:
                        seen_tags.add(tid)
                        seg_tags.append(('tag', t))
        seen_cmts = set()
        seg_cmts_inline = []
        if include_comments:
            for idx in indices:
                for c in comment_index.get(idx, []):
                    if c.get('start_char') is None or c.get('end_char') is None:
                        continue
                    cid = id(c)
                    if cid not in seen_cmts:
                        seen_cmts.add(cid)
                        seg_cmts_inline.append(('cmt', c))
        spans = seg_tags + seg_cmts_inline
        resolved = []
        for kind, span in spans:
            sc, ec = _inline_span_bounds(kind, span, text, indices, offset_by_idx)
            resolved.append((kind, span, sc, ec))

        # `indices` is bound as a default on purpose: both sort keys are defined
        # inside the per-block loop and consumed in the same iteration, so late
        # binding is harmless today — but only by accident. Binding it now keeps
        # the key correct if a later edit ever defers the sort.
        def _span_export_sort_key(item, indices=indices):
            kind, span, sc, ec = item
            cross_continues = (
                span.get('end_segment_idx') is not None
                and span.get('segment_idx') not in indices
            )
            if kind == 'tag' and cross_continues:
                sc_key = 0
            elif kind == 'cmt' and cross_continues:
                sc_key = 0
            else:
                sc_key = sc
            length = ec - sc
            if kind == 'tag':
                return (sc_key, 0, -length)
            return (sc_key, 1, ec)

        resolved.sort(key=_span_export_sort_key)

        tag_items = [(span, sc, ec) for kind, span, sc, ec in resolved if kind == 'tag']
        cmt_items = [(span, sc, ec) for kind, span, sc, ec in resolved if kind == 'cmt']

        merged_events = []
        for comp in _tag_overlapping_components(tag_items):
            mn = min(t[1] for t in comp)
            mx = max(t[2] for t in comp)
            md = _render_nested_tag_markdown(comp, text, tag_map)
            span0 = comp[0][0]
            merged_events.append(('tagx', mn, mx, md, span0))
        for span, sc, ec in cmt_items:
            merged_events.append(('cmt', span, sc, ec))

        def _merged_event_sort_key(ev, indices=indices):
            if ev[0] == 'tagx':
                _, mn, mx, _md, span0 = ev
                if span0.get('end_segment_idx') is not None and span0.get('segment_idx') not in indices:
                    sc_key = 0
                else:
                    sc_key = mn
                return (sc_key, 0, -(mx - mn))
            _, span, sc, ec = ev
            if (
                span.get('end_segment_idx') is not None
                and span.get('segment_idx') not in indices
            ):
                sc_key = 0
            else:
                sc_key = sc
            return (sc_key, 1, ec)

        merged_events.sort(key=_merged_event_sort_key)

        if merged_events:
            parts = []
            pos = 0
            for ev in merged_events:
                if ev[0] == 'tagx':
                    _, sc, ec, md, _span0 = ev
                    if ec <= pos and (pos > 0 or ec > 0):
                        continue
                    if sc < pos:
                        sc = pos
                    if sc >= ec:
                        continue
                    if sc > pos:
                        parts.append(text[pos:sc])
                    parts.append(md)
                    pos = ec
                else:
                    _, span, sc, ec = ev
                    if ec <= pos and (pos > 0 or ec > 0):
                        continue
                    if sc < pos:
                        sc = pos
                    if sc >= ec:
                        continue
                    if sc > pos:
                        parts.append(text[pos:sc])
                    parts.append(f"**{text[sc:ec]}** [Comment: {span.get('text', '')}]")
                    pos = ec
            if pos < len(text):
                parts.append(text[pos:])
            text = ''.join(parts)
        lines.append(f'**{speaker}**')
        if text:
            lines.append(text)
        lines.append('')
        # Legacy comments without position info (no start_char/end_char) as block quotes
        if include_comments:
            for idx in indices:
                for c in comment_index.get(idx, []):
                    if c.get('start_char') is None or c.get('end_char') is None:
                        lines.append(f'  > *Comment:* {c.get("text", "")}')
    return lines


def _build_project_header(project):
    """Return lines for all non-empty project detail fields (no blank line between heading and content)."""
    lines = []
    if project.description:
        lines += ['## Description', project.description]
    if project.objective:
        lines += ['## Research objective', project.objective]
    questions = project.get_questions()
    if questions:
        lines += ['## Research questions']
        for i, q in enumerate(questions, 1):
            lines.append(f'{i}. {q}')
    hypotheses = project.get_hypotheses()
    if hypotheses:
        lines += ['## Hypotheses']
        for h in hypotheses:
            lines.append(f'- {h}')
    stakeholders = project.get_stakeholders()
    if stakeholders:
        lines += ['## Stakeholders']
        for i, s in enumerate(stakeholders, 1):
            lines.append(f'{i}. {s}')
    if project.methodology:
        lines += ['## Methodology', project.methodology]
    if project.interview_guide:
        lines += ['## Interview Guide', project.interview_guide]
    if project.key_findings:
        lines += ['## Key findings', project.key_findings]
    if project.recommendations:
        lines += ['## Recommendations', project.recommendations]
    if project.results_recommendations:
        lines += ['## Results and recommendations', project.results_recommendations]
    if project.further_steps:
        lines += ['## Further steps', project.further_steps]
    return lines


def _transcript_file(project_dir, transcript_path):
    """The recording's transcript inside *project_dir*, or None if it points elsewhere."""
    if not transcript_path:
        return None
    try:
        return safe_child_path(project_dir, transcript_path)
    except ValueError:
        log.warning('Export skipped a transcript path outside %s: %r', project_dir, transcript_path)
        return None


class _AnchorRemap:
    """Move tag and comment offsets from a transcript onto its redacted copy.

    Anchors are character offsets, into one segment's text or into the merged
    text of a speaker block (see ``_inline_span_bounds``). Redaction changes
    lengths, so every offset is taken through the old block to the segment it
    falls in, moved with that segment's ``where`` map, and placed back into the
    block that segment sits in after redaction. The export then reads exactly
    as it would without PII removal, names replaced by their markers.
    """

    def __init__(self, old_segments, new_segments, maps):
        self._old, self._new, self._maps = old_segments, new_segments, maps
        self._old_blocks = self._blocks(old_segments)
        self._new_blocks = self._blocks(new_segments)

    @staticmethod
    def _blocks(segments):
        where = {}
        for block in merge_speaker_blocks(segments):
            indices = block['indices']
            _, offsets = block_text_and_offsets(segments, indices)
            for i in indices:
                where[i] = (indices, offsets)
        return where

    @staticmethod
    def _lead(segments, i):
        text = segments[i].get('text') or ''
        return len(text) - len(text.lstrip())

    def _known(self, anchor, pos):
        return (isinstance(anchor, int) and anchor in self._old_blocks
                and isinstance(pos, (int, float)))

    def merged(self, anchor, pos, side):
        """An offset into the block holding segment *anchor*, after redaction."""
        if not self._known(anchor, pos):
            return pos
        indices, offsets = self._old_blocks[anchor]
        target = indices[0]
        for i in indices:
            if offsets[i] <= pos:
                target = i
        local = pos - offsets[target]
        # Blocks join stripped segment texts; the maps work on the raw text.
        raw = self._maps[target](local + self._lead(self._old, target), side)
        moved = max(0, raw - self._lead(self._new, target))
        _, new_offsets = self._new_blocks.get(target, (None, {}))
        return new_offsets.get(target, 0) + moved

    def segment(self, anchor, pos, side):
        """An offset relative to segment *anchor* (it may run on into the block)."""
        if not self._known(anchor, pos):
            return pos
        _, old_offsets = self._old_blocks[anchor]
        _, new_offsets = self._new_blocks.get(anchor, (None, {}))
        moved = self.merged(anchor, old_offsets[anchor] + pos, side)
        return moved - new_offsets.get(anchor, 0)

    def span(self, span):
        out = dict(span)
        idx = span.get('segment_idx')
        end_idx = span.get('end_segment_idx')
        if out.get('start_char') is not None:
            out['start_char'] = self.segment(idx, out['start_char'], 'start')
        if out.get('end_char') is not None:
            out['end_char'] = self.segment(idx, out['end_char'], 'end')
        if out.get('merged_start') is not None:
            out['merged_start'] = self.merged(idx, out['merged_start'], 'start')
        if out.get('merged_end') is not None:
            out['merged_end'] = self.merged(idx, out['merged_end'], 'end')
        # Both renderers read either key as an offset into the end block.
        end_anchor = end_idx if end_idx is not None else idx
        for key in ('end_merged_end', 'end_seg_end_char'):
            if out.get(key) is not None:
                out[key] = self.merged(end_anchor, out[key], 'end')
        return out


def _remove_pii_for_export(app, recording, segments, tag_spans, comments, notes, ordinal):
    """Redact what an export of *recording* would otherwise carry out by name.

    The transcript text was always covered. A respondent's name more often sits
    in the places that were not: the speaker they were renamed to, the file
    name, a comment, the participant notes. Speakers whose label holds PII
    become "Speaker N" (by order of appearance, so blocks stay apart); a
    recording name that holds any becomes "Recording N". Tags and comments are
    moved onto the redacted text (``_AnchorRemap``). Any detector failure
    raises PIIError and the export stops: nothing goes out half-redacted.

    Only the export's copies change; annotation files on disk are untouched.

    Returns ``(segments, tag_spans, comments, notes, title)``.
    """
    import re

    from .pii_service import (
        PIIError,
        _load_model,
        detect_pii,
        redact_segments_with_offsets,
        redact_text,
    )

    original_segments = segments
    try:
        _load_model(app)
        threshold = float(Setting.get('pii_threshold', '0.8'))
        segments, offset_maps = redact_segments_with_offsets(segments, threshold=threshold)
    except PIIError:
        raise
    except Exception as exc:
        raise PIIError(f'PII removal failed: {exc}') from exc

    def redact(text):
        if not text:
            return text
        entities = detect_pii(text, threshold=threshold)
        return redact_text(text, entities) if entities else text

    speakers = []
    for seg in segments:
        spk = seg.get('speaker') or ''
        if spk and spk not in speakers:
            speakers.append(spk)
    pseudonyms = {
        spk: f'Speaker {n}' for n, spk in enumerate(speakers, 1) if redact(spk) != spk
    }
    if pseudonyms:
        segments = [
            dict(seg, speaker=pseudonyms.get(seg.get('speaker'), seg.get('speaker')))
            for seg in segments
        ]

    remap = _AnchorRemap(original_segments, segments, offset_maps)

    def moved(span):
        out = remap.span(span)
        # The fallback text a renderer shows for an anchor with no width.
        if out.get('anchor_text'):
            out['anchor_text'] = redact(out['anchor_text'])
        return out

    tag_spans = [moved(t) for t in tag_spans]
    comments = [dict(moved(c), text=redact(c.get('text', ''))) for c in comments]
    notes = redact(notes)

    title = recording.original_name or ''
    # File names join words with _ - . which hides a name from the detector.
    readable = re.sub(r'[_.\-]+', ' ', os.path.splitext(title)[0]).strip()
    if redact(readable) != readable or redact(title) != title:
        title = f'Recording {ordinal}'

    return segments, tag_spans, comments, notes, title


def export_recording_markdown(app, project_id, recording_id, opts):
    """Export a single recording to Markdown."""
    with app.app_context():
        project = db.session.get(Project, project_id)
        recording = db.session.get(Recording, recording_id)
        if not project or not recording or recording.project_id != project_id:
            return None, 'Not found'

        root = _projects_root(app)
        project_dir = os.path.join(root, project.folder_name)
        transcript_path = _transcript_file(project_dir, recording.transcript_path)
        if not transcript_path or not os.path.isfile(transcript_path):
            return None, 'Transcript not found'

        transcript = atomic_read_json(transcript_path)

        from .annotations import annotation_recording_ref, get_annotations, get_project_tags
        ann = get_annotations(project_dir, annotation_recording_ref(recording))
        tags = get_project_tags(project_dir)
        tag_map = {t['id']: t for t in tags}

        include_comments = opts.get('include_comments', True)
        include_tags = opts.get('include_tags', True)
        include_participant_details = opts.get('include_participant_details', True)
        include_project_details = opts.get('include_project_details', True)
        remove_pii = opts.get('remove_pii', False)

        segments = transcript.get('segments', [])

        speaker_labels = ann.get('speaker_labels', {})
        if speaker_labels:
            for seg in segments:
                spk = seg.get('speaker', '')
                if spk and spk in speaker_labels:
                    seg['speaker'] = speaker_labels[spk]

        tag_spans = ann.get('tag_spans', [])
        comments = ann.get('comments', [])
        notes = (recording.participant_notes or '').strip() or '—'
        title = recording.original_name
        if remove_pii:
            segments, tag_spans, comments, notes, title = _remove_pii_for_export(
                current_app._get_current_object(), recording, segments, tag_spans, comments,
                notes, 1,
            )

        lines = [f'# {project.name}']
        if include_project_details:
            lines += _build_project_header(project)

        lines.append(f'# {title} (Transcript)')
        if include_participant_details and not project.is_system:
            seg = db.session.get(Segment, recording.segment_id) if recording.segment_id else None
            segment_name = _segment_export_line(seg)
            lines += ['## Segment', segment_name]
            lines += ['## Additional details', notes]
        lines.append('')

        merged_blocks = _merge_consecutive_speakers(segments)
        lines += _render_transcript_blocks(
            segments, merged_blocks, tag_spans, comments, tag_map, include_tags, include_comments
        )

        lines += ['---', f'*Exported {datetime.now().strftime("%Y-%m-%d %H:%M")}*']
        body = '\n'.join(lines)
        if opts.get('include_prompt', True):
            if opts.get('use_recording_screen_prompt'):
                key = 'export_default_prompt_recording'
                prompt_default = DEFAULT_EXPORT_PROMPT_RECORDING
            else:
                key = 'export_default_prompt'
                prompt_default = DEFAULT_EXPORT_PROMPT
            prompt = (Setting.get(key, prompt_default) or '').strip()
            if prompt:
                body = prompt + '\n\n---\n\n' + body
        return body, None


def export_recording_odt(app, project_id, recording_id, opts):
    """Export to ODT (simplified - HTML-like content in ODT wrapper)."""
    md, err = export_recording_markdown(app, project_id, recording_id, opts)
    if err:
        return None, err
    return _markdown_to_odt(md), None


def export_project_markdown(app, project_id, recording_ids, opts):
    """Export multiple recordings as a single Markdown document."""
    with app.app_context():
        project = db.session.get(Project, project_id)
        if not project:
            return None, 'Project not found'

        root = _projects_root(app)
        project_dir = os.path.join(root, project.folder_name)
        from .annotations import get_project_tags
        tags = get_project_tags(project_dir)
        tag_map = {t['id']: t for t in tags}

        include_comments = opts.get('include_comments', True)
        include_tags = opts.get('include_tags', True)
        include_participant_details = opts.get('include_participant_details', True)
        include_project_details = opts.get('include_project_details', True)
        remove_pii = opts.get('remove_pii', False)

        lines = [f'# {project.name}']
        if include_project_details:
            lines += _build_project_header(project)

        exported = 0
        for rid in recording_ids:
            recording = db.session.get(Recording, rid)
            if not recording or recording.project_id != project_id:
                continue
            if recording.transcription_status != 'transcribed' or not recording.transcript_path:
                continue

            transcript_path = _transcript_file(project_dir, recording.transcript_path)
            if not transcript_path or not os.path.isfile(transcript_path):
                continue

            transcript = atomic_read_json(transcript_path)

            from .annotations import annotation_recording_ref, get_annotations
            ann = get_annotations(project_dir, annotation_recording_ref(recording))
            segments = transcript.get('segments', [])

            speaker_labels = ann.get('speaker_labels', {})
            if speaker_labels:
                for seg in segments:
                    spk = seg.get('speaker', '')
                    if spk and spk in speaker_labels:
                        seg['speaker'] = speaker_labels[spk]

            tag_spans = ann.get('tag_spans', [])
            comments = ann.get('comments', [])
            notes = (recording.participant_notes or '').strip() or '—'
            title = recording.original_name
            if remove_pii:
                segments, tag_spans, comments, notes, title = _remove_pii_for_export(
                    current_app._get_current_object(), recording, segments, tag_spans, comments,
                    notes, exported + 1,
                )

            lines.append(f'# {title} (Transcript)')
            if include_participant_details and not project.is_system:
                seg = db.session.get(Segment, recording.segment_id) if recording.segment_id else None
                segment_name = _segment_export_line(seg)
                lines += ['## Segment', segment_name]
                lines += ['## Additional details', notes]
            lines.append('')

            merged_blocks = _merge_consecutive_speakers(segments)
            lines += _render_transcript_blocks(
                segments, merged_blocks, tag_spans, comments, tag_map, include_tags, include_comments
            )

            lines.append('---')
            lines.append('')
            exported += 1

        if exported == 0:
            return None, 'No recordings could be exported'

        lines += [f'*Exported {datetime.now().strftime("%Y-%m-%d %H:%M")}*']
        body = '\n'.join(lines)
        if opts.get('include_prompt', True):
            prompt = (Setting.get('export_default_prompt', DEFAULT_EXPORT_PROMPT) or '').strip()
            if prompt:
                body = prompt + '\n\n---\n\n' + body
        return body, None


def export_project_odt(app, project_id, recording_ids, opts):
    """Export multiple recordings as a single ODT document."""
    md, err = export_project_markdown(app, project_id, recording_ids, opts)
    if err:
        return None, err
    return _markdown_to_odt(md), None


def _markdown_to_odt(md_text):
    """Convert markdown to minimal ODT (OpenDocument Text) XML."""
    import io
    import zipfile

    content = _md_to_odt_content(md_text)
    manifest = '''<?xml version="1.0" encoding="UTF-8"?>
<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0">
 <manifest:file-entry manifest:media-type="application/vnd.oasis.opendocument.text" manifest:full-path="/"/>
 <manifest:file-entry manifest:media-type="text/xml" manifest:full-path="content.xml"/>
 <manifest:file-entry manifest:media-type="text/xml" manifest:full-path="styles.xml"/>
</manifest:manifest>'''
    styles = '''<?xml version="1.0" encoding="UTF-8"?>
<office:document-styles xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" office:version="1.2">
 <office:styles/>
</office:document-styles>'''
    # Built in memory: the temp file this used to go through was never removed,
    # so every export left a copy of the transcript in the system temp folder.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('mimetype', 'application/vnd.oasis.opendocument.text', zipfile.ZIP_STORED)
        zf.writestr('content.xml', content)
        zf.writestr('styles.xml', styles)
        zf.writestr('META-INF/manifest.xml', manifest)
    return buf.getvalue()


def _parse_bold_to_odt(text):
    """Convert **bold** in text to ODF bold spans. Returns ODF XML string."""
    import re

    def esc(s):
        return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

    parts = re.split(r'\*\*(.+?)\*\*', text)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append(f'<text:span text:style-name="Bold">{esc(part)}</text:span>')
        else:
            out.append(esc(part))
    return ''.join(out)


def _parse_paragraph_to_odt_with_tag_comments(line, ann_counter):
    """Convert paragraph to ODT XML. **text** [tagname] becomes ODF comments (author PINE).
    **text** without tag becomes bold. ann_counter is [n] mutated in place."""
    import re

    def esc(s):
        return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

    # Split by **text** [tagname] first - these become annotations
    # Pattern: **captured_text** [tag_name]
    tag_pattern = re.compile(r'\*\*(.+?)\*\* \[([^\]]+)\]')
    parts = []
    last_end = 0
    for m in tag_pattern.finditer(line):
        if m.start() > last_end:
            # Plain text or **bold** before this match - process for bold only
            segment = line[last_end:m.start()]
            parts.append(('plain', _parse_bold_to_odt(segment)))
        ann_counter[0] += 1
        ann_id = f'ann{ann_counter[0]}'
        tagged_text = esc(m.group(1))
        tag_name = esc(m.group(2))
        date_iso = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        parts.append(('annotation', (ann_id, tagged_text, tag_name, date_iso)))
        last_end = m.end()
    if last_end < len(line):
        segment = line[last_end:]
        parts.append(('plain', _parse_bold_to_odt(segment)))

    out = []
    for ptype, pval in parts:
        if ptype == 'plain':
            out.append(pval)
        else:
            ann_id, tagged_text, tag_name, date_iso = pval
            out.append(
                f'<office:annotation office:name="{ann_id}">'
                f'<dc:creator>PINE</dc:creator>'
                f'<dc:date>{date_iso}</dc:date>'
                f'<text:p>{tag_name}</text:p>'
                f'</office:annotation>'
                f'{tagged_text}'
                f'<office:annotation-end office:name="{ann_id}"/>'
            )
    return ''.join(out)


def _md_to_odt_content(md):
    """Convert markdown to ODF content.xml body. **text** becomes real bold.
    **text** [tagname] becomes ODF comments (author PINE) covering the text."""
    lines = md.split('\n')
    out = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>')
    out.append('<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" office:version="1.2">')
    out.append('<office:automatic-styles><style:style style:name="Bold" style:family="text"><style:text-properties fo:font-weight="bold"/></style:style></office:automatic-styles>')
    out.append('<office:body><office:text>')
    ann_counter = [0]
    for line in lines:
        if line.startswith('# '):
            out.append(f'<text:h text:outline-level="1">{_parse_bold_to_odt(line[2:])}</text:h>')
        elif line.startswith('## '):
            out.append(f'<text:h text:outline-level="2">{_parse_bold_to_odt(line[3:])}</text:h>')
        elif line.startswith('### '):
            out.append(f'<text:h text:outline-level="3">{_parse_bold_to_odt(line[4:])}</text:h>')
        elif line.startswith('  > '):
            out.append(f'<text:p text:style-name="Comment">{_parse_bold_to_odt(line[4:])}</text:p>')
        elif line.strip():
            out.append(f'<text:p>{_parse_paragraph_to_odt_with_tag_comments(line, ann_counter)}</text:p>')
        else:
            out.append('<text:p/>')
    out.append('</office:text></office:body></office:document-content>')
    return '\n'.join(out)
