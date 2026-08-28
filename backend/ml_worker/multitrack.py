"""Transcribing a recording whose speakers already have their own tracks.

The single-file pipeline spends most of its wall time asking pyannote who spoke
when. When the material arrives as one track per speaker — Zoom's per-participant
files, or a multi-channel recording with a mic each — that question is already
answered, and the work reduces to transcribing each track and putting the
results back on one timeline.

Each track goes through the same three steps (``tracks.py`` has the detail):
find the speech, splice it into a short file, transcribe that, map the
timestamps back. The tracks are never mixed; the only thing that merges is the
list of segments at the end, and overlapping speech survives it intact, which is
the case per-track transcription exists for.

Memory sets the shape of the loop. A decoded two-hour track is ~460 MB, so only
one is ever held: decode, compact, release, transcribe. The exception is
multi-channel material, where deciding who owns a frame needs every channel's
loudness at once — those are cheap (one float per 20 ms), so a first pass
collects them and drops each waveform as it goes.
"""

import logging
import os
import shutil
import tempfile
import time

from .audio import load_audio_file, write_wav
from .constants import TRACK_JOIN_GAP_SEC, TRACK_JOIN_MAX_SEC
from .engines import TranscribeContext
from .tracks import (
    SAMPLE_RATE,
    compact,
    frame_db,
    mask_to_regions,
    remap,
    resolve_bleed,
    speech_mask,
    speech_thresholds,
)

log = logging.getLogger(__name__)

# Raw per-track speaker id, before map_speakers turns it into a display name.
RAW_SPEAKER = 'TRACK_{:02d}'


def _noop_status(**kwargs):
    return None


class _TrackProgress:
    """Folds each track's own 0→100 into the single transcribing band.

    Everything an engine reports is forwarded as ``transcribing`` regardless of
    what the engine called it. Alignment is a real stage, but here it happens
    once per track, interleaved with transcription — reporting it as its own
    stage would walk the global bar into the aligning band on track 1 and strand
    it there, because the scale only moves forward.
    """

    def __init__(self, status, weights):
        self._status = status
        self._total = max(len(weights), 1)
        # Cumulative share of the work each track ends at. Tracks are *not* an
        # equal fraction each: in an interview the moderator's track holds a
        # fraction of the speech the participant's does, so counting them 1/N
        # apiece would put the bar at half way after a fifth of the work — and
        # the ETA derived from that milestone would be wrong by the same factor.
        total = float(sum(weights)) or float(self._total)
        self._edges = []
        acc = 0.0
        for w in (weights or [1.0] * self._total):
            acc += (w or 0.0)
            self._edges.append(acc / total * 100.0)
        self._index = 0
        self._label = ''
        self._high = 0.0

    def track(self, index, label):
        self._index = index
        self._label = label
        return self

    def _span(self, index):
        start = self._edges[index - 1] if index > 0 else 0.0
        return start, self._edges[index]

    def _advance(self, pct):
        """Never hand out a figure below one already reported.

        An engine's local scale is its own business — chunked runs restart it,
        and a retry can repeat a chunk. ProgressMapper clamps too, but a
        progress source that walks backwards and relies on someone downstream
        to hide it is a trap for the next caller.
        """
        self._high = max(self._high, min(max(pct, 0.0), 100.0))
        return self._high

    def __call__(self, stage='', message='', percent=None, eta_secs=None, **extra):
        if message and self._label:
            message = f'{self._label} ({self._index + 1}/{self._total}): {message}'
        if percent is None:
            # Engines heartbeat with a message and no percent on long files.
            # Sending a percent anyway would place the bar at the *start* of
            # this track; saying nothing lets the mapper's ticker keep drifting
            # from where the last real milestone put it.
            self._status(stage='transcribing', message=message, **extra)
            return
        local = max(min(float(percent), 100.0), 0.0)
        start, end = self._span(self._index)
        overall = start + (end - start) * local / 100.0
        self._status(stage='transcribing', message=message,
                     percent=self._advance(overall), **extra)

    def done(self, index):
        """Mark a finished track, so the bar moves even for silent engines."""
        self._status(stage='transcribing', message='',
                     percent=self._advance(self._span(index)[1]))


def _regions_with_bleed(specs, check_cancel):
    """Speech regions for channels of one file, with the cross-talk resolved.

    Every mic in a room hears every speaker, so a quiet copy of someone else's
    voice can clear a channel's own gate. Loudness is compared across channels
    to hand each frame to whoever actually produced it.
    """
    levels, masks = [], []
    for spec in specs:
        check_cancel()
        samples = load_audio_file(spec.path, channel=spec.channel)
        db = frame_db(samples)
        del samples
        levels.append(db)
        masks.append(speech_mask(db, speech_thresholds(db)))

    resolved = resolve_bleed(levels, masks)
    return [mask_to_regions(mask) for mask in resolved]


def _speech_regions(samples):
    db = frame_db(samples)
    mask = speech_mask(db, speech_thresholds(db))
    return mask_to_regions(mask, duration=len(samples) / SAMPLE_RATE)


def prepare_tracks(specs, events, work_dir):
    """Decode, gate and compact every track before any of them is transcribed.

    Done as its own pass for two reasons. It is the only way to know each
    track's share of the work before committing to a progress scale — and the
    shares are lopsided, since an interviewer speaks a fraction of what the
    person they are interviewing does. And doing it up front means no track is
    decoded twice: the compacted audio goes to disk, so peak memory is one
    waveform (~460 MB for two hours) rather than all of them.

    Returns ``[{index, label, path, splices, speech_secs}]``, skipping tracks
    that turned out to hold no speech.
    """
    # Channels of one file bleed into each other and have to be compared with
    # each other; separate files do not, so that pass is spent only where it
    # buys something.
    multichannel = any(spec.channel is not None for spec in specs)
    preset_regions = (_regions_with_bleed(specs, events.check_cancel)
                      if multichannel else None)

    prepared = []
    for i, spec in enumerate(specs):
        events.check_cancel()
        label = spec.speaker_name or f'Track {i + 1}'
        events.status(stage='loading',
                      message=f'Finding speech in {label} ({i + 1}/{len(specs)})…')

        samples = load_audio_file(spec.path, channel=spec.channel)
        regions = (preset_regions[i] if preset_regions is not None
                   else _speech_regions(samples))
        short, splices = compact(samples, regions)
        del samples

        speech_secs = len(short) / SAMPLE_RATE
        log.info('Track %d (%s): %.0fs of speech in %d regions',
                 i, label, speech_secs, len(splices))
        if not splices:
            log.info('Track %d (%s) has no speech — skipping', i, label)
            del short
            continue

        path = os.path.join(work_dir, f'track_{i:02d}.wav')
        write_wav(path, short, SAMPLE_RATE)
        del short
        prepared.append({'index': i, 'label': label, 'path': path,
                         'splices': splices, 'speech_secs': speech_secs})

    return prepared


def join_runs(segments, max_gap=TRACK_JOIN_GAP_SEC, max_len=TRACK_JOIN_MAX_SEC):
    """Put back together the turns the speech gate cut in two.

    Every pause longer than the gate's own tolerance ends a speech region, and
    ``remap`` ends a segment wherever a region does — so one person drawing
    breath mid-sentence arrives as two lines in the transcript. Neighbours from
    the same speaker within ``max_gap`` are joined back up to ``max_len``.

    Only neighbours in the sorted list are candidates: anything between them is
    somebody else talking, and joining across it would put a segment on top of a
    turn it does not contain. Words move with the text, since the transcript is
    rebuilt from them; a segment carrying words is never joined to one without.
    """
    out = []
    for seg in segments:
        prev = out[-1] if out else None
        if (prev is not None
                and prev.get('speaker') == seg.get('speaker')
                and bool(prev.get('words')) == bool(seg.get('words'))
                and float(seg['start']) - float(prev['end']) <= max_gap
                and float(seg['end']) - float(prev['start']) <= max_len):
            prev['end'] = seg['end']
            prev['text'] = ' '.join(
                part for part in (str(prev.get('text', '') or '').strip(),
                                  str(seg.get('text', '') or '').strip())
                if part)
            if seg.get('words'):
                prev['words'] = list(prev.get('words') or []) + list(seg['words'])
            continue
        out.append(dict(seg))
    return out


def run_multitrack(engine, specs, language, events, work_dir=None):
    """Transcribe every track; returns ``(segments, speaker_names, language)``.

    ``segments`` are in the unified format on the original timeline, sorted by
    start, carrying raw ``TRACK_NN`` speaker ids. ``speaker_names`` maps those
    ids to the names the tracks came in with, for ``map_speakers``.
    """
    if not specs:
        return [], {}, language

    owned_dir = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix='pine_tracks_')
    names = {RAW_SPEAKER.format(i): (spec.speaker_name or f'Track {i + 1}')
             for i, spec in enumerate(specs)}

    merged = []
    try:
        prepared = prepare_tracks(specs, events, work_dir)
        progress = _TrackProgress(
            events.status, [p['speech_secs'] for p in prepared])
        log.info('Multitrack: %.0fs of speech across %d of %d tracks',
                 sum(p['speech_secs'] for p in prepared), len(prepared),
                 len(specs))

        # Every track handed over below has had its silence removed and its
        # turns spliced together. An engine with its own VAD has to know that:
        # at single-file settings the splice gap is too short to register, and
        # it will run one turn into another recorded minutes apart.
        engine.presegmented = True

        for slot, item in enumerate(prepared):
            events.check_cancel()
            raw_id = RAW_SPEAKER.format(item['index'])
            label = item['label']
            speech_secs = item['speech_secs']

            # The exact cuts for this track, so an engine that would otherwise
            # run its own VAD does not have to guess them back.
            engine.presegmented_regions = item['splices']

            t0 = time.monotonic()
            ctx = TranscribeContext(
                on_status=progress.track(slot, label),
                check_cancel=events.check_cancel,
            )
            try:
                out = engine.transcribe(item['path'], speech_secs, language, ctx)
            finally:
                _unlink(item['path'])

            # Only the single-file path reuses the decoded waveform (for
            # diarization); here it is just memory held per track.
            out.audio = None

            if not language and out.language:
                # Lock the language in after the first track that detected one.
                # Left to themselves the tracks detect independently, and one
                # conversation coming back in two languages is worse than one
                # track being transcribed in the wrong one.
                language = out.language
                log.info('Multitrack language locked to %s after track %d',
                         language, item['index'])

            for seg in remap(out.segments, item['splices']):
                seg['speaker'] = raw_id
                merged.append(seg)

            log.info('PERF: track %d (%s) took %.1fs for %.0fs of speech',
                     item['index'], label, time.monotonic() - t0, speech_secs)
            progress.done(slot)
    finally:
        if owned_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

    merged.sort(key=lambda s: (s['start'], s.get('speaker', '')))
    return join_runs(merged), names, language


def _unlink(path):
    try:
        os.unlink(path)
    except OSError:
        pass
