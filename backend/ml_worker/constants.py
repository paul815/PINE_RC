"""Tunables shared across the ML pipeline.

The numbers here are measured, not guessed — each one carries the observation it
came from — and they are literals on purpose. Every threshold used to also read
a ``PINE_*`` environment variable, thirty-six of them, none named anywhere else
in the repo: not in a test, not in the docs, not in a launcher. They were a
tuning session that ended, left behind as a promise the code no longer keeps, so
a reader had to check each default twice to learn it was the only value in play.
Retune by changing the value and saying what you measured.

Three overrides survive, because something outside this file knows them:

* ``PINE_PARALLEL_STAGES`` and ``PINE_DIARIZE_DEVICE`` (the latter lives in
  ``diarize.py``) — documentation/ARCHITECTURE.md tells you to measure with them.
* ``PINE_LANG_CONFIDENCE_MIN`` — documented in documentation/API.md.
"""

import os

# Chunked processing for long audio files (>30 min)
CHUNK_THRESHOLD_SEC = 1800  # switch to chunked if >= 30 min
CHUNK_SIZE_SEC = 1800       # each chunk = 30 min
CHUNK_OVERLAP_SEC = 30      # overlap to avoid cutting mid-word

# Run diarization alongside transcription instead of after it. pyannote reads
# only the audio — the transcript is needed by the speaker *assignment* step,
# which costs ~0s — so the two stages have no real dependency. Serially they
# cost STT + diarize; overlapped they cost roughly max(STT, diarize). On the
# 74-min reference recording that is 1762.8s -> ~985s.
#
# The win depends on the two stages not fighting over one accelerator: with
# Whisper on a Mac's one GPU, diarization wants to be somewhere else. Measure the
# combinations with PINE_PARALLEL_STAGES and PINE_DIARIZE_DEVICE before
# assuming a default fits.
#
# Default is off. On CUDA both stages land on the same card by default, and a
# 12 GB GPU running whisperx float16 (batch_size 16) next to pyannote spends
# 6+ minutes with both resident: the machine becomes unusable while the driver
# pages VRAM back to host RAM, and the first chunk's status line sits still for
# the whole of it. Sequential stages cost more wall clock and give it back in
# a responsive desktop. Set PINE_PARALLEL_STAGES=1 to overlap them again --
# worth it when diarization is on a different device (PINE_DIARIZE_DEVICE=cpu)
# or the GPU has headroom to spare.
#
# A Mac running mlx-whisper looked like the exception, and is not. With pyannote
# on MPS as well (2026-07-19, the same 74-min recording) the two queued on one
# GPU: STT nearly doubled and the whole job gained 17%. With pyannote on the
# CPU instead (2026-09-24, an 88-minute interview) STT kept its pace, 1081s
# against 1051s serially, but diarization had not finished 33 minutes after it
# started — MPS does the same file in 307s after STT — and the progress bar sat
# at 99% the whole time, since a stage running underneath reports nothing. So
# the stages run in turn everywhere; measure before turning this on.
_PARALLEL_ENV = os.environ.get('PINE_PARALLEL_STAGES', '').strip()
PARALLEL_STAGES = _PARALLEL_ENV not in ('', '0')

# Progress weighting: expected cost of each stage as a multiple of the audio
# duration. These only set how the single 0→100 scale is divided between
# stages, so being off by a bit costs pacing, not correctness. Measured on
# whisperx/CUDA over ~75-min recordings: STT 0.032–0.084x realtime (the spread
# is model size and GPU), diarization a steady ~0.019x.
#
# The Mac's former mlx-whisper/pyannote path sat an order of magnitude higher and
# in the opposite order — measured on a 74-min recording: STT 0.174x,
# diarization 0.220x. These defaults are therefore wrong for that path, but
# only for pacing: ProgressMapper re-derives the job's pace from its own
# milestones, so the plan is a starting prior, not the ETA. Retuning them per
# platform is a separate change, and would want more than one machine.
#
# These stay as shipped. What corrects them is measurement: every finished job
# reports how far off the plan was for this machine, model and mode, and the
# next job starts from that instead (see ProgressMapper.observed_scale and
# `progress_scale:*` in Settings). So the numbers below only have to be a
# sensible starting point for the very first recording — the alternative,
# hand-tuning them per platform, needs a machine of every kind and goes stale
# the moment anyone changes model.
PROGRESS_RTF_TRANSCRIBE = 0.045
PROGRESS_RTF_ALIGN = 0.010
PROGRESS_RTF_DIARIZE = 0.019
# Bounds on the learned correction, so one pathological run (a machine that
# went to sleep mid-job) cannot poison the estimate for every job after it.
PROGRESS_SCALE_MIN = 0.2
PROGRESS_SCALE_MAX = 5.0
# Weight of the newest measurement against the running one.
PROGRESS_SCALE_SMOOTHING = 0.4
# Below this much audio the flat costs dominate and the ratio measures disk
# cache rather than transcription speed, so nothing is learned from it.
PROGRESS_LEARN_MIN_SEC = 120.0
# Flat costs that don't scale with audio length.
PROGRESS_LOAD_SEC = 10.0
PROGRESS_FINALIZE_SEC = 2.0
# How often the interpolating ticker refreshes between real milestones.
PROGRESS_TICK_SEC = 2.0
# Key the measured pace travels under inside the transcript payload. Underscored
# because it is not part of the transcript — ``_finalize`` pops it before the
# file is written. It lives here rather than beside the pipeline so the Flask
# side can read it without importing the pipeline, and through it torch: that
# import ran once with a stale ``constants`` still in ``sys.modules`` and cost
# a finished 40-minute transcript.
PROGRESS_SCALE_KEY = '_progress_scale'

# ── Per-speaker tracks (see tracks.py) ──
#
# Voice activity detection on one speaker's own track, used to cut the pauses
# out before transcription. Tuned for interviews rather than for a general VAD:
# the thing most easily lost here is back-channel — "угу", "да", "понятно" —
# which runs 0.3-0.8s and would be dropped by the 0.5s minimum a general-purpose
# gate uses. Losing it costs the moderator's half of the conversation, so the
# minimum sits at 0.2s and short gaps are closed before anything is discarded.
VAD_FRAME_SEC = 0.02
VAD_MIN_SPEECH_SEC = 0.2
VAD_MERGE_GAP_SEC = 0.5
VAD_PAD_SEC = 0.25
# Thresholds are a fraction of each track's own quiet-to-loud range, so tracks
# recorded at different levels are each measured against themselves.
VAD_OPEN_FRACTION = 0.35
VAD_CLOSE_FRACTION = 0.20
VAD_MIN_MARGIN_DB = 6.0
# Digital silence is -inf dB; without a floor the range is unbounded and every
# threshold derived from it collapses. Also the level below which a track with
# no loud/quiet structure is read as empty rather than as speech throughout.
VAD_SILENCE_FLOOR_DB = -80.0
VAD_MIN_DYNAMIC_DB = 6.0
# How far one track must lead the others to be given a frame outright. Only
# bites on multi-channel recordings where the mics hear each other.
VAD_DOMINANCE_DB = 6.0
# Silence inserted between two spliced-together regions, so the model does not
# run a turn from minute 3 into a turn from minute 40.
VAD_COMPACT_GAP_SEC = 0.2
# Cutting the pauses out also cuts the transcript: a turn spoken across two
# speech regions comes back as two segments. Neighbours from one speaker no
# further apart than this are put back together — wide enough to close the
# breath the gate opened on, short enough to leave a real handover alone.
TRACK_JOIN_GAP_SEC = 1.5
# ...but only up to here. Someone talking steadily for ten minutes would
# otherwise arrive as one block nobody can scroll past or click into.
TRACK_JOIN_MAX_SEC = 30.0
# Per-track gate, for the multitrack path only (ml_worker/multitrack.py).
# Every region boundary there is a seam the transcript can tear along: whisperx
# reads the compacted track as continuous and the aligner spreads words over
# silence that was never spoken. The general-purpose 0.5s only closes the gaps
# inside a sentence, which cut a 40-minute interview into 330 regions; two
# seconds closes the breath between sentences as well, for ~5% more audio to
# decode.
TRACK_VAD_MERGE_GAP_SEC = 2.0
TRACK_VAD_PAD_SEC = 0.5
# Cutting the pauses out also moves the words: whisperx aligns against audio
# whose seams it cannot see, and a word landing a fifth of a second the wrong
# side of one is remapped tens of seconds away from the rest of its sentence.
# A lone word this close behind the previous one was never a second utterance —
# a real one arrives after a pause — so it is kept with the words it belongs to.
# Twice the inserted gap: nothing that follows a seam can be closer than one.
TRACK_SEAM_CONTIGUOUS_SEC = 0.4

# ── Dropping what Whisper wrote over non-speech (see pipeline.py) ──
#
# Whisper learned from YouTube subtitles, and those end on a credit over the
# closing silence. So where a recording holds a pause, noise or speech too
# garbled to read, the model sometimes writes the credit down. One whisperx
# interview came back with "Субтитры создавал DimaTorzok" in a five-minute gap
# no speaker was found in, and "Продолжение следует." twice — once stretched
# over 20s, once over the 7s the moderator then said had been inaudible. The
# VAD passes noise as speech, so neither engine is spared.
#
# Matched against the whole segment, after lowercasing, ё→е and stripping
# punctuation — never as a substring, so an answer that quotes one survives.
#
# Credits: nobody says these outside a subtitle file, so they always go.
HALLUCINATION_SIGNATURES = (
    r'субтитры (создавал|создал|сделал|делал|подготовил|подогнал)\b.*',
    r'(редактор|корректор) субтитров\b.*',
    r'субтитры (by|от)\b.*',
    r'.*\b(amara org|dimatorzok)\b.*',
)
# Phrases a person can actually say. These go only with a second sign that
# nobody said them: no speaker found under them, or words stretched over the
# silence — see HALLUCINATION_STRETCH_SEC_PER_WORD.
HALLUCINATION_PHRASES = frozenset({
    'продолжение следует',
    'спасибо за просмотр',
    'подписывайтесь на канал',
    'звук колокола',
    'thanks for watching',
})
# Speech runs ~0.3s a word. The two stretched credits above ran 10 and 3.6s;
# five times normal pace is far from both.
HALLUCINATION_STRETCH_SEC_PER_WORD = 1.5

# Splitting a segment where its words change speaker (pipeline.py,
# split_segments_at_speaker_changes). Each sentence goes whole to whoever said
# most of its words; a sentence is cut only where both sides keep at least this
# many words — in practice a stretch Whisper left unpunctuated. Measured on the
# 2026-11-21 recordings: at 5, "Which means that it would / be O of n of n."
# was cut mid-sentence; the run that needs a cut (SWE 60:08) is 40+ words a side.
SPEAKER_SPLIT_MIN_WORDS_IN_SENTENCE = 8

# Re-reading a window Whisper wrote without punctuation (ml_worker/punctuation.py).
# large-v3 in batched mode now and then drops every mark and capital for a whole
# 30-second window — "and so on so yeah hyundai is definitely a place i would uh
# recommend…" — 11% of the IELTS test transcript, 9% of the SWE one (2026-11-21).
# Such a window is decoded once more on its own, prompted with the transcript's
# own preceding text. On those two files that restored 15 of 17 windows, at
# 93–98% word agreement — the rest mostly fillers ("uh", "you know") the rest of
# each transcript leaves out too. A re-read outside these limits is discarded.
UNPUNCTUATED_MIN_WORDS = 25        # shorter text can lack a mark honestly
UNPUNCTUATED_MAX_MARK_RATIO = 0.03  # punctuated speech runs 10-25 marks per 100 words
REREAD_PROMPT_CHARS = 240          # ~60 tokens; Whisper's prompt allows 223
REREAD_MIN_SIMILARITY = 0.80       # measured worst accepted: 0.88
REREAD_MAX_LENGTH_CHANGE = 0.30    # word count may move this far, either way

SPEAKER_LABELS = [
    'Moderator',
    'Participant 1',
    'Participant 2',
    'Participant 3',
    'Participant 4',
    'Participant 5',
]

# Min probability on the language probe to skip asking the user (WhisperX: top-1 only).
LANG_CONFIDENCE_MIN = float(os.environ.get('PINE_LANG_CONFIDENCE_MIN', '0.82'))

# Language-specific WhisperX align-model overrides.
# These repos include tokenizer metadata needed by recent transformers/hf_hub stacks.
ALIGN_MODEL_CANDIDATES = {
    'ru': [
        'anton-l/wav2vec2-large-xlsr-53-russian',
        'jonatasgrosman/wav2vec2-large-xlsr-53-russian',
    ],
}
