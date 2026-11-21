"""Tier 2 tests: Mocked transcription pipeline — speaker assignment, chunking,
cancellation, requeue, and status flow.

The ML pipeline lives in the Flask-free ``ml_worker`` package; queue/cancel/
status orchestration stays in ``app.services.transcription``. Tests below
target whichever side owns the logic.
"""

import os
import sys
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

def _make_diarizer(device='cpu', pipeline=None):
    """Create a Diarizer without loading any ML models."""
    from ml_worker.diarize import Diarizer
    d = Diarizer(env=None, device=device)
    d.pipeline = pipeline
    return d


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


class TestJoinWords:
    """Whisper spells "как-то" as two word tokens and tells them apart by the
    leading space on the first. The Mac's former mlx engine stripped that space so
    the recording UI could find each word in the segment text, and the plain join that used to
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
            patch.multiple('ml_worker.engines.whisperx_engine',
                           free_vram_gb=MagicMock(return_value=8.0),
                           driver_free_vram_gb=MagicMock(return_value=None)),
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


class TestDecodeBatchSize:
    """Whisper's batch comes from the VRAM the driver says is free, not torch's figure.

    Measured 2026-11-18 on a 12 GB card under WDDM: torch reported 7.1 GB free while
    nvidia-smi showed 5.1 GB; batch 16 needs 5.9 GB on top of the model, so it paged
    and a 30-min chunk took 275-851s instead of ~30s.
    """

    @staticmethod
    def _engine(device='cuda'):
        from ml_worker.engines.whisperx_engine import WhisperXEngine
        eng = WhisperXEngine(env=None)
        eng._device = device
        return eng

    def test_driver_figure_wins_over_torch(self):
        eng = self._engine()
        with patch('ml_worker.engines.whisperx_engine.driver_free_vram_gb', return_value=5.1), \
                patch('ml_worker.engines.whisperx_engine.free_vram_gb', return_value=7.1):
            assert eng._pick_batch_size() == 8

    @pytest.mark.parametrize('free_gb, batch', [
        (11.0, 16), (7.0, 16), (6.9, 8), (4.5, 8), (4.4, 4), (2.6, 4), (2.5, 2), (0.3, 2),
    ])
    def test_batch_leaves_a_gigabyte_beside_the_decode(self, free_gb, batch):
        eng = self._engine()
        with patch('ml_worker.engines.whisperx_engine.driver_free_vram_gb', return_value=free_gb), \
                patch('ml_worker.engines.whisperx_engine.free_vram_gb', return_value=99.0):
            assert eng._pick_batch_size() == batch

    def test_torch_figure_is_the_fallback(self):
        eng = self._engine()
        with patch('ml_worker.engines.whisperx_engine.driver_free_vram_gb', return_value=None), \
                patch('ml_worker.engines.whisperx_engine.free_vram_gb', return_value=5.0):
            assert eng._pick_batch_size() == 8

    def test_cpu_never_asks_the_driver(self):
        eng = self._engine('cpu')
        with patch('ml_worker.engines.whisperx_engine.driver_free_vram_gb') as driver:
            assert eng._pick_batch_size() == 4
        driver.assert_not_called()


class TestMetal:
    """WhisperX on Apple Silicon's GPU, through CTranslate2's Metal backend (4.9+).

    mac-bench, 2026-11-21 (M4 Pro, 24 GB, 88-min Russian interview): 99.9% of
    words as on Windows; batch 32 no faster than 16; alignment grew 27->42->55s
    per chunk until the MPS cache was handed back; the idle worker held 15 GB.
    """

    @staticmethod
    def _engine(device='mps'):
        from ml_worker.engines.whisperx_engine import WhisperXEngine
        eng = WhisperXEngine(env=None)
        eng._device = device
        return eng

    @pytest.mark.parametrize('platform, version, count, expected', [
        ('darwin', '4.9.0', 1, True),
        ('darwin', '4.10.2', 1, True),
        ('darwin', '4.8.1', 1, False),
        ('darwin', '4.9.0', 0, False),
        ('win32', '4.9.0', 1, False),
    ])
    def test_metal_needs_ctranslate2_4_9_and_a_device(self, platform, version, count, expected):
        from ml_worker.engines import whisperx_engine as wx
        fake = MagicMock(__version__=version, get_mps_device_count=MagicMock(return_value=count))
        with patch.object(wx, '_ct2_metal', None), \
                patch.object(wx.sys, 'platform', platform), \
                patch.dict('sys.modules', {'ctranslate2': fake}):
            assert wx.ct2_metal_available() is expected

    def test_mac_device_is_metal_float16_or_cpu_int8(self):
        from ml_worker.engines import whisperx_engine as wx
        with patch.object(wx, 'ct2_metal_available', return_value=True):
            assert wx.detect_mac_device() == ('mps', 'float16')
        with patch.object(wx, 'ct2_metal_available', return_value=False):
            assert wx.detect_mac_device() == ('cpu', 'int8')

    def test_metal_that_will_not_load_falls_back_to_cpu(self):
        eng = self._engine()
        eng._compute_type = 'float16'
        eng.env = MagicMock(model_dir='/models/whisperx-large-v3')
        calls = []

        def load_model(model_dir, device, compute_type=None):
            calls.append((device, compute_type))
            if device == 'mps':
                raise RuntimeError('Metal device not supported')
            return MagicMock()

        with patch.dict('sys.modules', {'whisperx': MagicMock(load_model=load_model)}):
            eng.load()
        assert calls == [('mps', 'float16'), ('cpu', 'int8')]
        assert (eng._device, eng._compute_type) == ('cpu', 'int8')

    @pytest.mark.parametrize('ram_gb, batch', [(48, 16), (24, 16), (18, 16), (16, 8), (8, 8)])
    def test_batch_follows_the_macs_ram(self, ram_gb, batch):
        eng = self._engine()
        vm = MagicMock(total=ram_gb * 1024 ** 3)
        with patch.dict('sys.modules', {'psutil': MagicMock(virtual_memory=MagicMock(return_value=vm))}), \
                patch('ml_worker.engines.whisperx_engine.driver_free_vram_gb') as driver:
            assert eng._pick_batch_size() == batch
        driver.assert_not_called()

    @pytest.mark.parametrize('message, device, oom', [
        ('CUDA out of memory. Tried to allocate 2 GiB', 'cuda', True),
        ('failed to allocate buffer', 'cuda', False),     # CUDA keeps its one test
        ('MPS backend out of memory (MPS allocated: 18 GB)', 'mps', True),
        ('Failed to allocate Metal buffer of 2147483648 bytes', 'mps', True),
        ('Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)', 'mps', True),
        ('aten::_ctc_loss is not implemented for MPS', 'mps', False),
    ])
    def test_what_counts_as_out_of_memory(self, message, device, oom):
        from ml_worker.engines.whisperx_engine import _is_out_of_memory
        assert _is_out_of_memory(RuntimeError(message), device) is oom

    def test_metal_oom_halves_the_batch(self):
        eng = self._engine()
        batches = []

        def transcribe(audio, **kwargs):
            batches.append(kwargs['batch_size'])
            if len(batches) == 1:
                raise RuntimeError('MPS backend out of memory (MPS allocated: 18 GB)')
            return {'segments': []}

        eng._model = MagicMock(transcribe=transcribe)
        fake_torch = MagicMock()
        with patch.dict('sys.modules', {'torch': fake_torch}), \
                patch.object(eng, '_pick_batch_size', return_value=16):
            assert eng._transcribe_with_oom_retry(object()) == {'segments': []}
        assert batches == [16, 8]
        fake_torch.mps.empty_cache.assert_called()
        fake_torch.cuda.empty_cache.assert_not_called()

    def test_metal_aligns_on_the_cpu_and_hands_the_mps_cache_back(self):
        """wav2vec2 on MPS drove a 30-min chunk to 17.6 GB of Metal memory and
        kept 12.5 GB of it; the next chunk then ran out (2026-11-21)."""
        from ml_worker.engines.base import TranscribeContext

        eng, seen, patches = TestAlignModelParking._harness('mps')
        with patches[0], patches[1], patches[2]:
            fake_torch = sys.modules['torch']
            eng.transcribe('rec.wav', 2000, None, TranscribeContext())

        assert seen['decode'] == [None, 'cpu']
        assert seen['align'] == ['cpu', 'cpu']
        assert eng._cached_align_model.device == 'cpu'
        assert eng._cached_align_model.moves == ['cpu', 'cpu']   # parked, never sent to MPS
        fake_torch.mps.empty_cache.assert_called()
        fake_torch.cuda.empty_cache.assert_not_called()

    def test_alignment_failure_propagates_to_the_caller(self):
        class FakeWhisperX:
            @staticmethod
            def load_align_model(language_code, device, model_name=None):
                return MagicMock(), {'language': language_code}

            @staticmethod
            def align(segments, model, metadata, audio, device, return_char_alignments=False):
                raise NotImplementedError('op not implemented')

        for device in ('mps', 'cuda'):
            eng = self._engine(device)
            with patch.dict('sys.modules', {'torch': MagicMock()}), \
                    pytest.raises(NotImplementedError):
                eng._align(FakeWhisperX, [{'text': 'а'}], 'ru', object())

    def test_release_takes_whisper_off_metal_and_the_next_job_puts_it_back(self):
        eng = self._engine()
        ct2 = MagicMock(model_is_loaded=True)
        eng._model = MagicMock()
        eng._model.model.model = ct2
        fake_torch = MagicMock()
        with patch.dict('sys.modules', {'torch': fake_torch}):
            eng.release()
        ct2.unload_model.assert_called_once()
        fake_torch.mps.empty_cache.assert_called()

        ct2.model_is_loaded = False
        eng._ensure_resident()
        ct2.load_model.assert_called_once()

    @pytest.mark.parametrize('device', ['cuda', 'cpu'])
    def test_release_changes_nothing_off_metal(self, device):
        eng = self._engine(device)
        ct2 = MagicMock(model_is_loaded=True)
        eng._model = MagicMock()
        eng._model.model.model = ct2
        fake_torch = MagicMock()
        with patch.dict('sys.modules', {'torch': fake_torch}):
            eng.release()
        ct2.unload_model.assert_not_called()
        fake_torch.mps.empty_cache.assert_not_called()
        fake_torch.cuda.empty_cache.assert_not_called()

    def test_pipeline_releases_the_engine_after_every_job(self):
        from ml_worker import pipeline as P
        pl = P.MLPipeline()
        pl._engine = MagicMock()
        env = MagicMock(hf_offline=False, progress_scale=1.0)
        job = MagicMock(tracks=None, audio_path='rec.wav')
        with patch.object(P, 'get_duration_secs', return_value=60.0), \
                patch.object(P, 'ProgressMapper'), \
                patch.object(P.compat, 'apply_hf_offline'), \
                patch.object(P.MLPipeline, '_run', side_effect=ValueError('job failed')), \
                pytest.raises(ValueError):
            pl.run(env, job, P.PipelineEvents())
        pl._engine.release.assert_called_once()

    def test_a_failing_release_never_fails_the_job(self):
        from ml_worker import pipeline as P
        pl = P.MLPipeline()
        pl._engine = MagicMock(release=MagicMock(side_effect=RuntimeError('metal')))
        pl._release_engine()


class TestDriverFreeVram:
    """nvidia-smi parsing — matched by UUID, so CUDA_VISIBLE_DEVICES cannot mix up cards."""

    SMI = ('GPU-11111111-2222-3333-4444-555555555555, 900, 24576\n'
           'GPU-abe9dd30-4cd2-5141-6d8d-11799b512f55, 7081, 12282\n')

    def test_reads_the_card_torch_is_using(self):
        from ml_worker.engines.whisperx_engine import parse_nvidia_smi_free_gb
        free = parse_nvidia_smi_free_gb(self.SMI, 'abe9dd30-4cd2-5141-6d8d-11799b512f55')
        assert free == pytest.approx((12282 - 7081) / 1024)

    def test_unknown_card_or_garbage_gives_none(self):
        from ml_worker.engines.whisperx_engine import parse_nvidia_smi_free_gb
        assert parse_nvidia_smi_free_gb(self.SMI, 'deadbeef-0000-0000-0000-000000000000') is None
        assert parse_nvidia_smi_free_gb('NVIDIA-SMI has failed\n', 'abe9dd30') is None
        assert parse_nvidia_smi_free_gb('GPU-abe9dd30, [N/A], 12282\n', 'abe9dd30') is None

    def test_missing_nvidia_smi_gives_none(self):
        from ml_worker.engines.whisperx_engine import driver_free_vram_gb

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.get_device_properties.return_value.uuid = 'abe9dd30'
        with patch.dict('sys.modules', {'torch': fake_torch}), \
                patch('ml_worker.engines.whisperx_engine.subprocess.run',
                      side_effect=FileNotFoundError('nvidia-smi')):
            assert driver_free_vram_gb() is None


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


class TestSpeakerCountThreading:
    """Per-recording speaker count is forwarded into the diarization pipeline."""

    def test_speaker_kwargs_exact_and_fallback(self):
        from ml_worker.diarize import speaker_kwargs
        assert speaker_kwargs(2) == {'num_speakers': 2}
        assert speaker_kwargs(1) == {'num_speakers': 1}
        assert speaker_kwargs(None) == {'min_speakers': 2, 'max_speakers': 4}
        assert speaker_kwargs(0) == {'min_speakers': 2, 'max_speakers': 4}

    def _compute_with_capturing_pipe(self, captured, **kwargs):
        """Run Diarizer.compute through whisperx's path with a pipeline that records its kwargs."""
        class CapturePipe:
            model = MagicMock()

            def __call__(self, audio_input, **pipe_kwargs):
                captured.update(pipe_kwargs)
                return 'turns'   # no itertracks: compute hands it back as is

        d = _make_diarizer(device='cpu', pipeline=CapturePipe())
        dummy_audio = {"waveform": __import__('torch').zeros(1, 16000),
                       "sample_rate": 16000}
        # compute imports pandas for the DataFrame conversion it skips here.
        with patch.dict('sys.modules', {'pandas': MagicMock()}):
            return d.compute(dummy_audio, 1, **kwargs)

    @needs_torch
    def test_num_speakers_passed_to_pipeline(self):
        captured = {}
        assert self._compute_with_capturing_pipe(captured, num_speakers=2) == 'turns'
        assert captured == {'num_speakers': 2}

    @needs_torch
    def test_no_count_falls_back_to_min_max(self):
        captured = {}
        self._compute_with_capturing_pipe(captured)
        assert 'num_speakers' not in captured
        assert 'min_speakers' in captured and 'max_speakers' in captured


# ---------------------------------------------------------------------------
# Mac pipeline: model choice
# ---------------------------------------------------------------------------

class TestMacModelChoice:
    """The Mac downloads and runs the Windows model, on Metal.

    mac-bench, 2026-11-21 (88-min interview, M4 Pro): WhisperX on Metal matched
    the Windows transcript in 99.9% of words; mlx-whisper, the Mac's model
    until then, in 94.4% with 8 holes. mlx is retired on the Mac.
    """

    @patch('app.services.model_manager.IS_MAC', True)
    def test_model_on_mac(self):
        from app.services.model_manager import get_models_for_setup
        ids = get_models_for_setup(modules=[])
        assert 'whisperx-large-v3' in ids
        assert 'mlx-whisper-large-v3' not in ids

    @patch('app.services.model_manager.IS_MAC', True)
    def test_normalize_maps_the_retired_mlx_ids_to_whisperx_on_mac(self):
        from app.services.model_manager import normalize_stt_model_id
        assert normalize_stt_model_id('whisperx-large-v3') == 'whisperx-large-v3'
        assert normalize_stt_model_id('mlx-whisper-large-v3') == 'whisperx-large-v3'
        assert normalize_stt_model_id('mlx-whisper-large-v3-turbo') == 'whisperx-large-v3'

    @patch('app.services.model_manager.IS_MAC', True)
    def test_whisperx_model_is_registered_on_mac_and_mlx_is_gone(self):
        from app.services.model_manager import MODEL_REGISTRY, _model_for_platform
        assert _model_for_platform(MODEL_REGISTRY['whisperx-large-v3'])
        assert 'mlx-whisper-large-v3' not in MODEL_REGISTRY

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
    def test_check_ml_deps_asks_for_whisperx_on_mac(self, mock_is_installed):
        from app.services.pip_installer import check_ml_deps
        mock_is_installed.return_value = False
        deps = check_ml_deps()
        assert 'whisperx' in deps
        assert 'mlx-whisper' not in deps

    @pytest.mark.skipif(sys.platform != 'darwin', reason='the install list is chosen at import')
    def test_mac_installs_the_measured_stack(self):
        from app.services.pip_installer import PIP_INSTALL_TARGETS, _pinned
        for spec in ('ctranslate2==4.9.0', 'whisperx==3.8.6', 'faster-whisper==1.2.1',
                     'torch==2.8.0', 'pyannote-audio'):
            assert spec in PIP_INSTALL_TARGETS
        assert not any(t.startswith('mlx') for t in PIP_INSTALL_TARGETS)
        assert _pinned('whisperx') == 'whisperx==3.8.6'
        assert _pinned('pyannote-audio') == 'pyannote-audio'
        assert _pinned('not-pinned') == 'not-pinned'


class TestNltkPunkt:
    """whisperx's alignment needs NLTK's punkt_tab and fetches it through urllib.

    On a Mac with python.org's Python (no CA bundle) that fetch failed with
    CERTIFICATE_VERIFY_FAILED and the job ran without word timestamps
    (mac-bench, 2026-11-21). Onboarding fetches it with certifi's bundle.
    """

    def test_present_means_no_download(self, tmp_path, monkeypatch):
        from app.services import pip_installer as pi
        (tmp_path / 'tokenizers' / 'punkt_tab').mkdir(parents=True)
        monkeypatch.setattr(pi, '_nltk_data_dir', lambda: str(tmp_path))
        with patch.object(pi.subprocess, 'run') as run:
            assert pi.ensure_nltk_punkt() is True
        run.assert_not_called()

    def test_download_carries_a_ca_bundle(self, tmp_path, monkeypatch):
        from app.services import pip_installer as pi
        monkeypatch.setattr(pi, '_nltk_data_dir', lambda: str(tmp_path))
        monkeypatch.delenv('PINE_TESTING', raising=False)
        monkeypatch.delenv('SSL_CERT_FILE', raising=False)
        seen = {}

        def fake_run(cmd, env=None, **kwargs):
            seen['cmd'], seen['env'] = cmd, env
            (tmp_path / 'tokenizers' / 'punkt_tab').mkdir(parents=True)
            return MagicMock(returncode=0)

        monkeypatch.setattr(pi.subprocess, 'run', fake_run)
        assert pi.ensure_nltk_punkt() is True
        assert seen['cmd'][-1] == str(tmp_path)
        assert seen['env'].get('SSL_CERT_FILE')

    def test_failed_download_is_reported_not_raised(self, tmp_path, monkeypatch):
        import subprocess

        from app.services import pip_installer as pi
        monkeypatch.setattr(pi, '_nltk_data_dir', lambda: str(tmp_path))
        monkeypatch.delenv('PINE_TESTING', raising=False)

        def fake_run(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd, stderr='CERTIFICATE_VERIFY_FAILED')

        monkeypatch.setattr(pi.subprocess, 'run', fake_run)
        assert pi.ensure_nltk_punkt() is False

    def test_worker_gets_a_ca_bundle_on_a_mac(self, monkeypatch):
        from ml_worker import compat
        monkeypatch.setenv('SSL_CERT_FILE', '')
        monkeypatch.setattr(compat.sys, 'platform', 'darwin')
        compat.ensure_ca_bundle()
        assert os.environ['SSL_CERT_FILE']

    def test_web_process_gets_a_ca_bundle_on_a_mac(self, monkeypatch):
        """The alignment preload in onboarding failed with CERTIFICATE_VERIFY_FAILED
        when the bundle was set only just before it: urllib keeps the HTTPS
        context of its first request (2026-11-21). So it is set at import."""
        import app
        monkeypatch.setenv('SSL_CERT_FILE', '')
        monkeypatch.setattr(app.sys, 'platform', 'darwin')
        app._ensure_ca_bundle()
        assert os.environ['SSL_CERT_FILE']

    @pytest.mark.parametrize('platform, preset', [('win32', ''), ('darwin', '/own/bundle.pem')])
    def test_worker_bundle_left_alone(self, monkeypatch, platform, preset):
        from ml_worker import compat
        monkeypatch.setenv('SSL_CERT_FILE', preset)
        monkeypatch.setattr(compat.sys, 'platform', platform)
        compat.ensure_ca_bundle()
        assert os.environ['SSL_CERT_FILE'] == preset


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

    def test_every_model_id_runs_on_whisperx(self):
        """The MLX engine is gone: even a retired mlx id that slipped past
        normalize_stt_model_id() lands on WhisperX rather than a missing module."""
        from ml_worker.engines import ENGINE_WHISPERX, engine_kind_for_model

        for model_id in ('whisperx-large-v3', 'mlx-whisper-large-v3', ''):
            assert engine_kind_for_model(model_id) == ENGINE_WHISPERX

    def test_whisperx_on_mac_runs_on_metal_in_float16(self):
        from ml_worker.engines import select_engine_config

        engine, device, compute_type = select_engine_config(
            'whisperx-large-v3', prefer_mps=True,
            detected_device='cuda', detected_compute='float16', metal=True)

        assert (engine, device, compute_type) == ('whisperx', 'mps', 'float16')

    def test_whisperx_on_mac_without_metal_is_cpu_int8(self):
        from ml_worker.engines import select_engine_config

        engine, device, compute_type = select_engine_config(
            'whisperx-large-v3', prefer_mps=True,
            detected_device='cuda', detected_compute='float16', metal=False)

        assert (engine, device, compute_type) == ('whisperx', 'cpu', 'int8')

    @pytest.mark.parametrize('metal, expected', [(True, ('mps', 'float16')),
                                                 (False, ('cpu', 'int8'))])
    def test_create_engine_on_mac(self, metal, expected):
        from ml_worker.engines import create_engine
        env = MagicMock(stt_model_id='whisperx-large-v3')
        with patch('ml_worker.engines.whisperx_engine.ct2_metal_available', return_value=metal):
            engine = create_engine(env, prefer_mps=True)
        assert (engine._device, engine._compute_type) == expected


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
