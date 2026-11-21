"""Audio helpers: ffprobe/ffmpeg decode, WAV writing."""

import subprocess


def get_duration_secs(audio_path):
    """Return audio duration in seconds via ffprobe (no file loading)."""
    try:
        out = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def load_audio_range(audio_path, offset, duration):
    """Load a time range of audio as 16 kHz float32 mono numpy array.

    Uses the same format as whisperx.load_audio (s16le → float32 / 32768).
    """
    import numpy as np
    cmd = ['ffmpeg', '-nostdin', '-threads', '0',
           '-ss', str(offset), '-t', str(duration),
           '-i', audio_path,
           '-f', 's16le', '-ac', '1', '-acodec', 'pcm_s16le',
           '-ar', '16000', '-']
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def load_audio_file(audio_path, channel=None):
    """Load a whole file as 16 kHz float32 mono numpy array.

    ``channel`` selects one channel instead of downmixing all of them, which is
    what makes a channel-per-speaker recording usable: the default ``-ac 1``
    averages the speakers together, which is precisely what must not happen
    here.
    """
    import numpy as np
    cmd = ['ffmpeg', '-nostdin', '-threads', '0', '-i', audio_path]
    if channel is None:
        cmd += ['-ac', '1']
    else:
        cmd += ['-filter_complex', f'[0:a]pan=mono|c0=c{int(channel)}[a]',
                '-map', '[a]']
    cmd += ['-f', 's16le', '-acodec', 'pcm_s16le', '-ar', '16000', '-']
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def count_channels(audio_path):
    """Channels in the file's default audio stream; 0 when it has no audio."""
    try:
        out = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
             '-show_entries', 'stream=channels',
             '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return 0


def write_wav(path, audio_f32, sample_rate=16000):
    """Write float32 mono audio array to a 16-bit PCM WAV file."""
    import struct

    import numpy as np
    pcm = (audio_f32 * 32768).astype(np.int16)
    n_samples = len(pcm)
    data_size = n_samples * 2
    with open(path, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVEfmt ')
        f.write(struct.pack('<IHHIIHH', 16, 1, 1, sample_rate,
                            sample_rate * 2, 2, 16))
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(pcm.tobytes())


def fmt_elapsed(secs):
    """Format a duration in seconds as a human-readable string (e.g. '5m 30s')."""
    secs = int(secs)
    if secs < 60:
        return f'{secs}s'
    m, s = divmod(secs, 60)
    if m < 60:
        return f'{m}m {s:02d}s'
    h, m = divmod(m, 60)
    return f'{h}h {m:02d}m'
