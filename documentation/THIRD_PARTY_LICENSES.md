# Third-party licenses

PINE itself is [MIT](../LICENSE). It redistributes a handful of files, depends on
packages installed from PyPI, and downloads machine-learning models at setup.
Those carry their own terms, listed here.

> Compiled 2026-08-22. Entries marked **(confirm)** were taken from the upstream
> project's usual terms and should be checked against the current release page
> before publishing a release; the rest are stated in the file or repository that
> ships them. Re-check this file whenever a pinned version changes.

---

## Redistributed in this repository

These files are committed here, so their licenses must travel with them.

### Fonts — `backend/app/static/fonts/`

All six families are served from the Google Fonts OFL collection and subset to
Latin + Cyrillic by `backend/scripts/build_cyrillic_fonts.py`. Subsetting counts
as a modification under the SIL Open Font License, which permits it and requires
the license and copyright notice to be distributed with the font files.

| Family | Files | License | Upstream |
|---|---|---|---|
| Inter | `Inter-*.woff2` | SIL OFL 1.1 | [google/fonts · ofl/inter](https://github.com/google/fonts/tree/main/ofl/inter) |
| Roboto | `Roboto-*.woff2` | SIL OFL 1.1 | [google/fonts · ofl/roboto](https://github.com/google/fonts/tree/main/ofl/roboto) |
| Open Sans | `OpenSans-*.woff2` | SIL OFL 1.1 | [google/fonts · ofl/opensans](https://github.com/google/fonts/tree/main/ofl/opensans) |
| Source Sans 3 | `SourceSans3-*.woff2` | SIL OFL 1.1 | [google/fonts · ofl/sourcesans3](https://github.com/google/fonts/tree/main/ofl/sourcesans3) |
| JetBrains Mono | `JetBrainsMono-*.woff2` | SIL OFL 1.1 | [google/fonts · ofl/jetbrainsmono](https://github.com/google/fonts/tree/main/ofl/jetbrainsmono) |

**Outstanding:** the license texts are not in the repository yet. Each family's
`OFL.txt` carries its own copyright line, so fetch all five into
`backend/app/static/fonts/licenses/`:

```bash
mkdir -p backend/app/static/fonts/licenses && for f in inter roboto opensans sourcesans3 jetbrainsmono; do curl -sSL "https://raw.githubusercontent.com/google/fonts/main/ofl/$f/OFL.txt" -o "backend/app/static/fonts/licenses/$f-OFL.txt"; done
```

### JavaScript — `backend/app/static/js/`

| File | Component | License |
|---|---|---|
| `socket.io.min.js` | Socket.IO client 4.8.3, © 2014-2025 Guillermo Rauch | MIT (stated in the file header) |

---

## Python packages

### Base — `backend/requirements.txt`, installed by the launcher

| Package | License |
|---|---|
| Flask | BSD-3-Clause |
| Flask-SQLAlchemy | BSD-3-Clause |
| Flask-SocketIO | MIT |
| Flask-Cors | MIT |
| huggingface-hub | Apache-2.0 |
| psutil | BSD-3-Clause |
| static-ffmpeg | MIT — the package. It downloads **FFmpeg binaries**, which are LGPL-2.1+ or GPL depending on the build **(confirm which build is fetched before distributing anything alongside it)** |

The full resolved tree is in `backend/requirements-lock.txt`.

### Installed at runtime during onboarding

Pulled by `app/services/pip_installer.py`, so they never enter this repository —
but they are part of what a user ends up running.

| Package | License |
|---|---|
| torch, torchaudio | BSD-3-Clause |
| faster-whisper | MIT |
| ctranslate2 | MIT |
| whisperx | BSD-2-Clause **(confirm)** |
| pyannote.audio | MIT |
| onnxruntime | MIT |
| gliner (optional, PII) | Apache-2.0 **(confirm)** |

---

## Models

Downloaded from the HuggingFace Hub during setup, into the folder chosen there.
They are not redistributed by this project, and each carries the terms on its own
model card.

| Model | Used for | Terms |
|---|---|---|
| [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) | Transcription | MIT — OpenAI Whisper weights |
| [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | Diarization | Gated — you must accept the conditions on the model page with your own account **(confirm the license line)** |
| [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) | Speaker embeddings | **(confirm on the model card)** |
| [urchade/gliner_multi_pii-v1](https://huggingface.co/urchade/gliner_multi_pii-v1) | Optional PII removal | Apache-2.0 **(confirm)** |
| [microsoft/mdeberta-v3-base](https://huggingface.co/microsoft/mdeberta-v3-base) | Tokenizer for GLiNER | MIT |

---

## Icons and sounds

`backend/app/static/icons/` and `backend/app/static/sounds/` hold the
application icon and the completion chime. If either came from a third party,
add it here with its source and terms before the next release.
