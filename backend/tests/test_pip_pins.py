"""Windows installs the same pure-Python ML stack the Mac pins.

The Mac install list fixes whisperx, faster-whisper, transformers and
huggingface-hub to the versions mac-bench measured; Windows took whatever PyPI
had that day, so a compromised or broken release reached new installs at once.
torch and its companions stay with torchruntime, which picks the build for the
GPU it finds — a fixed version there would break some cards. pyannote-audio
floats on both platforms by the author's choice.
"""

import pytest


@pytest.fixture
def windows(monkeypatch, tmp_path):
    from app.services import pip_installer as pi

    monkeypatch.setattr(pi, 'IS_MAC', False)
    monkeypatch.setattr(pi, '_check_packages', lambda: [{'name': 'whisperx', 'installed': False}])
    monkeypatch.setattr(pi, '_install_log_path', lambda: str(tmp_path / 'install.log'))
    monkeypatch.setattr(pi, '_run_torchruntime_install', lambda: None)
    monkeypatch.setattr(pi, '_companion_pip_extra_args', lambda force_reinstall=False: [])
    monkeypatch.setattr(pi, '_torch_companion_channels_aligned', lambda: True)
    monkeypatch.setattr(pi, '_safe_emit', lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(pi, '_run_pip',
                        lambda packages, extra_args=None: calls.append(list(packages)) or True)
    return pi, calls


def test_windows_install_pins_the_python_ml_stack(windows):
    pi, calls = windows

    assert pi.install_pip_packages() is True

    installed = [spec for call in calls for spec in call]
    for spec in ('whisperx==3.8.6', 'faster-whisper==1.2.1',
                 'transformers==4.57.6', 'huggingface-hub==0.36.2'):
        assert spec in installed, f'{spec} was not requested'


def test_windows_leaves_torch_to_torchruntime_and_pyannote_floating(windows):
    pi, calls = windows

    pi.install_pip_packages()

    installed = [spec for call in calls for spec in call]
    assert 'torchaudio' in installed and 'torchvision' in installed
    assert 'pyannote-audio' in installed


def test_windows_whisperx_deps_cannot_replace_the_installed_torch(windows, monkeypatch):
    # pyannote-audio floats, and so does what it pulls in: pytorch-metric-learning
    # 3.0.4 asks for torch>=2.14, so pip swapped the CUDA torch torchruntime chose
    # for a CPU torch from PyPI, and the re-align step put the 2.75 GB CUDA wheel
    # back (seen 2026-11-22). Phase 5 holds the torch stack where Phase 2-3 left it.
    pi, _ = windows
    monkeypatch.setattr(pi, '_torch_stack_constraints',
                        lambda: ['torch==2.11.0+cu128', 'torchaudio==2.11.0+cu128',
                                 'torchvision==0.26.0+cu128'])
    seen = []

    def fake_run_pip(packages, extra_args=None):
        args = list(extra_args or [])
        constraints = None
        if '-c' in args:
            with open(args[args.index('-c') + 1], encoding='utf-8') as fh:
                constraints = fh.read().split()
        seen.append((list(packages), constraints))
        return True

    monkeypatch.setattr(pi, '_run_pip', fake_run_pip)

    assert pi.install_pip_packages() is True

    phase5 = [c for p, c in seen if 'pyannote-audio' in p]
    assert phase5 == [['torch==2.11.0+cu128', 'torchaudio==2.11.0+cu128',
                       'torchvision==0.26.0+cu128']]


def test_torch_stack_constraints_name_what_is_installed(monkeypatch):
    import importlib.metadata as md

    from app.services import pip_installer as pi

    versions = {'torch': '2.11.0+cu128', 'torchvision': '0.26.0+cu128'}

    def fake_version(name):
        if name not in versions:
            raise md.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(pi.importlib.metadata, 'version', fake_version)

    assert pi._torch_stack_constraints() == ['torch==2.11.0+cu128',
                                             'torchvision==0.26.0+cu128']


def test_windows_self_heal_installs_the_pinned_version(monkeypatch):
    from app.services import pip_installer as pi

    monkeypatch.setattr(pi, 'IS_MAC', False)

    assert pi._pinned('whisperx') == 'whisperx==3.8.6'
    assert pi._pinned('pyannote-audio') == 'pyannote-audio'
    assert pi._pinned('torch') == 'torch'


def test_both_platforms_pin_the_same_versions():
    from app.services import pip_installer as pi

    for name, version in pi.SHARED_ML_PINS.items():
        assert f'{name}=={version}' in pi.MAC_PIP_INSTALL_TARGETS
