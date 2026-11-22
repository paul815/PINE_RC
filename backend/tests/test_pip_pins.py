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
