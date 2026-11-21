import os
import shutil
import subprocess
import sys

import psutil

IS_MAC = sys.platform == 'darwin'


def _detect_nvidia_gpu():
    """Detect NVIDIA GPU via nvidia-smi (works even without CUDA PyTorch)."""
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().split('\n')[0]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _detect_cuda_toolkit():
    """Detect CUDA Toolkit installation.

    Checks nvcc via PATH first, then falls back to CUDA_PATH env var and the
    standard Windows installation directory — because Electron/subprocess
    environments often inherit a stripped PATH that omits the CUDA bin folder.
    """
    try:
        r = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    if sys.platform == 'win32':
        # CUDA installer sets CUDA_PATH regardless of PATH
        if os.environ.get('CUDA_PATH'):
            nvcc = os.path.join(os.environ['CUDA_PATH'], 'bin', 'nvcc.exe')
            if os.path.isfile(nvcc):
                return True
        # Fallback: check standard install location for any version
        toolkit_root = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
        if os.path.isdir(toolkit_root):
            for ver_dir in os.listdir(toolkit_root):
                nvcc = os.path.join(toolkit_root, ver_dir, 'bin', 'nvcc.exe')
                if os.path.isfile(nvcc):
                    return True

    return False


def _detect_apple_silicon():
    """Return the Apple Silicon chip name, or None on Intel Macs.

    Deliberately does not import torch: this runs on the first onboarding screen,
    before setup installs anything, and MPS support is a property of the hardware
    and OS -- not of whether PyTorch happens to be present yet. sysctl is used
    instead of platform.machine() because a Python running under Rosetta reports
    x86_64 on an Apple Silicon machine.
    """
    try:
        r = subprocess.run(
            ['sysctl', '-n', 'hw.optional.arm64'],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0 or r.stdout.strip() != '1':
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None

    try:
        r = subprocess.run(
            ['sysctl', '-n', 'machdep.cpu.brand_string'],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return 'Apple Silicon'


def _mac_version_supports_mps():
    """MPS requires macOS 12.3+. Unknown versions are treated as supported."""
    import platform
    ver = platform.mac_ver()[0]
    if not ver:
        return True
    try:
        parts = [int(p) for p in ver.split('.')[:2]]
    except ValueError:
        return True
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts) >= (12, 3)


def _existing_ancestor(path):
    """Return the nearest existing ancestor of path (or path itself if it exists)."""
    p = os.path.abspath(path)
    while not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return p


def run_system_check(models_path=None):
    checks = []

    py_ver = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
    py_ok = sys.version_info.major == 3 and sys.version_info.minor in (11, 12, 13)
    py_explanation = (
        'WhisperX and its dependencies (ctranslate2, PyTorch) only provide wheels for '
        'Python 3.11–3.13. Python 3.14+ is not yet supported by the ML ecosystem.'
    )
    if py_ok:
        checks.append({
            'name': 'Python',
            'status': 'ok',
            'detail': f'Installed: {py_ver} · {sys.executable}',
            'version': py_ver,
            'extra': 'required 3.11–3.13',
            'explanation': py_explanation,
        })
    else:
        checks.append({
            'name': 'Python',
            'status': 'warn',
            'detail': f'Installed: {py_ver} · {sys.executable}',
            'version': py_ver,
            'extra': 'required 3.11–3.13',
            'explanation': py_explanation,
            'download_url': 'https://www.python.org/downloads/',
            'link_label': 'Download Python',
        })

    # No ffmpeg check here on purpose. This runs on the first onboarding screen,
    # and the old version called static_ffmpeg.add_paths() whenever ffmpeg was
    # missing from PATH -- which on a fresh install is a ~190 MB download, so the
    # system-check step would stall for minutes. Reporting it without fetching is
    # no better: PINE now installs ffmpeg itself during the model download, so a
    # "not found" line here would flag something the user cannot act on and does
    # not need to. See app.services.ffmpeg_setup.

    if IS_MAC:
        # macOS: check for Metal (MPS) only — CUDA / NVIDIA are not relevant
        # The verdict comes from the hardware, not from torch: on a supported Mac
        # this screen must read "ok" on the very first run, before setup installs
        # anything. Flagging a capable machine amber only because torch is not
        # there yet is noise the user cannot act on.
        chip = _detect_apple_silicon()
        try:
            import torch
            torch_ver = f'PyTorch {torch.__version__}'
        except ImportError:
            torch_ver = None

        if chip and _mac_version_supports_mps():
            checks.append({
                'name': 'GPU acceleration',
                'status': 'ok',
                'detail': f'{chip} \u00b7 Metal (MPS) acceleration via WhisperX',
                'version': torch_ver,
            })
        elif chip:
            checks.append({
                'name': 'GPU acceleration',
                'status': 'warn',
                'detail': f'{chip} \u00b7 Metal acceleration requires macOS 12.3+ \u00b7 CPU mode',
                'version': torch_ver,
            })
        else:
            checks.append({
                'name': 'GPU acceleration',
                'status': 'warn',
                'detail': 'Intel Mac \u00b7 no Metal acceleration \u00b7 CPU mode',
                'version': torch_ver,
            })
    else:
        # Windows / Linux: check for CUDA and NVIDIA GPU
        try:
            import torch
            torch_ver = torch.__version__
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                vram_gb = getattr(props, 'total_memory', 0) or getattr(props, 'total_mem', 0)
                vram_gb = vram_gb / (1024 ** 3)
                checks.append({
                    'name': 'CUDA / GPU acceleration',
                    'status': 'ok',
                    'detail': f'{gpu_name} \u00b7 {vram_gb:.0f} GB VRAM',
                    'version': f'PyTorch {torch_ver}',
                    'has_nvidia_gpu': True,
                    'has_cuda': True,
                    'vram_gb': round(vram_gb, 1),
                })
                if vram_gb <= 4:
                    checks.append({
                        'name': 'VRAM',
                        'status': 'warn',
                        'detail': f'{vram_gb:.0f} GB \u2014 transcription may be slow or fail on large files',
                    })
            else:
                nvidia_gpu = _detect_nvidia_gpu()
                if nvidia_gpu:
                    has_cuda_toolkit = _detect_cuda_toolkit()
                    if has_cuda_toolkit:
                        # CUDA Toolkit is installed but torch build doesn't match — torchruntime fixes this
                        checks.append({
                            'name': 'CUDA / GPU acceleration',
                            'status': 'warn',
                            'detail': (
                                f'{nvidia_gpu} \u00b7 CUDA Toolkit detected but PyTorch {torch_ver} '
                                f'was built for an older CUDA version \u00b7 will be resolved during setup'
                            ),
                            'version': f'PyTorch {torch_ver}',
                            'has_nvidia_gpu': True,
                            'has_cuda': False,
                            'gpu_name': nvidia_gpu,
                            'recheckable': True,
                        })
                    else:
                        checks.append({
                            'name': 'CUDA / GPU acceleration',
                            'status': 'err',
                            'detail': f'{nvidia_gpu} detected but CUDA not available \u00b7 PyTorch {torch_ver}',
                            'version': f'PyTorch {torch_ver}',
                            'has_nvidia_gpu': True,
                            'has_cuda': False,
                            'gpu_name': nvidia_gpu,
                            'recheckable': True,
                            'download_url': 'https://developer.nvidia.com/cuda-downloads',
                            'link_label': 'Download CUDA Toolkit',
                            'explanation': (
                                'PINE requires the NVIDIA CUDA Toolkit to use your GPU for '
                                'transcription. Install CUDA, restart your computer, then '
                                'click Re-check.'
                            ),
                        })
                else:
                    checks.append({
                        'name': 'CUDA / GPU acceleration',
                        'status': 'warn',
                        'detail': f'No compatible GPU detected \u00b7 CPU mode (PyTorch {torch_ver})',
                        'version': f'PyTorch {torch_ver}',
                        'has_nvidia_gpu': False,
                        'has_cuda': False,
                        'cpu_opt_in_required': True,
                        'time_comparison': 'GPU: ~5 min per hour of audio \u00b7 CPU: ~1\u20133 hours per hour of audio',
                    })
        except ImportError:
            base_torch_ver = None
            if sys.base_exec_prefix != sys.exec_prefix:
                base_exe = os.path.join(
                    sys.base_exec_prefix,
                    'python.exe' if sys.platform == 'win32' else os.path.join('bin', 'python3'),
                )
                try:
                    r = subprocess.run(
                        [base_exe, '-c', 'import torch; print(torch.__version__)'],
                        capture_output=True, text=True, timeout=10,
                    )
                    if r.returncode == 0:
                        base_torch_ver = r.stdout.strip()
                except Exception:
                    pass

            nvidia_gpu = _detect_nvidia_gpu()
            has_cuda_toolkit = _detect_cuda_toolkit() if nvidia_gpu else False

            if nvidia_gpu and not has_cuda_toolkit:
                # NVIDIA GPU present but CUDA Toolkit not installed — block and warn
                checks.append({
                    'name': 'CUDA / GPU acceleration',
                    'status': 'err',
                    'detail': f'{nvidia_gpu} detected \u00b7 CUDA Toolkit not installed',
                    'has_nvidia_gpu': True,
                    'has_cuda': False,
                    'gpu_name': nvidia_gpu,
                    'recheckable': True,
                    'download_url': 'https://developer.nvidia.com/cuda-downloads',
                    'link_label': 'Download CUDA Toolkit',
                    'explanation': (
                        'PINE requires the NVIDIA CUDA Toolkit to use your GPU for '
                        'transcription. Install CUDA, restart your computer, then '
                        'click Re-check.'
                    ),
                })
            elif nvidia_gpu and has_cuda_toolkit:
                # NVIDIA GPU + CUDA Toolkit present — torch will be set up during install
                detail = f'{nvidia_gpu} \u00b7 CUDA detected \u00b7 PyTorch will be installed during setup'
                if base_torch_ver:
                    detail = f'{nvidia_gpu} \u00b7 CUDA detected \u00b7 PyTorch {base_torch_ver} (system-wide, will be moved to venv)'
                checks.append({
                    'name': 'CUDA / GPU acceleration',
                    'status': 'ok',
                    'detail': detail,
                    'has_nvidia_gpu': True,
                    'has_cuda': False,
                    'recheckable': True,
                })
            elif base_torch_ver:
                checks.append({
                    'name': 'CUDA / GPU acceleration',
                    'status': 'warn',
                    'detail': f'PyTorch {base_torch_ver} found system-wide \u00b7 run setup to install into venv',
                    'version': f'PyTorch {base_torch_ver}',
                    'recheckable': True,
                })
            else:
                checks.append({
                    'name': 'CUDA / GPU acceleration',
                    'status': 'warn',
                    'detail': 'PyTorch not installed \u00b7 will be installed during setup',
                    'version': None,
                    'recheckable': True,
                })

    check_path = models_path or os.path.dirname(os.path.abspath(__file__))
    try:
        drive = os.path.splitdrive(check_path)[0]
        disk_path = (drive + os.sep) if drive else _existing_ancestor(check_path)
        disk = shutil.disk_usage(disk_path)
        free_gb = disk.free / (1024 ** 3)
        used_pct = int((disk.used / disk.total) * 100)
        if free_gb >= 12:
            status = 'ok'
        elif free_gb >= 6:
            status = 'warn'
        else:
            status = 'err'
        checks.append({
            'name': 'Disk space',
            'status': status,
            'detail': f'{free_gb:.0f} GB available \u00b7 minimum 12 GB required for full install',
            'free_bytes': disk.free,
            'total_bytes': disk.total,
            'used_percent': used_pct,
        })
    except Exception as exc:
        checks.append({
            'name': 'Disk space',
            'status': 'warn',
            'detail': f'Could not check disk: {exc}',
        })

    ram = psutil.virtual_memory()
    ram_gb = ram.total / (1024 ** 3)
    if ram_gb >= 10:
        status = 'ok'
    elif ram_gb >= 6:
        status = 'warn'
    else:
        status = 'err'
    checks.append({
        'name': 'RAM',
        'status': status,
        'detail': f'{ram_gb:.0f} GB \u00b7 recommended >= 10 GB',
    })

    return checks
