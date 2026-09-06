"""
Cross-platform system volume control.

Provides a single `set_system_volume(percent)` function that maps to the
right backend for the current OS:
  - Windows -> pycaw (Core Audio)
  - macOS   -> `osascript`
  - Linux   -> `amixer` (ALSA)

If no backend is available, volume changes are silently skipped so the
rest of the app (hand tracking + UI) still runs.
"""

import platform
import subprocess

_SYSTEM = platform.system()
_backend_ready = False

if _SYSTEM == "Windows":
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        _devices = AudioUtilities.GetSpeakers()
        _interface = _devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        _volume_ctrl = cast(_interface, POINTER(IAudioEndpointVolume))
        _min_db, _max_db, _ = _volume_ctrl.GetVolumeRange()
        _backend_ready = True
    except Exception as exc:  # pragma: no cover - depends on OS/hardware
        print(f"[WARN] pycaw unavailable, volume control disabled: {exc}")


def set_system_volume(percent: float) -> None:
    """Set the system output volume to `percent` (0-100)."""
    percent = max(0.0, min(100.0, percent))

    if _SYSTEM == "Windows" and _backend_ready:
        db = _min_db + (percent / 100.0) * (_max_db - _min_db)
        _volume_ctrl.SetMasterVolumeLevel(db, None)

    elif _SYSTEM == "Darwin":
        subprocess.run(["osascript", "-e", f"set volume output volume {percent}"], check=False)

    elif _SYSTEM == "Linux":
        subprocess.run(["amixer", "-D", "pulse", "sset", "Master", f"{percent}%"], check=False, stdout=subprocess.DEVNULL)

    # Any other platform: no-op.
