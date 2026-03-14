"""
TrackMania Nations Forever – Input Controller
==============================================
Uses DirectInput scan codes (not VK codes) with the KEYEVENTF_SCANCODE flag so
that the game actually receives the inputs.  Arrow-key VK codes are commonly
ignored by games; scan codes are not.

Hold/release pattern (inspired by TMAI / LouisDeOliveira):
  Each call to execute_action() inspects the desired state of every key and
  either presses or releases it.  Keys therefore stay *held* across frames as
  long as the action vector keeps them >0.5 — exactly how a human driver would
  hold the accelerator through a straight.

Action vector layout:  [gas, steer_left, steer_right, brake]
  Each element is a float 0–1.  >0.5 → key held, ≤0.5 → key released.
"""

import ctypes
import time

# ── DirectInput scan codes ────────────────────────────────────────────────────
SCAN_UP    = 0xC8   # Arrow Up   (gas)
SCAN_DOWN  = 0xD0   # Arrow Down (reverse – rarely used)
SCAN_LEFT  = 0xCB   # Arrow Left (steer left)
SCAN_RIGHT = 0xCD   # Arrow Right (steer right)
SCAN_S     = 0x1F   # S key      (brake)
SCAN_DEL   = 0xD3   # Delete key (restart race in TMNF)

# ── SendInput constants ───────────────────────────────────────────────────────
INPUT_KEYBOARD       = 1
KEYEVENTF_SCANCODE   = 0x0008   # interpret wScan as a hardware scan code
KEYEVENTF_KEYUP      = 0x0002   # set this flag to release the key

PUL = ctypes.POINTER(ctypes.c_ulong)


# ── ctypes structs ────────────────────────────────────────────────────────────
class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk",         ctypes.c_ushort),
        ("wScan",       ctypes.c_ushort),
        ("dwFlags",     ctypes.c_ulong),
        ("time",        ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg",    ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx",          ctypes.c_long),
        ("dy",          ctypes.c_long),
        ("mouseData",   ctypes.c_ulong),
        ("dwFlags",     ctypes.c_ulong),
        ("time",        ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("ki", KEYBDINPUT),
        ("mi", MOUSEINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type",  ctypes.c_ulong),
        ("union", INPUT_UNION),
    ]


# ── Helper ────────────────────────────────────────────────────────────────────
def _send_scan(scan_code: int, key_up: bool = False):
    """Send a single key press or release via DirectInput scan code."""
    extra = ctypes.c_ulong(0)
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if key_up else 0)
    union = INPUT_UNION()
    union.ki = KEYBDINPUT(0, scan_code, flags, 0, ctypes.pointer(extra))
    inp = INPUT(INPUT_KEYBOARD, union)
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


# ── Main class ────────────────────────────────────────────────────────────────
class InputController:
    """
    Manages keyboard input for TrackMania Nations Forever.

    Keys
    ----
    Arrow Up    – gas
    Arrow Left  – steer left
    Arrow Right – steer right
    S           – brake
    Delete      – restart race
    """

    # Map action-vector index → scan code
    ACTION_KEYS = [SCAN_UP, SCAN_LEFT, SCAN_RIGHT, SCAN_S]
    ACTION_NAMES = ["GAS", "LEFT", "RIGHT", "BRAKE"]

    def __init__(self):
        # Track which keys are currently held so we don't send redundant events
        self._held: dict[int, bool] = {k: False for k in self.ACTION_KEYS}

    def _press(self, scan_code: int):
        if not self._held[scan_code]:
            _send_scan(scan_code, key_up=False)
            self._held[scan_code] = True

    def _release(self, scan_code: int):
        if self._held[scan_code]:
            _send_scan(scan_code, key_up=True)
            self._held[scan_code] = False

    def execute_action(self, action):
        """
        Apply a 4-element action vector [gas, left, right, brake] to the game.
        Each value >0.5 holds the key; ≤0.5 releases it.
        Keys remain held between calls — no tapping.
        """
        for scan_code, value in zip(self.ACTION_KEYS, action):
            if value > 0.5:
                self._press(scan_code)
            else:
                self._release(scan_code)

    def release_all(self):
        """Release every key — call this on pause or shutdown."""
        for scan_code in self.ACTION_KEYS:
            self._release(scan_code)

    def restart_race(self):
        """
        Press and release the Delete key to restart the current race in TMNF.
        Does NOT require restart to be mapped to DEL in-game — TMNF uses DEL
        by default.
        """
        self.release_all()
        _send_scan(SCAN_DEL, key_up=False)
        time.sleep(0.1)
        _send_scan(SCAN_DEL, key_up=True)
        time.sleep(0.5)   # give the game time to reload the track
