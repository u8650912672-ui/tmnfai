"""
TrackMania Nations Forever – Reward System
==========================================
Two-tier reward computation:

  Tier 1 – TMInterface (preferred)
    If the `tminterface` package is installed and the game is running through
    TMInterface, real game state (speed, yaw/pitch/roll) is read directly from
    game memory.  This is very reliable and fast.

  Tier 2 – Screen-only fallback
    If TMInterface is unavailable, speed is estimated by OCR on the HUD at the
    bottom-centre of the screen using pytesseract (if installed) or defaults to
    a fixed estimate.

Reward formula (inspired by TMAI / LouisDeOliveira):
  + speed_reward  = speed / 400          (reward going fast)
  + roll_penalty  = -|roll| / π          (penalise flipping, TMI only)
  + gas_reward    = action[0] * 2        (encourage pressing accelerator)
  + constant      = -0.3                 (small negative each step to
                                          encourage finishing quickly)
  + stuck_penalty = -5 * stuck_steps     (escalating penalty when nearly
                                          stationary for multiple frames)

NOTE: There is deliberately NO penalty for proximity to track edges.
      The AI is free to hug or ride walls — in TrackMania this is often
      the fastest line.
"""

from __future__ import annotations

import re
import time
from collections import deque

import cv2
import numpy as np

# ── Optional TMInterface import ───────────────────────────────────────────────
try:
    from tminterface.client import Client
    from tminterface.interface import TMInterface
    _TMI_AVAILABLE = True
except ImportError:
    _TMI_AVAILABLE = False

# ── Optional pytesseract import (speed OCR fallback) ─────────────────────────
try:
    import pytesseract
    _TESS_AVAILABLE = True
except ImportError:
    _TESS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# TMInterface threaded client
# ─────────────────────────────────────────────────────────────────────────────

class _SimStateClient(Client):
    """Thin TMInterface client that just stores the latest simulation state."""

    def __init__(self):
        super().__init__()
        self.sim_state = None

    def on_run_step(self, iface, _time: int):
        self.sim_state = iface.get_simulation_state()


class TMIThread:
    """
    Runs a TMInterface client in a background daemon thread so the main loop
    can read game state without blocking.

    Usage:
        tmi = TMIThread()          # start thread
        state = tmi.state          # read latest state (may be None initially)
        tmi.stop()                 # on shutdown
    """

    def __init__(self):
        import threading
        self._iface = TMInterface()
        self._client = _SimStateClient()
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        self._iface.register(self._client)
        while self._running and self._iface.running:
            time.sleep(0)
            with self._lock:
                pass   # just keep the thread alive; client updates itself

    @property
    def state(self):
        with self._lock:
            return self._client.sim_state

    def stop(self):
        self._running = False


# ─────────────────────────────────────────────────────────────────────────────
# Reward system
# ─────────────────────────────────────────────────────────────────────────────

class RewardSystem:
    """
    Computes per-step rewards for the TrackMania AI.

    Parameters
    ----------
    use_tmi : bool
        Try to connect to TMInterface on construction.  Set False to force the
        screen-only fallback regardless of whether TMInterface is installed.
    max_speed : float
        Expected maximum speed in km/h for normalisation (default 400).
    max_stuck_steps : int
        After this many consecutive low-speed steps the episode is considered
        stuck and should be reset (main.py checks get_stats()['stuck']).
    """

    # Speed below which the car is considered "nearly stopped"
    LOW_SPEED_THRESHOLD = 10.0   # km/h

    def __init__(
        self,
        use_tmi: bool = True,
        max_speed: float = 400.0,
        max_stuck_steps: int = 100,
    ):
        self.max_speed = max_speed
        self.max_stuck_steps = max_stuck_steps

        # TMInterface connection (optional)
        self.tmi_thread: TMIThread | None = None
        if use_tmi and _TMI_AVAILABLE:
            try:
                self.tmi_thread = TMIThread()
                print("[RewardSystem] TMInterface connected — using game state.")
            except Exception as e:
                print(f"[RewardSystem] TMInterface unavailable ({e}), using screen fallback.")
        else:
            if not _TMI_AVAILABLE:
                print("[RewardSystem] tminterface not installed — using screen fallback.")
            else:
                print("[RewardSystem] TMInterface disabled by user — using screen fallback.")

        # Running statistics
        self._speed_history: deque[float] = deque(maxlen=50)
        self._reward_history: deque[float] = deque(maxlen=200)
        self.stuck_steps = 0
        self.total_steps = 0
        self.episode_reward = 0.0
        self.best_speed = 0.0

        # HUD speed-text region (fraction of screen: left, top, right, bottom)
        # TMNF shows speed at the bottom-centre of the screen.
        self._speed_roi = (0.42, 0.88, 0.58, 0.98)

    # ── Speed extraction ──────────────────────────────────────────────────────

    def _speed_from_tmi(self) -> float | None:
        """Return speed in km/h from TMInterface state, or None if unavailable."""
        if self.tmi_thread is None:
            return None
        state = self.tmi_thread.state
        if state is None:
            return None
        try:
            return float(state.display_speed)
        except Exception:
            return None

    def _speed_from_screen(self, frame: np.ndarray) -> float:
        """
        Estimate speed by OCR-ing the HUD speedometer region.
        Falls back to 0.0 if pytesseract is not installed or OCR fails.
        """
        if not _TESS_AVAILABLE or frame is None:
            return 0.0
        h, w = frame.shape[:2]
        x1 = int(self._speed_roi[0] * w)
        y1 = int(self._speed_roi[1] * h)
        x2 = int(self._speed_roi[2] * w)
        y2 = int(self._speed_roi[3] * h)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        gray = cv2.convertScaleAbs(gray, alpha=2.5, beta=0)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        binary = cv2.resize(binary, None, fx=3, fy=3,
                            interpolation=cv2.INTER_CUBIC)
        try:
            cfg = "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(binary, config=cfg).strip()
            digits = re.sub(r"\D", "", text)
            return float(digits) if digits else 0.0
        except Exception:
            return 0.0

    def get_speed(self, frame: np.ndarray | None = None) -> float:
        """Return current speed in km/h from the best available source."""
        spd = self._speed_from_tmi()
        if spd is None:
            spd = self._speed_from_screen(frame)
        return max(0.0, spd)

    # ── Roll extraction (TMInterface only) ────────────────────────────────────

    def _get_roll(self) -> float:
        """Return the car's roll angle in radians, or 0 if unavailable."""
        if self.tmi_thread is None:
            return 0.0
        state = self.tmi_thread.state
        if state is None:
            return 0.0
        try:
            return float(state.yaw_pitch_roll[2])
        except Exception:
            return 0.0

    # ── Reward calculation ────────────────────────────────────────────────────

    def calculate_reward(
        self,
        frame: np.ndarray | None,
        action: np.ndarray,
        lidar_obs: np.ndarray | None = None,
    ) -> float:
        """
        Compute the reward for a single time step.

        Parameters
        ----------
        frame   : latest raw BGR frame (used for screen fallback)
        action  : [gas, left, right, brake] float array
        lidar_obs : LIDAR distance array — currently unused in reward calculation
                    (no edge penalty) but available for future extensions.

        Returns
        -------
        float reward, clipped to [-20, 20]
        """
        speed = self.get_speed(frame)
        self._speed_history.append(speed)
        self.total_steps += 1
        self.best_speed = max(self.best_speed, speed)

        # ── Speed reward ──────────────────────────────────────────────────────
        speed_reward = speed / self.max_speed   # 0–1

        # ── Roll penalty (only when TMInterface is available) ─────────────────
        roll = self._get_roll()
        roll_penalty = -abs(roll) / 3.14159     # 0 to -1

        # ── Gas reward — encourage pressing the accelerator ───────────────────
        gas = float(action[0])
        gas_reward = gas * 2.0 if gas > 0 else 0.0

        # ── Constant step cost (encourages finishing quickly) ─────────────────
        constant = -0.3

        # ── Stuck / low-speed penalty ─────────────────────────────────────────
        stuck_penalty = 0.0
        if speed < self.LOW_SPEED_THRESHOLD:
            self.stuck_steps += 1
            stuck_penalty = -5.0 * self.stuck_steps
        else:
            self.stuck_steps = 0

        # ── No edge penalty — wall-riding is allowed ──────────────────────────

        reward = speed_reward + roll_penalty + gas_reward + constant + stuck_penalty
        reward = float(np.clip(reward, -20.0, 20.0))

        self._reward_history.append(reward)
        self.episode_reward += reward
        return reward

    # ── Statistics ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        Return a dictionary of statistics for display and logging.
        'stuck' is True when the car has been at low speed long enough to
        warrant a race restart.
        """
        avg_speed = float(np.mean(self._speed_history)) if self._speed_history else 0.0
        avg_reward = float(np.mean(self._reward_history)) if self._reward_history else 0.0
        return {
            "speed":         self.get_speed(),
            "avg_speed":     avg_speed,
            "best_speed":    self.best_speed,
            "stuck_steps":   self.stuck_steps,
            "stuck":         self.stuck_steps >= self.max_stuck_steps,
            "total_steps":   self.total_steps,
            "episode_reward": self.episode_reward,
            "avg_reward":    avg_reward,
            "tmi_active":    self.tmi_thread is not None,
        }

    def reset_episode(self):
        """Call this after each race restart."""
        self.stuck_steps = 0
        self.episode_reward = 0.0

    def shutdown(self):
        """Clean up the TMInterface background thread."""
        if self.tmi_thread is not None:
            self.tmi_thread.stop()
