"""
TrackMania Nations Forever – Screen Capture
===========================================
Provides three things:
  1. Raw BGR frame capture (game window or full monitor).
  2. Pre-processed grayscale frame for the CNN (stacked into the frame buffer).
  3. LIDAR-style ray observation array from processed edge-detected image
     (inspired by TMAI / LouisDeOliveira – GameCapture.py).

The LIDAR approach casts N rays from the bottom-centre of a processed image
outward.  Each ray travels until it hits a detected track edge (bright pixel)
or exits the frame.  The normalised distances give the neural network a compact
representation of how much road is visible in each direction.
"""

import cv2
import numpy as np
import mss
import win32gui


GAME_WINDOW_NAME = "TmForever"


def get_window_geometry(window_name: str = GAME_WINDOW_NAME):
    """
    Return (left, top, right, bottom) for the named window.
    Returns None if the window cannot be found.
    """
    hwnd = win32gui.FindWindow(None, window_name)
    if not hwnd:
        return None
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    # Trim window chrome (title bar ~30px, borders ~8px)
    return left + 8, top + 30, right - 8, bottom - 8


class ScreenCapture:
    """
    Captures frames from TrackMania Nations Forever and processes them for
    the neural network and LIDAR observation.
    """

    def __init__(self, monitor=None):
        """
        :param monitor: mss monitor dict.  If None the primary monitor is used.
                        When get_window_geometry() succeeds, the game window rect
                        is used automatically instead.
        """
        self.sct = mss.mss()
        if monitor is None:
            self.monitor = self.sct.monitors[1]   # first physical monitor
        else:
            self.monitor = monitor

    def _get_capture_region(self):
        """Return an mss-compatible region dict, preferring the game window."""
        geom = get_window_geometry(GAME_WINDOW_NAME)
        if geom:
            l, t, r, b = geom
            return {"left": l, "top": t, "width": r - l, "height": b - t}
        return self.monitor

    # ── Raw capture ──────────────────────────────────────────────────────────

    def capture(self):
        """
        Capture the game window (or fallback monitor) as a BGR numpy array.
        """
        region = self._get_capture_region()
        sct_img = self.sct.grab(region)
        img = np.array(sct_img)          # BGRA
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # ── CNN input ────────────────────────────────────────────────────────────

    def capture_preprocessed(self, target_size=(128, 128)):
        """
        Capture and pre-process a frame for stacking into the CNN input.
        Returns a float32 array of shape (H, W, 1) normalised to [0, 1].
        """
        img = self.capture()
        img = cv2.resize(img, target_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.expand_dims(gray, axis=-1).astype(np.float32) / 255.0
        return gray

    # ── Edge-detection pipeline (from TMAI GameCapture.py) ───────────────────

    @staticmethod
    def process_screen(bgr_frame: np.ndarray, out_size=(128, 128)) -> np.ndarray:
        """
        Convert a raw BGR frame into a binary edge image suited for LIDAR
        raycasting.  Pipeline (TMAI-derived):
          grayscale → binary threshold → Canny edges →
          dilation → Gaussian blur → binary threshold

        Returns a uint8 image of shape (H//4, W) – the bottom half is cropped
        so rays only see what is in front of the car.
        """
        img = cv2.resize(bgr_frame, out_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 32, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(bw, 100, 300)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=3)
        edges = cv2.GaussianBlur(edges, (3, 3), 0)
        _, edges = cv2.threshold(edges, 1, 255, cv2.THRESH_BINARY)
        # Keep only the middle strip (what is ahead of the car)
        h = edges.shape[0]
        return edges[h // 2: h // 2 + h // 4, :]

    # ── LIDAR raycasting ─────────────────────────────────────────────────────

    @staticmethod
    def _ray_distance(frame: np.ndarray, angle: float) -> float:
        """
        Cast one ray from the bottom-centre of *frame* at *angle* radians.
        Returns the normalised distance (0–1) to the first bright pixel.
        """
        h, w = frame.shape[:2]
        ref_size = np.hypot(h, w) / 2.0
        cx, cy = w // 2, h - 1
        dx = np.cos(angle)
        dy = np.sin(angle)
        x, y = float(cx), float(cy)
        while 0 <= int(x) < w and 0 <= int(y) < h:
            if frame[int(y), int(x)] > 0:
                break
            x += dx
            y -= dy          # screen Y is inverted vs. math angle
        dist = np.hypot(x - cx, y - cy)
        # scaling that gives more weight to angles closer to vertical (ahead)
        scale = (1 + 3 * np.sin(angle)) / 4.0
        return float(np.clip(scale * dist / ref_size, 0.0, 1.0))

    def get_lidar_obs(self, n_rays: int = 16) -> np.ndarray:
        """
        Capture a frame, process it, cast *n_rays* evenly spaced from 0→π,
        and return a float32 array of shape (n_rays,) with normalised distances.

        A distance close to 0 means a track border is immediately in that
        direction; close to 1 means open road.
        """
        raw = self.capture()
        processed = self.process_screen(raw)
        distances = [
            self._ray_distance(processed, i * np.pi / (n_rays - 1))
            for i in range(n_rays)
        ]
        return np.array(distances, dtype=np.float32)
