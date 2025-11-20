import os
import numpy as np
import cv2
from typing import Dict, Tuple, Optional

class AdaptiveHSVExtractor:
    """Use dataset-level HSV stats (from scripts/compute_hsv_stats.py) to build adaptive HSV masks.

    Fallback gracefully to fixed heuristics if stats not found or invalid.

    Optional overrides allow grid-search tuning without regenerating stats:
      - override_s_low_thr: float, saturation low threshold
      - override_h_low: float, hue lower bound (0-179 OpenCV scale)
      - override_h_high: float, hue upper bound (0-179 OpenCV scale)
      - override_v_half_width: int, half-width around V-peak for the window
    """
    def __init__(self, stats_path: Optional[str] = None,
                 override_s_low_thr: Optional[float] = None,
                 override_h_low: Optional[float] = None,
                 override_h_high: Optional[float] = None,
                 override_v_half_width: Optional[int] = None):
        # default search path: env ALT_HSV_STATS or data/RoadOnly/hsv_stats.npz
        self.stats_path = stats_path or os.environ.get('ALT_HSV_STATS') or os.path.join('data', 'RoadOnly', 'hsv_stats.npz')
        self.stats: Dict[str, np.ndarray] = {}
        self.ok = False
        # overrides
        self.ovr_s = override_s_low_thr
        self.ovr_hl = override_h_low
        self.ovr_hh = override_h_high
        self.ovr_vhw = override_v_half_width
        try:
            if os.path.isfile(self.stats_path):
                self.stats = dict(np.load(self.stats_path))
                self.ok = True
        except Exception:
            self.ok = False

    def _fixed_params(self) -> Tuple[float, float, float, int, int, int]:
        # same defaults as current code: S<50, H in [65,133], V around peak +-15
        return 50.0, 65.0, 133.0, 128, 113, 143

    def _get_params(self, v_channel: np.ndarray) -> Tuple[float, float, float, int, int, int]:
        if not self.ok:
            return self._fixed_params()
        s_low_thr = float(self.stats.get('s_low_thr', 50.0))
        h_low = float(self.stats.get('h_low', 65.0))
        h_high = float(self.stats.get('h_high', 133.0))
        # Prefer refined peak if available
        v_peak_ds = int(self.stats.get('v_peak_refined', self.stats.get('v_peak', 128)))
        # if override half-width provided, recompute dataset window around v_peak
        if self.ovr_vhw is not None:
            hw = int(max(0, self.ovr_vhw))
            v_low_ds = int(max(0, v_peak_ds - hw))
            v_high_ds = int(min(255, v_peak_ds + hw))
        else:
            # Window mode from env: legacy (std-clamped) vs mad
            v_win_mode = (os.environ.get('ALT_V_WINDOW_MODE', 'legacy') or 'legacy').lower()
            if v_win_mode == 'mad' and ('v_low_mad' in self.stats and 'v_high_mad' in self.stats):
                v_low_ds = int(self.stats.get('v_low_mad'))
                v_high_ds = int(self.stats.get('v_high_mad'))
            else:
                v_low_ds = int(self.stats.get('v_low', max(0, v_peak_ds - 15)))
                v_high_ds = int(self.stats.get('v_high', min(255, v_peak_ds + 15)))
        # apply scalar overrides for S/H if provided
        if self.ovr_s is not None:
            s_low_thr = float(self.ovr_s)
        if self.ovr_hl is not None:
            h_low = float(self.ovr_hl)
        if self.ovr_hh is not None:
            h_high = float(self.ovr_hh)
        # clamp H bounds
        h_low = float(np.clip(h_low, 0.0, 179.0))
        h_high = float(np.clip(h_high, 0.0, 179.0))
        if h_high < h_low:
            # swap to ensure valid interval
            h_low, h_high = h_high, h_low
        # align image peak to dataset peak shift (robust to bright tails)
        hist, _ = np.histogram(v_channel, bins=256, range=(0,256))
        peak_img = int(np.argmax(hist))
        shift = peak_img - v_peak_ds
        v_low = int(np.clip(v_low_ds + shift, 0, 255))
        v_high = int(np.clip(v_high_ds + shift, 0, 255))
        v_peak = int(np.clip(v_peak_ds + shift, 0, 255))
        return s_low_thr, h_low, h_high, v_peak, v_low, v_high

    def build_masks(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s_low_thr, h_low, h_high, v_peak, v_low, v_high = self._get_params(v)
        # masks with smoothing to create soft indicators
        s_mask = cv2.GaussianBlur((s.astype(np.float32) < s_low_thr).astype(np.float32), (5,5), 0)
        h_mask = ((h.astype(np.float32) >= h_low) & (h.astype(np.float32) <= h_high)).astype(np.float32)
        h_mask = cv2.GaussianBlur(h_mask, (5,5), 0)
        v_mask = ((v.astype(np.float32) >= v_low) & (v.astype(np.float32) <= v_high)).astype(np.float32)
        v_mask = cv2.GaussianBlur(v_mask, (5,5), 0)
        return s_mask, h_mask, v_mask

__all__ = ['AdaptiveHSVExtractor']
