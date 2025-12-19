"""Coverage analysis for depth scanning."""

import numpy as np
from typing import Optional


class CoverageAnalyzer:
    """Analyzes depth coverage and quality metrics."""

    def __init__(self):
        self._frame_count = 0
        self._accumulated_valid_mask = None

    def analyze(
        self,
        depth: np.ndarray,
        dead_zone_mask: np.ndarray,
    ) -> dict:
        """
        Analyze coverage statistics for a depth frame.

        Args:
            depth: Depth image (H, W) in meters
            dead_zone_mask: Boolean mask where True = dead zone

        Returns:
            Dictionary with coverage statistics
        """
        total_pixels = depth.size
        valid_pixels = np.sum(~dead_zone_mask)
        dead_pixels = np.sum(dead_zone_mask)

        coverage_percent = (valid_pixels / total_pixels) * 100
        dead_percent = (dead_pixels / total_pixels) * 100

        # Depth range statistics (only valid pixels)
        valid_depths = depth[~dead_zone_mask]
        if len(valid_depths) > 0:
            depth_min = float(np.nanmin(valid_depths))
            depth_max = float(np.nanmax(valid_depths))
            depth_mean = float(np.nanmean(valid_depths))
            depth_std = float(np.nanstd(valid_depths))
        else:
            depth_min = depth_max = depth_mean = depth_std = 0.0

        return {
            "coverage_percent": coverage_percent,
            "dead_zone_percent": dead_percent,
            "dead_zone_pixels": int(dead_pixels),
            "valid_pixels": int(valid_pixels),
            "total_pixels": int(total_pixels),
            "depth_min_m": depth_min,
            "depth_max_m": depth_max,
            "depth_mean_m": depth_mean,
            "depth_std_m": depth_std,
            "frame_count": self._frame_count,
        }

    def analyze_accumulated(
        self,
        depths: list,
        masks: list,
    ) -> dict:
        """
        Analyze coverage across multiple accumulated frames.

        Args:
            depths: List of depth arrays
            masks: List of dead zone masks

        Returns:
            Accumulated coverage statistics
        """
        if not depths or not masks:
            return {"accumulated_coverage": 0}

        h, w = depths[0].shape
        accumulated_valid = np.zeros((h, w), dtype=np.int32)

        for depth, mask in zip(depths, masks):
            accumulated_valid += (~mask).astype(np.int32)

        # Pixels seen at least once
        seen_once = accumulated_valid > 0
        total_seen = np.sum(seen_once)
        total_pixels = h * w

        # Pixels seen multiple times (higher confidence)
        seen_multiple = accumulated_valid >= len(depths) // 2
        total_multi = np.sum(seen_multiple)

        return {
            "accumulated_coverage": (total_seen / total_pixels) * 100,
            "multi_view_coverage": (total_multi / total_pixels) * 100,
            "total_frames": len(depths),
            "pixels_seen_once": int(total_seen),
            "pixels_multi_view": int(total_multi),
        }

    def get_coverage_map(
        self,
        depths: list,
        masks: list,
    ) -> np.ndarray:
        """
        Generate a coverage heatmap from accumulated frames.

        Args:
            depths: List of depth arrays
            masks: List of dead zone masks

        Returns:
            Coverage count map (H, W) - higher values = more observations
        """
        if not depths:
            return None

        h, w = depths[0].shape
        coverage_map = np.zeros((h, w), dtype=np.int32)

        for mask in masks:
            coverage_map += (~mask).astype(np.int32)

        return coverage_map

    def suggest_scan_angle(
        self,
        dead_zone_mask: np.ndarray,
    ) -> Optional[str]:
        """
        Suggest which direction to move camera based on dead zone location.

        Args:
            dead_zone_mask: Boolean mask where True = dead zone

        Returns:
            String suggestion or None if coverage is good
        """
        if np.mean(dead_zone_mask) < 0.05:
            return None  # Coverage is good

        h, w = dead_zone_mask.shape

        # Analyze quadrants
        top_half = dead_zone_mask[:h // 2, :]
        bottom_half = dead_zone_mask[h // 2:, :]
        left_half = dead_zone_mask[:, :w // 2]
        right_half = dead_zone_mask[:, w // 2:]

        dead_top = np.mean(top_half)
        dead_bottom = np.mean(bottom_half)
        dead_left = np.mean(left_half)
        dead_right = np.mean(right_half)

        suggestions = []

        if dead_top > dead_bottom + 0.1:
            suggestions.append("tilt down")
        elif dead_bottom > dead_top + 0.1:
            suggestions.append("tilt up")

        if dead_left > dead_right + 0.1:
            suggestions.append("pan right")
        elif dead_right > dead_left + 0.1:
            suggestions.append("pan left")

        # Check center vs edges (specular reflection often in center)
        center = dead_zone_mask[h // 4:3 * h // 4, w // 4:3 * w // 4]
        if np.mean(center) > 0.3:
            suggestions.append("change angle (specular reflection detected)")

        return ", ".join(suggestions) if suggestions else None
