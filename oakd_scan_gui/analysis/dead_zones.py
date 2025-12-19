"""Dead zone detection for depth scanning."""

import numpy as np
import cv2
from typing import Tuple


class DeadZoneDetector:
    """Detects and classifies dead zones in depth data."""

    # Thresholds
    NOISE_THRESHOLD = 0.02  # 2cm standard deviation = noisy
    CONFIDENCE_THRESHOLD = 0.7  # Below this = weak confidence
    EDGE_KERNEL_SIZE = 5

    def __init__(self):
        self._history = []
        self._history_size = 5

    def detect(self, depth: np.ndarray) -> np.ndarray:
        """
        Detect dead zones (NaN/invalid) in depth map.

        Args:
            depth: Depth image (H, W) in meters, NaN = no return

        Returns:
            Boolean mask where True = dead zone
        """
        # Primary dead zone: NaN or zero values
        dead_mask = np.isnan(depth) | (depth <= 0)

        return dead_mask

    def detect_weak_zones(
        self,
        depth: np.ndarray,
        dead_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Detect weak/noisy zones that have depth but low confidence.

        Args:
            depth: Depth image (H, W) in meters
            dead_mask: Boolean mask of dead zones

        Returns:
            Boolean mask where True = weak/noisy zone
        """
        # Add to history for temporal analysis
        self._history.append(depth.copy())
        if len(self._history) > self._history_size:
            self._history.pop(0)

        # Need multiple frames for noise analysis
        if len(self._history) < 3:
            return np.zeros_like(dead_mask)

        # Stack history and compute temporal variance
        stack = np.stack(self._history, axis=0)

        # Compute variance at each pixel (ignoring NaN)
        with np.errstate(all='ignore'):
            temporal_std = np.nanstd(stack, axis=0)

        # High variance = noisy/weak
        weak_mask = temporal_std > self.NOISE_THRESHOLD

        # Exclude already-dead zones
        weak_mask = weak_mask & ~dead_mask

        # Also check edges (depth discontinuities are unreliable)
        edge_mask = self._detect_depth_edges(depth)
        weak_mask = weak_mask | (edge_mask & ~dead_mask)

        return weak_mask

    def _detect_depth_edges(self, depth: np.ndarray) -> np.ndarray:
        """Detect edges in depth map (unreliable regions)."""
        # Fill NaN for gradient computation
        depth_filled = np.nan_to_num(depth, nan=0)

        # Sobel gradients
        grad_x = cv2.Sobel(depth_filled, cv2.CV_64F, 1, 0, ksize=self.EDGE_KERNEL_SIZE)
        grad_y = cv2.Sobel(depth_filled, cv2.CV_64F, 0, 1, ksize=self.EDGE_KERNEL_SIZE)

        # Gradient magnitude
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Threshold for significant edges (depth jumps > 5cm)
        edge_threshold = 0.05
        edge_mask = grad_mag > edge_threshold

        # Dilate to mark neighborhood of edges
        kernel = np.ones((3, 3), dtype=np.uint8)
        edge_mask = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        return edge_mask

    def create_overlay(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        dead_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Create RGB overlay with dead zone visualization.

        Args:
            rgb: Original RGB image (H, W, 3)
            depth: Depth image (H, W)
            dead_mask: Boolean dead zone mask

        Returns:
            RGB image with overlay
        """
        result = rgb.copy()

        # Detect weak zones
        weak_mask = self.detect_weak_zones(depth, dead_mask)

        # Create colored overlay
        overlay = result.copy()

        # Red for dead zones
        overlay[dead_mask] = [255, 0, 0]  # RGB red

        # Yellow for weak zones
        overlay[weak_mask] = [255, 255, 0]  # RGB yellow

        # Blend with original
        alpha = 0.4
        result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)

        # Draw contours around dead zones for visibility
        dead_uint8 = (dead_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            dead_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, (255, 0, 0), 1)

        return result

    def classify_dead_zones(
        self,
        depth: np.ndarray,
        dead_mask: np.ndarray,
    ) -> dict:
        """
        Classify dead zones by likely cause.

        Args:
            depth: Depth image
            dead_mask: Boolean dead zone mask

        Returns:
            Dictionary with dead zone classifications
        """
        h, w = depth.shape

        # Find connected components in dead zone mask
        dead_uint8 = (dead_mask * 255).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dead_uint8, connectivity=8
        )

        classifications = {
            "specular": 0,  # Large central blobs (shiny surfaces)
            "edge_dropout": 0,  # At image edges
            "scattered": 0,  # Small scattered pixels
            "total_regions": num_labels - 1,  # Exclude background
        }

        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]

            # Check if at edge
            x, y, bw, bh = stats[i, :4]
            at_edge = (x == 0 or y == 0 or x + bw >= w or y + bh >= h)

            if at_edge:
                classifications["edge_dropout"] += 1
            elif area > 500:  # Large blob
                classifications["specular"] += 1
            else:
                classifications["scattered"] += 1

        return classifications

    def get_improvement_suggestion(
        self,
        depth: np.ndarray,
        dead_mask: np.ndarray,
    ) -> str:
        """
        Suggest how to reduce dead zones.

        Args:
            depth: Depth image
            dead_mask: Boolean dead zone mask

        Returns:
            Suggestion string
        """
        classification = self.classify_dead_zones(depth, dead_mask)
        dead_percent = np.mean(dead_mask) * 100

        if dead_percent < 5:
            return "Coverage looks good!"

        suggestions = []

        if classification["specular"] > 0:
            suggestions.append(
                "Specular reflections detected - try changing viewing angle "
                "or using sub-human-vision lights"
            )

        if classification["edge_dropout"] > 0:
            suggestions.append("Edge dropouts - adjust camera position")

        if classification["scattered"] > 5:
            suggestions.append(
                "Scattered noise - move closer or adjust stereo settings"
            )

        if not suggestions:
            suggestions.append(
                f"{dead_percent:.0f}% dead zones - try multi-angle scanning"
            )

        return " | ".join(suggestions)
