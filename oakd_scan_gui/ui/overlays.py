"""Overlay rendering utilities for dead zone visualization."""

import numpy as np
import cv2


class OverlayRenderer:
    """Renders colored overlays on images for dead zone visualization."""

    # Overlay colors (BGR format for OpenCV)
    COLOR_DEAD = (0, 0, 255)      # Red - NaN/no return
    COLOR_WEAK = (0, 255, 255)    # Yellow - low confidence
    COLOR_GOOD = (0, 255, 0)      # Green - good coverage

    # Overlay opacity
    ALPHA = 0.4

    @classmethod
    def create_dead_zone_overlay(
        cls,
        rgb: np.ndarray,
        dead_mask: np.ndarray,
        weak_mask: np.ndarray = None,
    ) -> np.ndarray:
        """
        Create RGB image with dead zone overlay.

        Args:
            rgb: Original RGB image (H, W, 3)
            dead_mask: Boolean mask where True = dead zone
            weak_mask: Optional boolean mask where True = weak/noisy

        Returns:
            RGB image with colored overlay
        """
        result = rgb.copy()

        # Apply dead zone overlay (red)
        if dead_mask is not None and np.any(dead_mask):
            overlay = result.copy()
            overlay[dead_mask] = cls.COLOR_DEAD
            result = cv2.addWeighted(result, 1 - cls.ALPHA, overlay, cls.ALPHA, 0)

        # Apply weak zone overlay (yellow)
        if weak_mask is not None and np.any(weak_mask):
            overlay = result.copy()
            overlay[weak_mask] = cls.COLOR_WEAK
            result = cv2.addWeighted(result, 1 - cls.ALPHA, overlay, cls.ALPHA, 0)

        return result

    @classmethod
    def create_coverage_heatmap(
        cls,
        depth: np.ndarray,
        dead_mask: np.ndarray,
        weak_mask: np.ndarray = None,
    ) -> np.ndarray:
        """
        Create a coverage quality heatmap.

        Args:
            depth: Depth image (H, W)
            dead_mask: Boolean mask where True = dead zone
            weak_mask: Optional boolean mask where True = weak/noisy

        Returns:
            BGR heatmap image
        """
        h, w = depth.shape
        heatmap = np.zeros((h, w, 3), dtype=np.uint8)

        # Good coverage (green)
        good_mask = ~dead_mask
        if weak_mask is not None:
            good_mask = good_mask & ~weak_mask
        heatmap[good_mask] = cls.COLOR_GOOD

        # Weak coverage (yellow)
        if weak_mask is not None:
            heatmap[weak_mask] = cls.COLOR_WEAK

        # Dead zones (red)
        heatmap[dead_mask] = cls.COLOR_DEAD

        return heatmap

    @classmethod
    def draw_contours(
        cls,
        image: np.ndarray,
        mask: np.ndarray,
        color: tuple = (0, 0, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw contours around masked regions.

        Args:
            image: Image to draw on
            mask: Boolean mask
            color: Contour color (BGR)
            thickness: Line thickness

        Returns:
            Image with contours drawn
        """
        result = image.copy()
        mask_uint8 = (mask * 255).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(result, contours, -1, color, thickness)

        return result

    @classmethod
    def add_grid_overlay(
        cls,
        image: np.ndarray,
        grid_size: int = 50,
        color: tuple = (100, 100, 100),
    ) -> np.ndarray:
        """
        Add grid lines to image for spatial reference.

        Args:
            image: Image to draw on
            grid_size: Grid cell size in pixels
            color: Grid line color (BGR)

        Returns:
            Image with grid overlay
        """
        result = image.copy()
        h, w = result.shape[:2]

        # Vertical lines
        for x in range(0, w, grid_size):
            cv2.line(result, (x, 0), (x, h), color, 1)

        # Horizontal lines
        for y in range(0, h, grid_size):
            cv2.line(result, (0, y), (w, y), color, 1)

        return result

    @classmethod
    def colorize_point_cloud(
        cls,
        points: np.ndarray,
        colors: np.ndarray,
        dead_mask: np.ndarray = None,
    ) -> np.ndarray:
        """
        Colorize point cloud with coverage quality.

        Args:
            points: 3D points (N, 3)
            colors: Original RGB colors (N, 3)
            dead_mask: Optional mask for dead zone points

        Returns:
            Modified colors array
        """
        result = colors.copy()

        if dead_mask is not None:
            # Convert BGR to RGB and normalize
            red = np.array([1.0, 0.0, 0.0])  # RGB red for dead zones
            result[dead_mask] = red

        return result
