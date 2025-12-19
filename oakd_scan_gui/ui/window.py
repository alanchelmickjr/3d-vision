"""Main window UI for OAK-D Scan GUI using DearPyGui."""

from typing import Callable, Optional
import numpy as np

import dearpygui.dearpygui as dpg


class ScanWindow:
    """Main scanning window with RGB, depth, and point cloud views."""

    def __init__(
        self,
        on_capture: Callable = None,
        on_auto_toggle: Callable = None,
        on_reset: Callable = None,
        on_export: Callable = None,
    ):
        self.on_capture = on_capture
        self.on_auto_toggle = on_auto_toggle
        self.on_reset = on_reset
        self.on_export = on_export

        # Texture handles
        self._rgb_texture = None
        self._depth_texture = None

        # Image dimensions
        self._width = 640
        self._height = 480

        # State
        self._auto_enabled = False

    def setup(self):
        """Set up the main window and all UI elements."""
        # Create textures for image display
        with dpg.texture_registry():
            # RGB texture (RGBA format for DearPyGui)
            self._rgb_data = np.zeros((self._height, self._width, 4), dtype=np.float32)
            self._rgb_texture = dpg.add_raw_texture(
                self._width,
                self._height,
                self._rgb_data.flatten(),
                format=dpg.mvFormat_Float_rgba,
                tag="rgb_texture",
            )

            # Depth texture
            self._depth_data = np.zeros((self._height, self._width, 4), dtype=np.float32)
            self._depth_texture = dpg.add_raw_texture(
                self._width,
                self._height,
                self._depth_data.flatten(),
                format=dpg.mvFormat_Float_rgba,
                tag="depth_texture",
            )

        # Main window
        with dpg.window(label="OAK-D 3D Scanner", tag="main_window"):
            # Top row: RGB and Depth side by side
            with dpg.group(horizontal=True):
                # RGB feed with overlay
                with dpg.child_window(width=660, height=520):
                    dpg.add_text("RGB + Dead Zone Overlay", color=(200, 200, 200))
                    dpg.add_image("rgb_texture", width=640, height=480)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Legend:", color=(150, 150, 150))
                        dpg.add_text("RED=Dead", color=(255, 100, 100))
                        dpg.add_text("YELLOW=Weak", color=(255, 255, 100))
                        dpg.add_text("GREEN=Good", color=(100, 255, 100))

                # Depth feed
                with dpg.child_window(width=660, height=520):
                    dpg.add_text("Depth Map (Colorized)", color=(200, 200, 200))
                    dpg.add_image("depth_texture", width=640, height=480)

            # Stats bar
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text("Coverage:", color=(150, 150, 150))
                dpg.add_text("---%", tag="coverage_text", color=(100, 255, 100))
                dpg.add_spacer(width=30)
                dpg.add_text("Dead Zones:", color=(150, 150, 150))
                dpg.add_text("---", tag="dead_zones_text", color=(255, 100, 100))
                dpg.add_spacer(width=30)
                dpg.add_text("Points:", color=(150, 150, 150))
                dpg.add_text("0", tag="points_text", color=(100, 200, 255))
                dpg.add_spacer(width=30)
                dpg.add_text("FPS:", color=(150, 150, 150))
                dpg.add_text("--", tag="fps_text", color=(200, 200, 200))

            # Control buttons
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Capture",
                    callback=self._handle_capture,
                    width=100,
                    height=40,
                )
                dpg.add_button(
                    label="Auto: OFF",
                    tag="auto_button",
                    callback=self._handle_auto_toggle,
                    width=100,
                    height=40,
                )
                dpg.add_button(
                    label="Reset",
                    callback=self._handle_reset,
                    width=100,
                    height=40,
                )
                dpg.add_spacer(width=20)
                dpg.add_text("Export:", color=(150, 150, 150))
                dpg.add_button(
                    label=".PLY",
                    callback=lambda: self._handle_export("ply"),
                    width=60,
                    height=40,
                )
                dpg.add_button(
                    label=".STL",
                    callback=lambda: self._handle_export("stl"),
                    width=60,
                    height=40,
                )
                dpg.add_button(
                    label=".OBJ",
                    callback=lambda: self._handle_export("obj"),
                    width=60,
                    height=40,
                )

        # Set primary window
        dpg.set_primary_window("main_window", True)

    def update_rgb(self, image: np.ndarray):
        """Update RGB display with overlay."""
        if image is None:
            return

        # Convert BGR to RGBA float
        if image.shape[2] == 3:
            rgba = np.zeros((self._height, self._width, 4), dtype=np.float32)
            rgba[:, :, :3] = image[:, :, ::-1] / 255.0  # BGR -> RGB
            rgba[:, :, 3] = 1.0
        else:
            rgba = image.astype(np.float32) / 255.0

        dpg.set_value("rgb_texture", rgba.flatten())

    def update_depth(self, depth_colored: np.ndarray):
        """Update depth display."""
        if depth_colored is None:
            return

        # Convert to RGBA float
        rgba = np.zeros((self._height, self._width, 4), dtype=np.float32)
        rgba[:, :, :3] = depth_colored[:, :, ::-1] / 255.0  # BGR -> RGB
        rgba[:, :, 3] = 1.0

        dpg.set_value("depth_texture", rgba.flatten())

    def update_stats(self, stats: dict):
        """Update coverage statistics display."""
        coverage = stats.get("coverage_percent", 0)
        dead_count = stats.get("dead_zone_pixels", 0)
        dead_percent = stats.get("dead_zone_percent", 0)

        # Color code coverage
        if coverage >= 90:
            color = (100, 255, 100)  # Green
        elif coverage >= 70:
            color = (255, 255, 100)  # Yellow
        else:
            color = (255, 100, 100)  # Red

        dpg.set_value("coverage_text", f"{coverage:.1f}%")
        dpg.configure_item("coverage_text", color=color)

        dpg.set_value("dead_zones_text", f"{dead_count:,} ({dead_percent:.1f}%)")

    def update_point_cloud(self, points: np.ndarray, colors: np.ndarray):
        """Update point cloud display."""
        if points is None:
            return

        num_points = len(points)
        dpg.set_value("points_text", f"{num_points:,}")

    def clear_point_cloud(self):
        """Clear the point cloud display."""
        dpg.set_value("points_text", "0")

    def _handle_capture(self):
        """Handle capture button click."""
        if self.on_capture:
            self.on_capture()

    def _handle_auto_toggle(self):
        """Handle auto toggle button click."""
        self._auto_enabled = not self._auto_enabled

        if self._auto_enabled:
            dpg.set_item_label("auto_button", "Auto: ON")
            dpg.configure_item("auto_button", label="Auto: ON")
        else:
            dpg.set_item_label("auto_button", "Auto: OFF")
            dpg.configure_item("auto_button", label="Auto: OFF")

        if self.on_auto_toggle:
            self.on_auto_toggle(self._auto_enabled)

    def _handle_reset(self):
        """Handle reset button click."""
        if self.on_reset:
            self.on_reset()

    def _handle_export(self, format_type: str):
        """Handle export button click."""
        if self.on_export:
            self.on_export(format_type)
