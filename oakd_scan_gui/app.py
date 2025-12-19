"""Main application entry point for OAK-D Scan GUI."""

import argparse
import sys
from pathlib import Path

import dearpygui.dearpygui as dpg
import numpy as np

from oakd_scan_gui.capture.oak import OakDCapture
from oakd_scan_gui.ui.window import ScanWindow
from oakd_scan_gui.analysis.coverage import CoverageAnalyzer
from oakd_scan_gui.analysis.dead_zones import DeadZoneDetector
from oakd_scan_gui.export.mesh import MeshExporter


class OakDScanApp:
    """Main application class for OAK-D 3D scanning."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("./scans")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Core components
        self.capture = None
        self.window = None
        self.coverage_analyzer = CoverageAnalyzer()
        self.dead_zone_detector = DeadZoneDetector()
        self.exporter = MeshExporter(self.output_dir)

        # State
        self.running = False
        self.auto_capture = False
        self.accumulated_points = None
        self.accumulated_colors = None

    def initialize(self) -> bool:
        """Initialize camera and UI."""
        print("Initializing OAK-D camera...")
        try:
            self.capture = OakDCapture()
            if not self.capture.initialize():
                print("ERROR: Failed to initialize OAK-D camera")
                print("Make sure the camera is connected via USB-C")
                return False
            print("Camera initialized successfully")
        except Exception as e:
            print(f"ERROR: Camera initialization failed: {e}")
            return False

        print("Initializing GUI...")
        dpg.create_context()
        self.window = ScanWindow(
            on_capture=self._on_capture,
            on_auto_toggle=self._on_auto_toggle,
            on_reset=self._on_reset,
            on_export=self._on_export,
        )
        self.window.setup()
        print("GUI initialized")

        return True

    def run(self):
        """Main application loop."""
        self.running = True
        print("Starting scan loop... Press Ctrl+C to exit")

        dpg.create_viewport(title="OAK-D 3D Scanner", width=1400, height=900)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        while dpg.is_dearpygui_running() and self.running:
            self._update_frame()
            dpg.render_dearpygui_frame()

        self.cleanup()

    def _update_frame(self):
        """Process and display one frame."""
        frame_data = self.capture.get_frame()
        if frame_data is None:
            return

        rgb, depth = frame_data["rgb"], frame_data["depth"]

        # Detect dead zones and analyze coverage
        dead_zone_mask = self.dead_zone_detector.detect(depth)
        coverage_stats = self.coverage_analyzer.analyze(depth, dead_zone_mask)

        # Create overlay visualization
        overlay = self.dead_zone_detector.create_overlay(rgb, depth, dead_zone_mask)

        # Update UI
        self.window.update_rgb(overlay)
        self.window.update_depth(self._colorize_depth(depth))
        self.window.update_stats(coverage_stats)

        # Auto capture if enabled
        if self.auto_capture:
            self._accumulate_points(frame_data)
            self.window.update_point_cloud(
                self.accumulated_points, self.accumulated_colors
            )

    def _colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Colorize depth map with dead zone highlighting."""
        # Normalize depth for visualization
        valid_mask = ~np.isnan(depth) & (depth > 0)
        if not np.any(valid_mask):
            return np.zeros((*depth.shape, 3), dtype=np.uint8)

        normalized = np.zeros_like(depth)
        normalized[valid_mask] = (
            (depth[valid_mask] - depth[valid_mask].min())
            / (depth[valid_mask].max() - depth[valid_mask].min() + 1e-6)
            * 255
        )

        # Apply colormap
        import cv2
        colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_TURBO)

        # Mark dead zones in red
        dead_mask = ~valid_mask
        colored[dead_mask] = [0, 0, 255]  # BGR red

        return colored

    def _on_capture(self):
        """Handle manual capture button."""
        frame_data = self.capture.get_frame()
        if frame_data:
            self._accumulate_points(frame_data)
            self.window.update_point_cloud(
                self.accumulated_points, self.accumulated_colors
            )
        print("Frame captured")

    def _on_auto_toggle(self, enabled: bool):
        """Handle auto capture toggle."""
        self.auto_capture = enabled
        print(f"Auto capture: {'ON' if enabled else 'OFF'}")

    def _on_reset(self):
        """Reset accumulated point cloud."""
        self.accumulated_points = None
        self.accumulated_colors = None
        self.window.clear_point_cloud()
        print("Point cloud reset")

    def _on_export(self, format_type: str):
        """Export accumulated point cloud."""
        if self.accumulated_points is None:
            print("No points to export")
            return

        output_path = self.exporter.export(
            self.accumulated_points,
            self.accumulated_colors,
            format_type=format_type,
        )
        print(f"Exported to: {output_path}")

    def _accumulate_points(self, frame_data: dict):
        """Add frame points to accumulated cloud."""
        points = frame_data.get("points")
        colors = frame_data.get("colors")

        if points is None:
            # Generate points from depth
            depth = frame_data["depth"]
            rgb = frame_data["rgb"]
            intrinsics = self.capture.get_intrinsics()
            points, colors = self._depth_to_points(depth, rgb, intrinsics)

        if self.accumulated_points is None:
            self.accumulated_points = points
            self.accumulated_colors = colors
        else:
            self.accumulated_points = np.vstack([self.accumulated_points, points])
            self.accumulated_colors = np.vstack([self.accumulated_colors, colors])

    def _depth_to_points(
        self, depth: np.ndarray, rgb: np.ndarray, intrinsics: dict
    ) -> tuple:
        """Convert depth map to 3D points."""
        h, w = depth.shape
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]

        # Create pixel coordinate grids
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Valid depth mask
        valid = ~np.isnan(depth) & (depth > 0)

        # Compute 3D coordinates
        z = depth[valid]
        x = (u[valid] - cx) * z / fx
        y = (v[valid] - cy) * z / fy

        points = np.stack([x, y, z], axis=-1)
        colors = rgb[valid] / 255.0  # Normalize to 0-1

        return points, colors

    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        if self.capture:
            self.capture.close()
        dpg.destroy_context()
        print("Done")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OAK-D 3D Scanner with dead zone visualization"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("./scans"),
        help="Output directory for exports (default: ./scans)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with simulated camera",
    )
    args = parser.parse_args()

    # Demo mode for testing without camera
    if args.demo:
        print("Running in DEMO mode (simulated camera)")
        from oakd_scan_gui.capture.oak import OakDCapture
        OakDCapture.DEMO_MODE = True

    app = OakDScanApp(output_dir=args.output)

    if not app.initialize():
        print("\nTroubleshooting:")
        print("  1. Check USB-C connection to OAK-D camera")
        print("  2. Run: python -c \"import depthai; print(depthai.Device.getAllAvailableDevices())\"")
        print("  3. Try: oakd-scan --demo  (for testing without camera)")
        sys.exit(1)

    try:
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()
