"""OAK-D camera capture module."""

from typing import Optional
import numpy as np

try:
    import depthai as dai
    DEPTHAI_AVAILABLE = True
except ImportError:
    DEPTHAI_AVAILABLE = False
    dai = None


class OakDCapture:
    """OAK-D camera interface for RGB + Depth capture."""

    DEMO_MODE = False  # Set True for testing without camera

    # OAK-D S3 Pro specs
    RGB_RESOLUTION = (1920, 1080)
    DEPTH_RESOLUTION = (640, 480)
    FPS = 30

    def __init__(self):
        self.pipeline = None
        self.device = None
        self.rgb_queue = None
        self.depth_queue = None
        self._intrinsics = None

    def initialize(self) -> bool:
        """Initialize camera pipeline."""
        if self.DEMO_MODE:
            print("DEMO MODE: Using simulated camera")
            self._setup_demo_intrinsics()
            return True

        if not DEPTHAI_AVAILABLE:
            print("ERROR: depthai library not installed")
            print("Install with: pip install depthai")
            return False

        # Check for connected devices
        devices = dai.Device.getAllAvailableDevices()
        if not devices:
            print("No OAK-D devices found")
            return False

        print(f"Found {len(devices)} OAK-D device(s):")
        for d in devices:
            print(f"  - {d.getMxId()} ({d.state.name})")

        # Create pipeline
        self.pipeline = dai.Pipeline()

        # RGB camera
        cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(640, 480)  # Match depth for overlay
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam_rgb.setFps(self.FPS)

        # Stereo depth
        mono_left = self.pipeline.create(dai.node.MonoCamera)
        mono_right = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)

        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        mono_left.setCamera("left")
        mono_left.setFps(self.FPS)

        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        mono_right.setCamera("right")
        mono_right.setFps(self.FPS)

        # Stereo depth config - optimized for close-range scanning
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Align to RGB

        # Configure for close-range (sub-1m scanning)
        config = stereo.initialConfig.get()
        config.postProcessing.speckleFilter.enable = True
        config.postProcessing.speckleFilter.speckleRange = 50
        config.postProcessing.temporalFilter.enable = True
        config.postProcessing.spatialFilter.enable = True
        config.postProcessing.spatialFilter.holeFillingRadius = 2
        config.postProcessing.spatialFilter.numIterations = 1
        config.postProcessing.thresholdFilter.minRange = 100  # 10cm min
        config.postProcessing.thresholdFilter.maxRange = 2000  # 2m max
        stereo.initialConfig.set(config)

        # Link nodes
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # Output queues
        xout_rgb = self.pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        # Start device
        try:
            self.device = dai.Device(self.pipeline)
            self.rgb_queue = self.device.getOutputQueue(
                name="rgb", maxSize=4, blocking=False
            )
            self.depth_queue = self.device.getOutputQueue(
                name="depth", maxSize=4, blocking=False
            )

            # Get camera intrinsics
            calib = self.device.readCalibration()
            intrinsics = calib.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_A,
                resizeWidth=640,
                resizeHeight=480,
            )
            self._intrinsics = {
                "fx": intrinsics[0][0],
                "fy": intrinsics[1][1],
                "cx": intrinsics[0][2],
                "cy": intrinsics[1][2],
            }
            print(f"Camera intrinsics: {self._intrinsics}")

            return True

        except Exception as e:
            print(f"Failed to start device: {e}")
            return False

    def _setup_demo_intrinsics(self):
        """Setup fake intrinsics for demo mode."""
        self._intrinsics = {
            "fx": 500.0,
            "fy": 500.0,
            "cx": 320.0,
            "cy": 240.0,
        }

    def get_frame(self) -> Optional[dict]:
        """Get current RGB and depth frame."""
        if self.DEMO_MODE:
            return self._get_demo_frame()

        if not self.device:
            return None

        rgb_msg = self.rgb_queue.tryGet()
        depth_msg = self.depth_queue.tryGet()

        if rgb_msg is None or depth_msg is None:
            return None

        rgb = rgb_msg.getCvFrame()
        depth_raw = depth_msg.getFrame()

        # Convert depth to meters, mark invalid as NaN
        depth = depth_raw.astype(np.float32) / 1000.0
        depth[depth_raw == 0] = np.nan

        return {
            "rgb": rgb,
            "depth": depth,
            "timestamp": rgb_msg.getTimestamp().total_seconds(),
        }

    def _get_demo_frame(self) -> dict:
        """Generate simulated frame for demo mode."""
        import time

        # Create gradient RGB image
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.tile(np.linspace(50, 200, 640), (480, 1))
        rgb[:, :, 1] = np.tile(np.linspace(100, 150, 480), (640, 1)).T
        rgb[:, :, 2] = 100

        # Create depth with some NaN regions (simulating dead zones)
        depth = np.ones((480, 640), dtype=np.float32) * 0.5

        # Add some depth variation
        x, y = np.meshgrid(np.arange(640), np.arange(480))
        depth += 0.1 * np.sin(x / 50 + time.time()) * np.cos(y / 50)

        # Add dead zones (NaN regions) - simulating reflective surfaces
        center_x, center_y = 320, 240
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        dead_zone = (dist < 50) | ((x > 400) & (y > 300))
        depth[dead_zone] = np.nan

        # Add some random dead pixels
        random_dead = np.random.random((480, 640)) < 0.01
        depth[random_dead] = np.nan

        return {
            "rgb": rgb,
            "depth": depth,
            "timestamp": time.time(),
        }

    def get_intrinsics(self) -> dict:
        """Get camera intrinsics for 3D projection."""
        return self._intrinsics or {
            "fx": 500.0,
            "fy": 500.0,
            "cx": 320.0,
            "cy": 240.0,
        }

    def close(self):
        """Release camera resources."""
        if self.device:
            self.device.close()
            self.device = None
        print("Camera closed")
