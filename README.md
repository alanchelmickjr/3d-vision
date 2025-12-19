# 3D Vision - OAK-D Scan GUI

Real-time 3D scanning GUI for OAK-D cameras with dead zone visualization. Built for manufacturing parts from robot scans.

## Features

- **Live RGB + Depth feeds** with dead zone overlay
- **Dead zone detection** (NaN/no-return regions shown in RED)
- **Weak zone detection** (noisy/low-confidence regions shown in YELLOW)
- **Coverage percentage** tracking
- **Accumulated point cloud** from multiple angles
- **Export** to PLY, STL, OBJ formats

## UI Layout

```
┌────────────────────┬────────────────────┐
│  RGB Feed          │  Depth (colorized) │
│  + OVERLAY:        │                    │
│  RED = dead zones  │                    │
│  YELLOW = weak     │                    │
├────────────────────┴────────────────────┤
│  Coverage: XX% | Dead zones: XX         │
├─────────────────────────────────────────┤
│  [Capture] [Auto] [Reset] [Export]      │
└─────────────────────────────────────────┘
```

## Installation

### M4 Mac (Apple Silicon)

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package
pip install -e .

# Or install dependencies directly
pip install depthai opencv-python open3d numpy dearpygui
```

### Jetson Orin

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install depthai (NVIDIA-specific build)
pip install depthai

# Install other dependencies
pip install opencv-python numpy

# Open3D on Jetson (use pip or build from source)
pip install open3d

# DearPyGui on Jetson
pip install dearpygui
```

## Usage

### Quick Start

```bash
# With OAK-D camera connected via USB-C
oakd-scan

# Demo mode (no camera required)
oakd-scan --demo

# Specify output directory
oakd-scan -o ./my_scans
```

### Controls

| Button | Action |
|--------|--------|
| **Capture** | Take single frame, add to point cloud |
| **Auto** | Toggle continuous capture mode |
| **Reset** | Clear accumulated point cloud |
| **.PLY/.STL/.OBJ** | Export current point cloud |

### Scanning Tips

1. **Shiny surfaces** (aluminum, chrome) cause specular reflection = dead zones
   - Scan from multiple angles
   - Use diffuse lighting or "sub-human vision" lights (IR)
   - Matte spray for critical parts

2. **Multi-pass scanning** - hit each area from 3+ angles to fill gaps

3. **Watch the coverage %** - aim for >90% before export

## Project Structure

```
3d-vision/
├── oakd_scan_gui/
│   ├── app.py              # Main application entry
│   ├── ui/
│   │   ├── window.py       # DearPyGui window
│   │   └── overlays.py     # Overlay rendering
│   ├── capture/
│   │   └── oak.py          # OAK-D camera interface
│   ├── analysis/
│   │   ├── coverage.py     # Coverage statistics
│   │   └── dead_zones.py   # Dead zone detection
│   └── export/
│       └── mesh.py         # PLY/STL/OBJ export
├── pyproject.toml
└── README.md
```

## Hardware Setup

### OAK-D S3 Pro Connection

1. Connect OAK-D via USB-C (use USB 3.0+ port)
2. Verify detection:
   ```bash
   python -c "import depthai; print(depthai.Device.getAllAvailableDevices())"
   ```

### Recommended Configuration

- **Distance**: 20-50cm for small parts
- **Lighting**: Diffuse, avoid direct point lights
- **Background**: Matte dark surface

## API Usage

```python
from oakd_scan_gui.capture.oak import OakDCapture
from oakd_scan_gui.analysis.dead_zones import DeadZoneDetector
from oakd_scan_gui.export.mesh import MeshExporter

# Initialize camera
capture = OakDCapture()
capture.initialize()

# Get frame
frame = capture.get_frame()
rgb, depth = frame["rgb"], frame["depth"]

# Detect dead zones
detector = DeadZoneDetector()
dead_mask = detector.detect(depth)

# Export points
exporter = MeshExporter("./scans")
# ... accumulate points and export
```

## Daredevil Integration

This module is designed as a building block for the Daredevil robot project:

- Robots scan environment and parts
- Generate 3D models for fabrication
- Portable across M4 Mac (development) and Jetson Orin (deployment)

## Troubleshooting

### Camera not detected

```bash
# Check USB connection
lsusb | grep Luxonis

# Check depthai can see device
python -c "import depthai; print(depthai.Device.getAllAvailableDevices())"
```

### High dead zone percentage

- Change camera angle
- Add diffuse lighting
- Move closer/further from subject
- Check for IR interference

### Performance issues

- Reduce resolution in `oak.py`
- Disable temporal filtering
- Close other GPU applications

## License

MIT
