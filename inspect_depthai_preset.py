import depthai as dai

try:
    print(f"PresetMode dir: {dir(dai.node.StereoDepth.PresetMode)}")
except Exception as e:
    print(f"Error: {e}")
