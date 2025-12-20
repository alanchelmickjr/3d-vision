import depthai as dai

stereo = dai.node.StereoDepth()
print(f"Stereo init config type: {type(stereo.initialConfig)}")
print(f"Stereo init config dir: {dir(stereo.initialConfig)}")
