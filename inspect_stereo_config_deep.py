import depthai as dai
import inspect

stereo = dai.node.StereoDepth()
print(f"Stereo initialConfig type: {type(stereo.initialConfig)}")
print("Properties/Methods:")
for name in dir(stereo.initialConfig):
    if not name.startswith("__"):
        print(f"  {name}")
