import depthai as dai
import inspect

print(f"DepthAI Version: {dai.__version__}")

pipeline = dai.Pipeline()

print("\n--- Pipeline Attributes (dir) ---")
for attr in dir(pipeline):
    if 'create' in attr:
        print(attr)

print("\n--- dai.node Attributes (dir) ---")
if hasattr(dai, 'node'):
    for attr in dir(dai.node):
        if 'XLink' in attr:
            print(attr)
else:
    print("dai.node does not exist")

print("\n--- dai Attributes (dir) ---")
for attr in dir(dai):
    if 'XLink' in attr:
        print(attr)
