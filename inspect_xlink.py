gimport depthai as dai
import inspect

# Check dai.node
if hasattr(dai, 'node'):
    print("dai.node exists")
    if hasattr(dai.node, 'XLinkOut'):
        print("dai.node.XLinkOut exists")
    else:
        print("dai.node.XLinkOut DOES NOT EXIST")
else:
    print("dai.node DOES NOT EXIST")

# Check dai.XLinkOut directly
if hasattr(dai, 'XLinkOut'):
    print("dai.XLinkOut exists")
