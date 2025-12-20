import depthai as dai
pipeline = dai.Pipeline()

try:
    print("Attempting pipeline.create('XLinkOut')...")
    # Some bindings allow string names
    # node = pipeline.create("XLinkOut") 
    # But depthai typically uses class types. 
    pass
except Exception as e:
    print(f"Failed: {e}")

try:
    print("\nAttempting to find XLinkOut in dai...")
    found = False
    for attr_name in dir(dai):
        attr = getattr(dai, attr_name)
        if 'XLinkOut' in attr_name:
             print(f"Found in dai: {attr_name}")
             found = True
    if not found:
        print("Not found in top level 'dai' module")

    if hasattr(dai, 'node'):
        print("\nAttempting to find XLinkOut in dai.node...")
        for attr_name in dir(dai.node):
            if 'XLinkOut' in attr_name:
                print(f"Found in dai.node: {attr_name}")
    else:
        print("\ndai.node does not exist")

except Exception as e:
    print(f"Failed: {e}")
