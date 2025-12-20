import depthai as dai

try:
    devices = dai.Device.getAllAvailableDevices()
    if devices:
        d = devices[0]
        print(f"Type: {type(d)}")
        print(f"Dir: {dir(d)}")
    else:
        print("No devices found")
except Exception as e:
    print(f"Error: {e}")
