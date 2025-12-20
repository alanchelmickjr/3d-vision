import depthai as dai
import inspect

print("Searching for XLinkOut in depthai module...")

def recursive_search(obj, name, path, depth=0):
    if depth > 3: return
    try:
        for attr in dir(obj):
            if attr.startswith('_'): continue
            val = getattr(obj, attr)
            current_path = f"{path}.{attr}"
            
            if name.lower() in attr.lower():
                print(f"FOUND MATCH: {current_path}")
            
            if inspect.ismodule(val) or inspect.isclass(val):
                # Don't recurse into common system modules
                if hasattr(val, '__name__') and 'depthai' in val.__name__:
                     recursive_search(val, name, current_path, depth+1)
    except Exception:
        pass

recursive_search(dai, 'XLinkOut', 'dai')

print("\nChecking dai.node contents explicitly:")
if hasattr(dai, 'node'):
    print(dir(dai.node))
else:
    print("dai.node does not exist")
