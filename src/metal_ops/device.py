import platform
import ctypes


def mtl_create_system_default_device_check() -> bool:
    """
    Checks for the presence of a Metal device by calling 
    MTLCreateSystemDefaultDevice() using ctypes.
    """
    # 1. Check OS for macOS compatibility
    if platform.system() != 'Darwin':
        print("Metal is only supported on macOS (Darwin).")
        return False

    # 2. Load necessary system libraries
    try:
        # Load the CoreFoundation framework (needed for the Objective-C Runtime)
        cf = ctypes.CDLL(
            '/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')

        # Load the Metal framework
        metal = ctypes.CDLL('/System/Library/Frameworks/Metal.framework/Metal')
    except OSError as e:
        # This will catch cases where the frameworks are not found (e.g., on Linux/Windows)
        print(f"Error loading system frameworks: {e}")
        return False

    # 3. Get a handle to the Objective-C Runtime function (for message passing)
    # The Objective-C runtime function used to call methods on objects
    objc_msgSend = ctypes.CFUNCTYPE(
        ctypes.c_void_p,         # Return type: id (pointer to object)
        # Argument 1: id (The receiver object, often the class)
        ctypes.c_void_p,
        # Argument 2: SEL (The selector, i.e., the method name)
        ctypes.c_void_p,
        # ... any subsequent arguments ...
    )(metal.objc_msgSend)

    # 4. Get the Selector for the function
    # In Objective-C, you need to get the selector (the internal name) for the method you want to call.
    sel_registerName = metal.sel_registerName
    sel_registerName.restype = ctypes.c_void_p
    # The selector for the standard Metal device creation function
    sel_device_name = b"MTLCreateSystemDefaultDevice"
    sel_device = sel_registerName(sel_device_name)

    # 5. Call MTLCreateSystemDefaultDevice
    # This function is usually not called via objc_msgSend, but is a straight C function export.
    # However, for simplicity and stability across minor OS versions, we'll try to find
    # the exported C function symbol directly.

    try:
        # Define the C-function signature:
        # id<MTLDevice> MTLCreateSystemDefaultDevice(void);
        MTLCreateSystemDefaultDevice = metal.MTLCreateSystemDefaultDevice
        # returns a pointer to the MTLDevice object
        MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        MTLCreateSystemDefaultDevice.argtypes = []  # takes no arguments

        # Execute the function
        device_pointer = MTLCreateSystemDefaultDevice()

        # Check the result
        if device_pointer:
            return True
        else:
            return False

    except AttributeError:
        # Fallback if the function is not exported as a simple C function (less common)
        print("Could not find MTLCreateSystemDefaultDevice symbol. Metal API might be missing or changed.")
        return False


# --- Execution ---
is_metal_available = mtl_create_system_default_device_check()

print("\n--- Metal Availability Check via ctypes ---")
print(f"Result of MTLCreateSystemDefaultDevice call: **{is_metal_available}**")
if is_metal_available:
    print("Metal device is detected.")
else:
    print("No Metal device detected or API access failed.")
