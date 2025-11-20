import ctypes
import platform


def metal_is_available(dll_path: str) -> bool:
    """
    Checks for the presence of a Metal device by directly calling 
    MTLCreateSystemDefaultDevice() from the Metal framework.
    """

    try:
        metal = ctypes.CDLL(dll_path)

        MTLCreateSystemDefaultDevice = metal.MTLCreateSystemDefaultDevice

        # It returns a pointer (id<MTLDevice>), which is a void pointer in ctypes.
        MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p

        # It takes no arguments.
        MTLCreateSystemDefaultDevice.argtypes = []

        # 4. Execute the function
        device_pointer = MTLCreateSystemDefaultDevice()

        # 5. Check the result: A non-null pointer means a device is available.
        return bool(device_pointer)

    except Exception:
        # Catch exceptions if the framework file is missing or the symbol isn't found
        return False


dll_metal = '/System/Library/Frameworks/Metal.framework/Metal'
metal = metal_is_available(dll_metal)
print(f'Metal: {metal}')
