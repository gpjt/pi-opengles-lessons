import ctypes
import ctypes.util

platform = None

PLATFORM_PI = "pi"
PLATFORM_LINUX = "linux"


bcm_name = ctypes.util.find_library('bcm_host')
if bcm_name:
    platform = PLATFORM_PI
    bcm = ctypes.CDLL(bcm_name)
    opengles = ctypes.CDLL('libbrcmGLESv2.so')
    openegl = ctypes.CDLL('libbrcmEGL.so')
else:
    platform = PLATFORM_LINUX
    bcm = None
    opengles = ctypes.CDLL('libGLESv2.so')
    openegl = ctypes.CDLL('libEGL.so')


openegl.eglGetDisplay.restype = ctypes.c_void_p

openegl.eglInitialize.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
]

openegl.eglChooseConfig.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
openegl.eglChooseConfig.restype = ctypes.c_int

openegl.eglCreateContext.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_void_p,
]
openegl.eglCreateContext.restype = ctypes.c_void_p

openegl.eglCreateWindowSurface.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int32,
]
openegl.eglCreateWindowSurface.restype = ctypes.c_void_p

openegl.eglMakeCurrent.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p
]