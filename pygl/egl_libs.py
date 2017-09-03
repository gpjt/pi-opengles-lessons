import ctypes
import ctypes.util

platform = None

PLATFORM_PI = "pi"
PLATFORM_LINUX = "linux"


if ctypes.util.find_library('bcm_host'):
    platform = PLATFORM_PI
    opengles = ctypes.CDLL('libbrcmGLESv2.so')
    openegl = ctypes.CDLL('libbrcmEGL.so')
    from pi_backend import Backend  # noqa
else:
    platform = PLATFORM_LINUX
    opengles = ctypes.CDLL('libGLESv2.so')
    openegl = ctypes.CDLL('libEGL.so')


def eglints(L):
    """Converts a tuple to an array of eglints (would a pointer return be better?)"""
    return (ctypes.c_int*len(L))(*L)


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