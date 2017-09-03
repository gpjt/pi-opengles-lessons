import ctypes
import ctypes.util

bcm_name = ctypes.util.find_library('bcm_host')
bcm = ctypes.CDLL(name)
opengles = ctypes.CDLL('libbrcmGLESv2.so')
openegl = ctypes.CDLL('libbrcmEGL.so')

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