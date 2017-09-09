import ctypes
import ctypes.util


def eglints(L):
    return (ctypes.c_int * len(L))(*L)


if ctypes.util.find_library('bcm_host'):
    opengles = ctypes.CDLL('libbrcmGLESv2.so')
    openegl = ctypes.CDLL('libbrcmEGL.so')
    from pygl.pi_backend import Backend  # NOQA
else:
    opengles = ctypes.CDLL('libGLESv2.so')
    openegl = ctypes.CDLL('libEGL.so')
    from pygl.linux_backend import Backend  # NOQA


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

openegl.eglSwapBuffers.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p
]

opengles.glUniformMatrix4fv.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_ubyte,
    ctypes.c_void_p
]
opengles.glUniformMatrix3fv.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_ubyte,
    ctypes.c_void_p
]