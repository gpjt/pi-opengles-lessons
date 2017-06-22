import ctypes
import platform

# Pick up our constants extracted from the header files with prepare_constants.py
from pygl import egl_constants

# Define some extra constants that the automatic extraction misses
egl_constants.EGL_DEFAULT_DISPLAY = 0
egl_constants.EGL_NO_CONTEXT = 0
egl_constants.EGL_NO_DISPLAY = 0
egl_constants.EGL_NO_SURFACE = 0
egl_constants.DISPMANX_PROTECTION_NONE = 0

from pygl.gl import GL

PLATFORM_PI = "pi"
PLATFORM_LINUX = "linux"
if platform.system().lower() != "linux":
    raise Exception("Linux-only right now :-(")
platform = PLATFORM_LINUX

# Open the libraries
try:
    bcm = ctypes.CDLL("libbcm_host.so")
    platform = PLATFORM_PI
except OSError:
    libX11 = ctypes.CDLL("libX11.so")

opengles = ctypes.CDLL('libGLESv2.so')
openegl = ctypes.CDLL('libEGL.so')

eglint = ctypes.c_int


def eglints(L):
    return (eglint * len(L))(*L)


class EGL(object):

    def __init__(self):
        if platform == PLATFORM_PI:
            b = bcm.bcm_host_init()
            if b != 0:
                raise Exception("Could not initialize Pi GPU")

            width = eglint()
            height = eglint()
            s = bcm.graphics_get_display_size(0, ctypes.byref(width), ctypes.byref(height))
            if s < 0:
                raise Exception("Could not get display size")
            self.width = width
            self.height = height

        elif platform == PLATFORM_LINUX:
            print("Linux, opening display")
            display = libX11.XOpenDisplay(None)
            screen = libX11.XDefaultScreenOfDisplay(display)
            self.width = ctypes.c_int(libX11.XWidthOfScreen(screen))
            self.height = ctypes.c_int(libX11.XHeightOfScreen(screen))

        self.display = openegl.eglGetDisplay(egl_constants.EGL_DEFAULT_DISPLAY)
        if self.display == egl_constants.EGL_NO_DISPLAY:
            raise Exception("Could not open EGL display")

        if openegl.eglInitialize(self.display, 0, 0) == egl_constants.EGL_NO_DISPLAY:
            raise Exception("Could not initialise EGL")

        attribute_list = eglints((
            egl_constants.EGL_RED_SIZE, 8,
            egl_constants.EGL_GREEN_SIZE, 8,
            egl_constants.EGL_BLUE_SIZE, 8,
            egl_constants.EGL_ALPHA_SIZE, 8,
            egl_constants.EGL_DEPTH_SIZE, 24,
            egl_constants.EGL_SURFACE_TYPE, egl_constants.EGL_WINDOW_BIT,
            egl_constants.EGL_NONE
        ))

        numconfig = eglint()
        config = ctypes.c_void_p()
        r = openegl.eglChooseConfig(
            self.display,
            ctypes.byref(attribute_list),
            ctypes.byref(config), 1,
            ctypes.byref(numconfig)
        )
        if r == 0:
            raise Exception("Could not choose EGL config")

        r = openegl.eglBindAPI(egl_constants.EGL_OPENGL_ES_API)
        if r == 0:
            raise Exception("Could not bind config")

        context_attribs = eglints((egl_constants.EGL_CONTEXT_CLIENT_VERSION, 2, egl_constants.EGL_NONE))
        self.context = openegl.eglCreateContext(
            self.display, config,
            egl_constants.EGL_NO_CONTEXT,
            ctypes.byref(context_attribs)
        )
        if self.context == egl_constants.EGL_NO_CONTEXT:
            raise Exception("Could not create EGL context")


        if platform == PLATFORM_PI:
            dispman_display = bcm.vc_dispmanx_display_open(0)
            if dispman_display == 0:
                raise Exception("Could not open display")

            dispman_update = bcm.vc_dispmanx_update_start(0)
            if dispman_update == 0:
                raise Exception("Could not start updating display")

            dst_rect = eglints((0, 0, width.value, height.value))
            src_rect = eglints((0, 0, width.value << 16, height.value << 16))
            dispman_element = bcm.vc_dispmanx_element_add(
                dispman_update, dispman_display,
                0, ctypes.byref(dst_rect), 0,
                ctypes.byref(src_rect),
                egl_constants.DISPMANX_PROTECTION_NONE,
                0, 0, 0
            )
            bcm.vc_dispmanx_update_submit_sync(dispman_update)

            nativewindow = eglints((dispman_element, width, height))
            nw_p = ctypes.pointer(nativewindow)
            self.nw_p = nw_p
            self.surface = openegl.eglCreateWindowSurface(self.display, config, nw_p, 0)
            if self.surface == egl_constants.EGL_NO_SURFACE:
                raise Exception("Could not create surface")
        elif platform == PLATFORM_LINUX:
            print("Linux, creating surface")
            root = libX11.XRootWindowOfScreen(screen)
            print("width={}, height={}".format(self.width, self.height))
            window = libX11.XCreateSimpleWindow(display, root, 0, 0, self.width.value, self.height.value, 1, 0, 0)
            s = ctypes.create_string_buffer(b'WM_DELETE_WINDOW')
            WM_DELETE_WINDOW = ctypes.c_ulong(libX11.XInternAtom(display, s, 0))
            libX11.XSetWMProtocols(display, window, ctypes.byref(WM_DELETE_WINDOW), 1)
            KeyPressMask =   (1<<0)
            KeyReleaseMask =   (1<<1)
            libX11.XSelectInput(display, window, KeyPressMask | KeyReleaseMask)
            libX11.XMapWindow(display, window)
            self.surface = openegl.eglCreateWindowSurface(display, config, window, 0)

        r = openegl.eglMakeCurrent(self.display, self.surface, self.surface, self.context)
        if r == 0:
            raise Exception("Could not make our surface current")


    def get_context(self):
        return GL()


    def swap_buffers(self):
        openegl.eglSwapBuffers(self.display, self.surface)
