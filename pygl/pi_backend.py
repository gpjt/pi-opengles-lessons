import ctypes

from pygl import egl_constants
from pygl.egl_libs import eglints, openegl

bcm_name = ctypes.util.find_library('bcm_host')
bcm = ctypes.CDLL(bcm_name)


class Backend(object):

    def initialize(self):
        if bcm.bcm_host_init() != 0:
            raise Exception("Could not initialize Pi GPU")


    def get_display_size(self):
        width = ctypes.c_int()
        height = ctypes.c_int()
        error_code = bcm.graphics_get_display_size(
            0, ctypes.byref(width), ctypes.byref(height)
        )
        if error_code < 0:
            raise Exception("Could not get display size")
        return width, height


    def create_surface(self, display, config, width, height):
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

        native_window = eglints((dispman_element, width, height))
        # We need to keep a reference to this because if it gets
        # GCed we stop displaying stuff.
        self.native_window_pointer = ctypes.pointer(native_window)
        surface = openegl.eglCreateWindowSurface(
            display, config, self.native_window_pointer, 0
        )
        if surface == egl_constants.EGL_NO_SURFACE:
            raise Exception("Could not create surface")
        return surface
