import ctypes

from pygl import egl_constants
from pygl.egl_libs import Backend, eglints, openegl
from pygl.gl import GL

# Define some extra constants that the automatic extraction misses
egl_constants.EGL_FALSE = 0
egl_constants.EGL_DEFAULT_DISPLAY = 0
egl_constants.EGL_NO_CONTEXT = 0
egl_constants.EGL_NO_DISPLAY = 0
egl_constants.EGL_NO_SURFACE = 0
egl_constants.DISPMANX_PROTECTION_NONE = 0


eglint = ctypes.c_int


class EGL(object):

    def __init__(self):
        self.backend = Backend()
        self.backend.initialize()

        self.display = openegl.eglGetDisplay(egl_constants.EGL_DEFAULT_DISPLAY)
        if self.display == egl_constants.EGL_NO_DISPLAY:
            raise Exception(
                "Could not open EGL display: {}".format(openegl.eglGetError())
            )

        init_error = openegl.eglInitialize(self.display, None, None)
        if init_error == egl_constants.EGL_FALSE:
            raise Exception(
                "Could not initialise EGL: {}".format(openegl.eglGetError())
            )

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
        config_ok = openegl.eglChooseConfig(
            self.display,
            ctypes.byref(attribute_list),
            ctypes.byref(config), 1,
            ctypes.byref(numconfig)
        )
        if config_ok == 0:
            raise Exception(
                "Could not choose EGL config: {}".format(openegl.eglGetError())
            )

        if openegl.eglBindAPI(egl_constants.EGL_OPENGL_ES_API) == 0:
            raise Exception(
                "Could not bind config: {}".format(openegl.eglGetError())
            )

        context_attribs = eglints((
            egl_constants.EGL_CONTEXT_CLIENT_VERSION,
            2, egl_constants.EGL_NONE
        ))
        self.context = openegl.eglCreateContext(
            self.display, config,
            egl_constants.EGL_NO_CONTEXT,
            ctypes.byref(context_attribs)
        )
        if self.context == egl_constants.EGL_NO_CONTEXT:
            raise Exception(
                "Could not create EGL context: {}".format(
                    openegl.eglGetError()
                )
            )

        self.width, self.height = self.backend.get_display_size()
        self.surface = self.backend.create_surface(
            self.display, config, self.width, self.height
        )

        make_current_error = openegl.eglMakeCurrent(
            self.display, self.surface, self.surface, self.context
        )
        if make_current_error == 0:
            raise Exception(
                "Could not make our surface current: {}".format(
                    openegl.eglGetError()
                )
            )


    def get_context(self):
        return GL()


    def swap_buffers(self):
        openegl.eglSwapBuffers(self.display, self.surface)


    def pump_events(self):
        self.backend.pump_events()
