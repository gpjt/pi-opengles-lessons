import ctypes
import Xlib.display
import Xlib.Xatom
import Xlib.Xutil

from pygl import egl_constants
from pygl.egl_libs import openegl

bcm_name = ctypes.util.find_library('bcm_host')
bcm = ctypes.CDLL(bcm_name)


class Backend(object):

    def initialize(self):
        pass


    def get_display_size(self):
        self.x_display = Xlib.display.Display()
        self.screen = self.x_display.screen()
        self.width = ctypes.c_int(self.screen.width_in_pixels)
        self.height = ctypes.c_int(self.screen.height_in_pixels)
        return self.width, self.height


    def create_surface(self, display, config, width, height):
        window = self.screen.root.create_window(
            x=0, y=0, width=self.width.value, height=self.height.value,
            border_width=0, depth=self.screen.root_depth
        )
        atom_net_wm_state = self.x_display.intern_atom(
            '_NET_WM_STATE', True
        )
        atom_net_wm_state_fullscreen = self.x_display.intern_atom(
            '_NET_WM_STATE_FULLSCREEN', True
        )

        window.change_property(
            atom_net_wm_state,
            Xlib.Xatom.ATOM,
            32,
            [atom_net_wm_state_fullscreen],
        )

        window.set_wm_normal_hints(
            flags=(
                Xlib.Xutil.PPosition |
                Xlib.Xutil.PSize |
                Xlib.Xutil.PMinSize
            ),
            min_width=self.width.value,
            min_height=self.height.value,
        )

        window.map()

        surface = openegl.eglCreateWindowSurface(
            display, config, window.id, 0
        )
        if surface == egl_constants.EGL_NO_SURFACE:
            raise Exception(
                "Could not create surface: {}".format(openegl.eglGetError())
            )
        return surface


    def pump_events(self):
        n = self.x_display.pending_events()
        for i in range(n):
            self.x_display.next_event()
