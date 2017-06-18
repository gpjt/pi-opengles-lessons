#
# Copyright (c) 2017 Giles Thomas
# Copyright (c) 2012 Peter de Rivaz
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted.
#

import ctypes
import time
import math
import numpy as np

# Pick up our constants extracted from the header files with prepare_constants.py
from egl import *
from gl2 import *
from gl2ext import *

# Define some extra constants that the automatic extraction misses
EGL_DEFAULT_DISPLAY = 0
EGL_NO_CONTEXT = 0
EGL_NO_DISPLAY = 0
EGL_NO_SURFACE = 0
DISPMANX_PROTECTION_NONE = 0

# Open the libraries
bcm = ctypes.CDLL('libbcm_host.so')
opengles = ctypes.CDLL('libGLESv2.so')
openegl = ctypes.CDLL('libEGL.so')

eglint = ctypes.c_int

eglshort = ctypes.c_short

def eglints(L):
    """Converts a tuple to an array of eglints (would a pointer return be better?)"""
    return (eglint*len(L))(*L)

eglfloat = ctypes.c_float


def eglfloats(L):
    return (eglfloat*len(L))(*L)

                
class EGL(object):

    def __init__(self):
        """Opens up the OpenGL library and prepares a window for display"""
        b = bcm.bcm_host_init()
        if b != 0:
            raise Exception("Could not initialize Pi GPU")

        self.display = openegl.eglGetDisplay(EGL_DEFAULT_DISPLAY)
        if self.display == 0:
            raise Exception("Could not open EGL display")

        if openegl.eglInitialize(self.display, 0, 0) == 0:
            raise Exception("Could not initialise EGL")

        attribute_list = eglints((
            EGL_RED_SIZE, 8,
            EGL_GREEN_SIZE, 8,
            EGL_BLUE_SIZE, 8,
            EGL_ALPHA_SIZE, 8,
            EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
            EGL_NONE
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

        r = openegl.eglBindAPI(EGL_OPENGL_ES_API)
        if r == 0:
            raise Exception("Could not bind config")

        context_attribs = eglints((EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE))
        self.context = openegl.eglCreateContext(
            self.display, config,
            EGL_NO_CONTEXT,
            ctypes.byref(context_attribs)
        )
        if self.context == EGL_NO_CONTEXT:
            raise Exception("Could not create EGL context")

        width = eglint()
        height = eglint()
        s = bcm.graphics_get_display_size(0, ctypes.byref(width),ctypes.byref(height))
        self.width = width
        self.height = height
        if s < 0:
            raise Exception("Could not get display size")

        dispman_display = bcm.vc_dispmanx_display_open(0)
        if dispman_display == 0:
            raise Exception("Could not open display")

        dispman_update = bcm.vc_dispmanx_update_start(0)
        if dispman_update == 0:
            raise Exception("Could not start updating display")

        dst_rect = eglints((0, 0, width.value, height.value))
        src_rect = eglints((0, 0, width.value<<16, height.value<<16))
        dispman_element = bcm.vc_dispmanx_element_add(
            dispman_update, dispman_display,
            0, ctypes.byref(dst_rect), 0,
            ctypes.byref(src_rect),
            DISPMANX_PROTECTION_NONE,
            0, 0, 0
        )
        bcm.vc_dispmanx_update_submit_sync(dispman_update)

        nativewindow = eglints((dispman_element, width, height))
        nw_p = ctypes.pointer(nativewindow)
        self.nw_p = nw_p
        self.surface = openegl.eglCreateWindowSurface(self.display, config, nw_p, 0)
        if self.surface == EGL_NO_SURFACE:
            raise Exception("Could not create surface")

        r = openegl.eglMakeCurrent(self.display, self.surface, self.surface, self.context) 
        if r == 0:
            raise Exception("Could not make our surface current")


FRAGMENT_SHADER = b"""
    precision mediump float;
    void main(void) {
        gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
    }
"""

VERTEX_SHADER = b"""
    attribute vec3 aVertexPosition;
    uniform mat4 uMVMatrix;
    uniform mat4 uPMatrix;
    void main(void) {
        gl_Position = uPMatrix * uMVMatrix * vec4(aVertexPosition, 1.0);
    }
"""


def create_shader(code, shader_type):
    shader = opengles.glCreateShader(shader_type);
    opengles.glShaderSource(shader, 1, ctypes.byref(ctypes.c_char_p(code)), 0)
    check_for_error()
    opengles.glCompileShader(shader);
    check_for_error()

    compile_status = ctypes.c_int()
    opengles.glGetShaderiv(shader, GL_COMPILE_STATUS, ctypes.byref(compile_status))
    check_for_error()
    if compile_status == 0:
        N = 1024
        log = (ctypes.c_char*N)()
        loglen = ctypes.c_int()
        opengles.glGetShaderInfoLog(shader, N, ctypes.byref(loglen), ctypes.byref(log))
        check_for_error()
        raise Exception("Shader compile error: {}".format(log.value))

    return shader
    

class Program:
    def __init__(self):
        self.gl_program = opengles.glCreateProgram()


def init_shaders():
    fragment_shader = create_shader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    vertex_shader = create_shader(VERTEX_SHADER, GL_VERTEX_SHADER)

    shader_program = Program()
    opengles.glAttachShader(shader_program.gl_program, vertex_shader)
    check_for_error()
    opengles.glAttachShader(shader_program.gl_program, fragment_shader)
    check_for_error()
    opengles.glLinkProgram(shader_program.gl_program)
    check_for_error()

    link_status = ctypes.c_int()
    opengles.glGetProgramiv(
        shader_program.gl_program, GL_LINK_STATUS, ctypes.byref(link_status)
    )
    check_for_error()
    if link_status == 0:
        N = 1024
        log = (ctypes.c_char*N)()
        loglen = ctypes.c_int()
        opengles.glGetProgramInfoLog(
            shader_program.gl_program, N, ctypes.byref(loglen),ctypes.byref(log)
        )
        check_for_error()
        raise Exception("Program link error: {}".format(log.value))

    opengles.glUseProgram(shader_program.gl_program)
    check_for_error()
    shader_program.vertex_position_attribute = opengles.glGetAttribLocation(
        shader_program.gl_program, b"aVertexPosition"
    )
    check_for_error()
    opengles.glEnableVertexAttribArray(shader_program.vertex_position_attribute)
    check_for_error()

    shader_program.p_matrix_uniform = opengles.glGetUniformLocation(
        shader_program.gl_program, b"uPMatrix"
    )
    check_for_error()
    shader_program.mv_matrix_uniform = opengles.glGetUniformLocation(
        shader_program.gl_program, b"uMVMatrix"
    )
    check_for_error()

    return shader_program



class Buffer:

    def __init__(self):
        self.gl_buffer = ctypes.c_int()
        opengles.glGenBuffers(1, ctypes.byref(self.gl_buffer))
        check_for_error()
        


def init_buffers(): 
    triangle_vertex_position_buffer = Buffer()
    opengles.glBindBuffer(GL_ARRAY_BUFFER, triangle_vertex_position_buffer.gl_buffer)
    check_for_error()
    vertices = eglfloats((
        0.0,  1.0,  0.0,
       -1.0, -1.0,  0.0,
        1.0, -1.0,  0.0,
    ))
    opengles.glBufferData(
        GL_ARRAY_BUFFER, 
        ctypes.sizeof(vertices), ctypes.byref(vertices),
        GL_STATIC_DRAW
    )
    check_for_error()
    triangle_vertex_position_buffer.item_size = 3
    triangle_vertex_position_buffer.num_items = 3

    square_vertex_position_buffer = Buffer()
    opengles.glBindBuffer(GL_ARRAY_BUFFER, square_vertex_position_buffer.gl_buffer)
    check_for_error()
    vertices = eglfloats((
        1.0,  1.0,  0.0,
       -1.0,  1.0,  0.0,
        1.0, -1.0,  0.0,
       -1.0, -1.0,  0.0
    ))
    opengles.glBufferData(
        GL_ARRAY_BUFFER, 
        ctypes.sizeof(vertices), ctypes.byref(vertices),
        GL_STATIC_DRAW
    )
    check_for_error()
    square_vertex_position_buffer.item_size = 3
    square_vertex_position_buffer.num_items = 4

    return triangle_vertex_position_buffer, square_vertex_position_buffer


def check_for_error():
    e = opengles.glGetError()
    if e:
        raise Exception("GL error {}".format(hex(e)))

def perspective(fovy, aspect, n, f):
    s = 1.0 / math.tan(math.radians(fovy) / 2.0)
    sx, sy = s / aspect, s
    zz = (f+n) / (n-f)
    zw = 2 * f * n / (n-f)
    return np.matrix([
        [sx,  0,  0, 0],
        [0,  sy,  0, 0],
        [0,   0, zz,zw],
        [0,   0, -1, 0]
    ])


def translate(xyz):
    x, y, z = xyz
    return np.matrix([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])




def draw_scene(
    egl, shader_program, triangle_vertex_position_buffer, square_vertex_position_buffer
):
    opengles.glBindFramebuffer(GL_FRAMEBUFFER, 0)
    check_for_error()
    opengles.glViewport(0, 0, egl.width, egl.height)
    check_for_error()

    opengles.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    check_for_error()

    p_matrix = perspective(45, float(egl.width.value) / float(egl.height.value), 0.1, 100.0)
    mv_matrix = np.identity(4)

    mv_matrix = mv_matrix * translate([-1.5, 0.0, -7.0])
    opengles.glBindBuffer(GL_ARRAY_BUFFER, triangle_vertex_position_buffer.gl_buffer)
    check_for_error()
    opengles.glVertexAttribPointer(
        shader_program.vertex_position_attribute, 
        triangle_vertex_position_buffer.item_size, GL_FLOAT, False, 0, 0
    )
    check_for_error()
    opengles.glUniformMatrix4fv(
        shader_program.p_matrix_uniform, 1, False, 
        np.ascontiguousarray(p_matrix.T, dtype=np.float32).ctypes.data
    )
    check_for_error()
    opengles.glUniformMatrix4fv(
        shader_program.mv_matrix_uniform, 1, False,
        np.ascontiguousarray(mv_matrix.T, dtype=np.float32).ctypes.data
    )
    check_for_error()
    opengles.glDrawArrays(GL_TRIANGLES, 0, triangle_vertex_position_buffer.num_items)
    check_for_error()

    mv_matrix = mv_matrix * translate([3.0, 0.0, 0.0])
    opengles.glBindBuffer(GL_ARRAY_BUFFER, square_vertex_position_buffer.gl_buffer)
    check_for_error()
    opengles.glVertexAttribPointer(
        shader_program.vertex_position_attribute,
        square_vertex_position_buffer.item_size, GL_FLOAT, False, 0, 0
    )
    check_for_error()
    opengles.glUniformMatrix4fv(
        shader_program.p_matrix_uniform, 1, False, 
        np.ascontiguousarray(p_matrix.T, dtype=np.float32).ctypes.data
    )
    check_for_error()
    opengles.glUniformMatrix4fv(
        shader_program.mv_matrix_uniform, 1, False,
        np.ascontiguousarray(mv_matrix.T, dtype=np.float32).ctypes.data
    )
    check_for_error()
    opengles.glDrawArrays(GL_TRIANGLE_STRIP, 0, square_vertex_position_buffer.num_items)
    check_for_error()

    opengles.glBindBuffer(GL_ARRAY_BUFFER, 0);
    check_for_error()

    opengles.glFlush()
    check_for_error()
    opengles.glFinish()
    check_for_error()

    openegl.eglSwapBuffers(egl.display, egl.surface);
    check_for_error()


def main():
    egl = EGL()
    shader_program = init_shaders()
    triangle_vertex_position_buffer, square_vertex_position_buffer = init_buffers()
    opengles.glClearColor(eglfloat(0.0), eglfloat(0.0), eglfloat(0.0), eglfloat(1.0))
    opengles.glEnable(GL_DEPTH_TEST)
    while True:
        draw_scene(
            egl, shader_program,
            triangle_vertex_position_buffer,
            square_vertex_position_buffer
        )

    
if __name__ == "__main__":
    main()
