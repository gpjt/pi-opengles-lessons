#
# Copyright (c) 2017 Giles Thomas
# Copyright (c) 2012 Peter de Rivaz
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted.
#

import ctypes
import numpy as np

from pygl.egl import EGL
from pygl.matrix_utils import perspective, translate


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


def create_shader(gl, code, shader_type):
    shader = gl.createShader(shader_type);
    gl.shaderSource(shader, 1, ctypes.byref(ctypes.c_char_p(code)), 0)
    gl.check_for_error()
    gl.compileShader(shader);
    gl.check_for_error()

    compile_status = ctypes.c_int()
    gl.getShaderiv(shader, gl.COMPILE_STATUS, ctypes.byref(compile_status))
    gl.check_for_error()
    if compile_status == 0:
        N = 1024
        log = (ctypes.c_char*N)()
        loglen = ctypes.c_int()
        gl.getShaderInfoLog(shader, N, ctypes.byref(loglen), ctypes.byref(log))
        gl.check_for_error()
        raise Exception("Shader compile error: {}".format(log.value))

    return shader
    

class Program:
    def __init__(self, gl):
        self.gl_program = gl.createProgram()


def init_shaders(gl):
    fragment_shader = create_shader(gl, FRAGMENT_SHADER, gl.FRAGMENT_SHADER)
    vertex_shader = create_shader(gl, VERTEX_SHADER, gl.VERTEX_SHADER)

    shader_program = Program(gl)
    gl.attachShader(shader_program.gl_program, vertex_shader)
    gl.check_for_error()
    gl.attachShader(shader_program.gl_program, fragment_shader)
    gl.check_for_error()
    gl.linkProgram(shader_program.gl_program)
    gl.check_for_error()

    link_status = ctypes.c_int()
    gl.getProgramiv(
        shader_program.gl_program, gl.LINK_STATUS, ctypes.byref(link_status)
    )
    gl.check_for_error()
    if link_status == 0:
        N = 1024
        log = (ctypes.c_char*N)()
        loglen = ctypes.c_int()
        gl.getProgramInfoLog(
            shader_program.gl_program, N, ctypes.byref(loglen),ctypes.byref(log)
        )
        gl.check_for_error()
        raise Exception("Program link error: {}".format(log.value))

    gl.useProgram(shader_program.gl_program)
    gl.check_for_error()
    shader_program.vertex_position_attribute = gl.getAttribLocation(
        shader_program.gl_program, b"aVertexPosition"
    )
    gl.check_for_error()
    gl.enableVertexAttribArray(shader_program.vertex_position_attribute)
    gl.check_for_error()

    shader_program.p_matrix_uniform = gl.getUniformLocation(
        shader_program.gl_program, b"uPMatrix"
    )
    gl.check_for_error()
    shader_program.mv_matrix_uniform = gl.getUniformLocation(
        shader_program.gl_program, b"uMVMatrix"
    )
    gl.check_for_error()

    return shader_program



class Buffer:

    def __init__(self, gl):
        self.gl_buffer = ctypes.c_int()
        gl.genBuffers(1, ctypes.byref(self.gl_buffer))
        gl.check_for_error()
        

def ctypes_floats(L):
    return (ctypes.c_float*len(L))(*L)


def init_buffers(gl): 
    triangle_vertex_position_buffer = Buffer(gl)
    gl.bindBuffer(gl.ARRAY_BUFFER, triangle_vertex_position_buffer.gl_buffer)
    gl.check_for_error()
    vertices = ctypes_floats((
        0.0,  1.0,  0.0,
       -1.0, -1.0,  0.0,
        1.0, -1.0,  0.0,
    ))
    gl.bufferData(
        gl.ARRAY_BUFFER, 
        ctypes.sizeof(vertices), ctypes.byref(vertices),
        gl.STATIC_DRAW
    )
    gl.check_for_error()
    triangle_vertex_position_buffer.item_size = 3
    triangle_vertex_position_buffer.num_items = 3

    square_vertex_position_buffer = Buffer(gl)
    gl.bindBuffer(gl.ARRAY_BUFFER, square_vertex_position_buffer.gl_buffer)
    gl.check_for_error()
    vertices = ctypes_floats((
        1.0,  1.0,  0.0,
       -1.0,  1.0,  0.0,
        1.0, -1.0,  0.0,
       -1.0, -1.0,  0.0
    ))
    gl.bufferData(
        gl.ARRAY_BUFFER, 
        ctypes.sizeof(vertices), ctypes.byref(vertices),
        gl.STATIC_DRAW
    )
    gl.check_for_error()
    square_vertex_position_buffer.item_size = 3
    square_vertex_position_buffer.num_items = 4

    return triangle_vertex_position_buffer, square_vertex_position_buffer


def draw_scene(
    egl, gl, shader_program, triangle_vertex_position_buffer, square_vertex_position_buffer
):
    gl.bindFramebuffer(gl.FRAMEBUFFER, 0)
    gl.check_for_error()
    gl.viewport(0, 0, egl.width, egl.height)
    gl.check_for_error()

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
    gl.check_for_error()

    p_matrix = perspective(45, float(egl.width.value) / float(egl.height.value), 0.1, 100.0)
    mv_matrix = np.identity(4)

    mv_matrix = mv_matrix * translate([-1.5, 0.0, -7.0])
    gl.bindBuffer(gl.ARRAY_BUFFER, triangle_vertex_position_buffer.gl_buffer)
    gl.check_for_error()
    gl.vertexAttribPointer(
        shader_program.vertex_position_attribute, 
        triangle_vertex_position_buffer.item_size, gl.FLOAT, False, 0, 0
    )
    gl.check_for_error()
    gl.uniformMatrix4fv(
        shader_program.p_matrix_uniform, 1, False, 
        np.ascontiguousarray(p_matrix.T, dtype=np.float32).ctypes.data
    )
    gl.check_for_error()
    gl.uniformMatrix4fv(
        shader_program.mv_matrix_uniform, 1, False,
        np.ascontiguousarray(mv_matrix.T, dtype=np.float32).ctypes.data
    )
    gl.check_for_error()
    gl.drawArrays(gl.TRIANGLES, 0, triangle_vertex_position_buffer.num_items)
    gl.check_for_error()

    mv_matrix = mv_matrix * translate([3.0, 0.0, 0.0])
    gl.bindBuffer(gl.ARRAY_BUFFER, square_vertex_position_buffer.gl_buffer)
    gl.check_for_error()
    gl.vertexAttribPointer(
        shader_program.vertex_position_attribute,
        square_vertex_position_buffer.item_size, gl.FLOAT, False, 0, 0
    )
    gl.check_for_error()
    gl.uniformMatrix4fv(
        shader_program.p_matrix_uniform, 1, False, 
        np.ascontiguousarray(p_matrix.T, dtype=np.float32).ctypes.data
    )
    gl.check_for_error()
    gl.uniformMatrix4fv(
        shader_program.mv_matrix_uniform, 1, False,
        np.ascontiguousarray(mv_matrix.T, dtype=np.float32).ctypes.data
    )
    gl.check_for_error()
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, square_vertex_position_buffer.num_items)
    gl.check_for_error()

    gl.bindBuffer(gl.ARRAY_BUFFER, 0);
    gl.check_for_error()

    gl.flush()
    gl.check_for_error()
    gl.finish()
    gl.check_for_error()

    egl.swap_buffers()


def main():
    egl = EGL()
    gl = egl.get_context()
    shader_program = init_shaders(gl)
    triangle_vertex_position_buffer, square_vertex_position_buffer = init_buffers(gl)
    gl.ClearColor(
        ctypes.c_float(0.0), ctypes.c_float(0.0), ctypes.c_float(0.0), ctypes.c_float(1.0)
    )
    gl.Enable(gl.DEPTH_TEST)
    while True:
        draw_scene(
            egl, gl, shader_program,
            triangle_vertex_position_buffer,
            square_vertex_position_buffer
        )

    
if __name__ == "__main__":
    main()
