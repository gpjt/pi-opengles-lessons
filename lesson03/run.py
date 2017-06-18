import ctypes
import numpy as np
import time

from pygl.egl import EGL
from matrix_utils import perspective, rotate, translate


FRAGMENT_SHADER = """
    precision mediump float;

    varying vec4 vColor;

    void main(void) {
        gl_FragColor = vColor;
    }
"""

VERTEX_SHADER = """
    attribute vec3 aVertexPosition;
    attribute vec4 aVertexColor;

    uniform mat4 uMVMatrix;
    uniform mat4 uPMatrix;

    varying vec4 vColor;

    void main(void) {
        gl_Position = uPMatrix * uMVMatrix * vec4(aVertexPosition, 1.0);
        vColor = aVertexColor;
    }
"""


def create_shader(gl, code, shader_type):
    shader = gl.createShader(shader_type);
    gl.shaderSource(shader, code)
    gl.compileShader(shader);

    if not gl.getShaderParameter(shader, gl.COMPILE_STATUS):
        log = gl.getShaderInfoLog(shader)
        raise Exception("Shader compile error: {}".format(log))

    return shader
    

class Program:
    def __init__(self, gl):
        self.gl_program = gl.createProgram()


def init_shaders(gl):
    fragment_shader = create_shader(gl, FRAGMENT_SHADER, gl.FRAGMENT_SHADER)
    vertex_shader = create_shader(gl, VERTEX_SHADER, gl.VERTEX_SHADER)

    shader_program = Program(gl)
    gl.attachShader(shader_program.gl_program, vertex_shader)
    gl.attachShader(shader_program.gl_program, fragment_shader)
    gl.linkProgram(shader_program.gl_program)

    if gl.getProgramParameter(shader_program.gl_program, gl.LINK_STATUS) == 0:
        log = gl.getProgramInfoLog(shader_program.gl_program)
        raise Exception("Program link error: {}".format(log))

    gl.useProgram(shader_program.gl_program)
    shader_program.vertex_position_attribute = gl.getAttribLocation(
        shader_program.gl_program, "aVertexPosition"
    )
    gl.enableVertexAttribArray(shader_program.vertex_position_attribute)

    shader_program.vertex_color_attribute = gl.getAttribLocation(
        shader_program.gl_program, "aVertexColor"
    )
    gl.enableVertexAttribArray(shader_program.vertex_color_attribute)

    shader_program.p_matrix_uniform = gl.getUniformLocation(
        shader_program.gl_program, "uPMatrix"
    )
    shader_program.mv_matrix_uniform = gl.getUniformLocation(
        shader_program.gl_program, "uMVMatrix"
    )

    return shader_program



class Buffer:

    def __init__(self, gl):
        self.gl_buffer = gl.createBuffer()



def init_buffers(gl): 
    triangle_vertex_position_buffer = Buffer(gl)
    gl.bindBuffer(gl.ARRAY_BUFFER, triangle_vertex_position_buffer.gl_buffer)
    vertices = [
        0.0,  1.0,  0.0,
       -1.0, -1.0,  0.0,
        1.0, -1.0,  0.0,
    ]
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW)
    triangle_vertex_position_buffer.item_size = 3
    triangle_vertex_position_buffer.num_items = 3

    triangle_vertex_color_buffer = Buffer(gl)
    gl.bindBuffer(gl.ARRAY_BUFFER, triangle_vertex_color_buffer.gl_buffer)
    colors = [
        1.0, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 1.0,
    ]
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW)
    triangle_vertex_color_buffer.item_size = 4
    triangle_vertex_color_buffer.num_items = 3

    square_vertex_position_buffer = Buffer(gl)
    gl.bindBuffer(gl.ARRAY_BUFFER, square_vertex_position_buffer.gl_buffer)
    vertices = [
        1.0,  1.0,  0.0,
       -1.0,  1.0,  0.0,
        1.0, -1.0,  0.0,
       -1.0, -1.0,  0.0
    ]
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW)
    square_vertex_position_buffer.item_size = 3
    square_vertex_position_buffer.num_items = 4

    square_vertex_color_buffer = Buffer(gl)
    gl.bindBuffer(gl.ARRAY_BUFFER, square_vertex_color_buffer.gl_buffer)
    colors = []
    for _ in range(4):
        colors += [0.5, 0.5, 1.0, 1.0]
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW)
    square_vertex_color_buffer.item_size = 4
    square_vertex_color_buffer.num_items = 4

    return (
        triangle_vertex_position_buffer,
        triangle_vertex_color_buffer,
        square_vertex_position_buffer,
        square_vertex_color_buffer,
    )


def draw_scene(
    egl, gl, shader_program,
    triangle_vertex_position_buffer,
    triangle_vertex_color_buffer,
    r_tri,
    square_vertex_position_buffer,
    square_vertex_color_buffer,
    r_square,
):
    gl.bindFramebuffer(gl.FRAMEBUFFER, 0)
    gl.viewport(0, 0, egl.width, egl.height)

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

    p_matrix = perspective(45, float(egl.width.value) / float(egl.height.value), 0.1, 100.0)
    mv_matrix = np.identity(4)

    mv_matrix = mv_matrix * translate([-1.5, 0.0, -7.0])
    gl.bindBuffer(gl.ARRAY_BUFFER, triangle_vertex_position_buffer.gl_buffer)
    gl.vertexAttribPointer(
        shader_program.vertex_position_attribute, 
        triangle_vertex_position_buffer.item_size, gl.FLOAT, False, 0, 0
    )
    gl.bindBuffer(gl.ARRAY_BUFFER, triangle_vertex_color_buffer.gl_buffer)
    gl.vertexAttribPointer(
        shader_program.vertex_color_attribute, 
        triangle_vertex_color_buffer.item_size, gl.FLOAT, False, 0, 0
    )
    gl.uniformMatrix4fv(shader_program.p_matrix_uniform, False, p_matrix.T)
    triangle_mv_matrix = mv_matrix * rotate(r_tri, np.array([0, 1, 0]))
    gl.uniformMatrix4fv(shader_program.mv_matrix_uniform, False, triangle_mv_matrix.T)
    gl.drawArrays(gl.TRIANGLES, 0, triangle_vertex_position_buffer.num_items)

    mv_matrix = mv_matrix * translate([3.0, 0.0, 0.0])
    gl.bindBuffer(gl.ARRAY_BUFFER, square_vertex_position_buffer.gl_buffer)
    gl.vertexAttribPointer(
        shader_program.vertex_position_attribute,
        square_vertex_position_buffer.item_size, gl.FLOAT, False, 0, 0
    )
    gl.bindBuffer(gl.ARRAY_BUFFER, square_vertex_color_buffer.gl_buffer)
    gl.vertexAttribPointer(
        shader_program.vertex_color_attribute,
        square_vertex_color_buffer.item_size, gl.FLOAT, False, 0, 0
    )
    gl.uniformMatrix4fv(shader_program.p_matrix_uniform, False, p_matrix.T)
    square_mv_matrix =  mv_matrix * rotate(r_square, np.array([1, 0, 0]))
    gl.uniformMatrix4fv(shader_program.mv_matrix_uniform, False, square_mv_matrix.T)
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, square_vertex_position_buffer.num_items)

    gl.bindBuffer(gl.ARRAY_BUFFER, 0);

    gl.flush()
    gl.finish()

    egl.swap_buffers()


def animate(r_tri, r_square):
    return (r_tri + 1) % 360, (r_square + 1) % 360


def main():
    egl = EGL()
    gl = egl.get_context()
    shader_program = init_shaders(gl)
    (triangle_vertex_position_buffer, triangle_vertex_color_buffer, square_vertex_position_buffer, square_vertex_color_buffer) = init_buffers(gl)
    gl.ClearColor(
        ctypes.c_float(0.0), ctypes.c_float(0.0), ctypes.c_float(0.0), ctypes.c_float(1.0)
    )
    gl.Enable(gl.DEPTH_TEST)
    r_tri = 0
    r_square = 0
    while True:
        start = time.time()
        draw_scene(
            egl, gl, shader_program,
            triangle_vertex_position_buffer,
            triangle_vertex_color_buffer,
            r_tri,
            square_vertex_position_buffer,
            square_vertex_color_buffer,
            r_square,
        )
        r_tri, r_square = animate(r_tri, r_square)
        remainder = (1/60.0) - (time.time() - start)
        if remainder > 0:
            time.sleep(remainder)


    
if __name__ == "__main__":
    main()
