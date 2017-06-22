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


def init_texture():
    pass


class Shape:

    def __init__(self, gl, positions, colors, indices=None, texture_coords=None):
        self.position_buffer = gl.createBuffer()
        gl.bindBuffer(gl.ARRAY_BUFFER, self.position_buffer)
        gl.bufferData(gl.ARRAY_BUFFER, sum(positions, []), gl.STATIC_DRAW)
        self.position_item_size = len(positions[0])

        self.color_buffer = gl.createBuffer()
        gl.bindBuffer(gl.ARRAY_BUFFER, self.color_buffer)
        gl.bufferData(gl.ARRAY_BUFFER, sum(colors, []), gl.STATIC_DRAW)
        self.color_item_size = len(colors[0])

        self.num_vertices = len(positions)

        if indices is not None:
            self.index_buffer = gl.createBuffer()
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, self.index_buffer)
            gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW)
            self.num_indices = len(indices)

        if texture_coords is None:
            self.texture_coords_buffer = gl.createBuffer()
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, self.self.texture_coords_buffer)
            gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, texture_coords, gl.STATIC_DRAW)
            self.texture_coord_item_size = len(texture_coords[0])

        self.angleX = 0
        self.angleY = 0
        self.angleZ = 0



def init_buffers(gl): 
    cube_face_colors = [
        [1.0, 0.0, 0.0, 1.0],     # Front face
        [1.0, 1.0, 0.0, 1.0],     # Back face
        [0.0, 1.0, 0.0, 1.0],     # Top face
        [1.0, 0.5, 0.5, 1.0],     # Bottom face
        [1.0, 0.0, 1.0, 1.0],     # Right face
        [0.0, 0.0, 1.0, 1.0],     # Left face
    ]
    cube_colors = []
    for color in cube_face_colors:
        for _ in range(4):
            cube_colors.append(color)
    cube_shape = Shape(
        gl,
        [
            # Front face
            [-1.0, -1.0,  1.0],
            [ 1.0, -1.0,  1.0],
            [ 1.0,  1.0,  1.0],
            [-1.0,  1.0,  1.0],
            # Back face
            [-1.0, -1.0, -1.0],
            [-1.0,  1.0, -1.0],
            [ 1.0,  1.0, -1.0],
            [ 1.0, -1.0, -1.0],
            # Top face
            [-1.0,  1.0, -1.0],
            [-1.0,  1.0,  1.0],
            [ 1.0,  1.0,  1.0],
            [ 1.0,  1.0, -1.0],
            # Bottom face
            [-1.0, -1.0, -1.0],
            [ 1.0, -1.0, -1.0],
            [ 1.0, -1.0,  1.0],
            [-1.0, -1.0,  1.0],
            # Right face
            [ 1.0, -1.0, -1.0],
            [ 1.0,  1.0, -1.0],
            [ 1.0,  1.0,  1.0],
            [ 1.0, -1.0,  1.0],
            # Left face
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0,  1.0],
            [-1.0,  1.0,  1.0],
            [-1.0,  1.0, -1.0],
        ],
        cube_colors,
        indices=[
            0, 1, 2,      0, 2, 3,     # Front face
            4, 5, 6,      4, 6, 7,     # Back face
            8, 9, 10,     8, 10, 11,   # Top face
            12, 13, 14,   12, 14, 15,  # Bottom face
            16, 17, 18,   16, 18, 19,  # Right face
            20, 21, 22,   20, 22, 23   # Left face
        ],
        texture_coords = [
            # Front face
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,

            # Back face
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,
            0.0, 0.0,

            # Top face
            0.0, 1.0,
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,

            # Bottom face
            1.0, 1.0,
            0.0, 1.0,
            0.0, 0.0,
            1.0, 0.0,

            # Right face
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,
            0.0, 0.0,

            # Left face
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,
        ]
    )

    return cube_shape



def draw_scene(egl, gl, shader_program, cube_shape, texture):
    gl.bindFramebuffer(gl.FRAMEBUFFER, 0)
    gl.viewport(0, 0, egl.width, egl.height)

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

    p_matrix = perspective(45, float(egl.width.value) / float(egl.height.value), 0.1, 100.0)
    mv_matrix = np.identity(4)

    mv_matrix = mv_matrix * translate([0.0, 0.0, -5.0])
    mv_matrix = mv_matrix * rotate(cube_shape.angleX, np.array([1, 0, 0]))
    mv_matrix = mv_matrix * rotate(cube_shape.angleY, np.array([0, 1, 0]))
    mv_matrix = mv_matrix * rotate(cube_shape.angleZ, np.array([0, 0, 1]))
    gl.bindBuffer(gl.ARRAY_BUFFER, cube_shape.position_buffer)
    gl.vertexAttribPointer(
        shader_program.vertex_position_attribute,
        cube_shape.position_item_size, gl.FLOAT, False, 0, 0
    )
    gl.bindBuffer(gl.ARRAY_BUFFER, cube_shape.color_buffer)
    gl.vertexAttribPointer(
        shader_program.vertex_color_attribute,
        cube_shape.color_item_size, gl.FLOAT, False, 0, 0
    )
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, cube_shape.index_buffer)
    gl.uniformMatrix4fv(shader_program.p_matrix_uniform, False, p_matrix)
    gl.uniformMatrix4fv(shader_program.mv_matrix_uniform, False, mv_matrix)
    gl.drawElements(gl.TRIANGLES, cube_shape.num_indices, gl.UNSIGNED_SHORT, 0)

    egl.swap_buffers()


def animate(cube_shape):
    cube_shape.angleX = (cube_shape.angleX - 0.8) % 360
    cube_shape.angleY = (cube_shape.angleY - 0.8) % 360
    cube_shape.angleZ = (cube_shape.angleZ - 0.8) % 360


def main():
    egl = EGL()
    gl = egl.get_context()
    shader_program = init_shaders(gl)
    cube_shape = init_buffers(gl)
    texture = init_texture()
    gl.clearColor(0.0, 0.0, 0.0, 1.0)
    gl.enable(gl.DEPTH_TEST)
    while True:
        start = time.time()
        draw_scene(egl, gl, shader_program, cube_shape, texture)
        animate(cube_shape)
        remainder = (1/60.0) - (time.time() - start)
        if remainder > 0:
            time.sleep(remainder)


    
if __name__ == "__main__":
    main()
