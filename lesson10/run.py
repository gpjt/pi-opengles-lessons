import curses
import math
import numpy as np
import os
from PIL import Image
from random import random
import time

from pygl.egl import EGL
from matrix_utils import perspective, rotate, translate



FRAGMENT_SHADER = """
    precision mediump float;

    varying vec2 vTextureCoord;

    uniform sampler2D uSampler;

    void main(void) {
        gl_FragColor = texture2D(uSampler, vec2(vTextureCoord.s, vTextureCoord.t));
    }
"""

VERTEX_SHADER = """
    attribute vec3 aVertexPosition;
    attribute vec2 aTextureCoord;

    uniform mat4 uMVMatrix;
    uniform mat4 uPMatrix;

    varying vec2 vTextureCoord;

    void main(void) {
        gl_Position = uPMatrix * uMVMatrix * vec4(aVertexPosition, 1.0);
        vTextureCoord = aTextureCoord;
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

    shader_program.texture_coord_attribute = gl.getAttribLocation(
        shader_program.gl_program, "aTextureCoord"
    )
    gl.enableVertexAttribArray(shader_program.texture_coord_attribute)

    shader_program.p_matrix_uniform = gl.getUniformLocation(
        shader_program.gl_program, "uPMatrix"
    )
    shader_program.mv_matrix_uniform = gl.getUniformLocation(
        shader_program.gl_program, "uMVMatrix"
    )
    shader_program.sampler_uniform = gl.getUniformLocation(
        shader_program.gl_program, "uSampler"
    )

    return shader_program


def init_texture(gl):
    image = Image.open("mud.gif")
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image_data = np.array(image.convert("RGBA"))

    texture = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.RGBA, image.width, image.height,
        0, gl.RGBA, gl.UNSIGNED_BYTE, image_data
    )
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)

    gl.bindTexture(gl.TEXTURE_2D, None)

    return texture



class World(object):

    def __init__(self, shape):
        self.pitch = 0

        self.yaw = 0
        
        self.xPos = 0
        self.yPos = 0.4
        self.zPos = 0

        self.shape = shape


def move(world, amount):
    world.xPos -= math.sin(math.radians(world.yaw)) * amount
    world.zPos -= math.cos(math.radians(world.yaw)) * amount


def handle_keys(stdscr, world):
    c = stdscr.getch()
    if c == curses.KEY_PPAGE:
        world.pitch += 0.1
    elif c == curses.KEY_NPAGE:
        world.pitch -= 0.1
    elif c == curses.KEY_LEFT:
        world.yaw += 0.1
    elif c == curses.KEY_RIGHT:
        world.yaw -= 0.1
    elif c == curses.KEY_UP:
        move(world, 0.3)
    elif c == curses.KEY_DOWN:
        move(world, -0.3)
    elif c != -1:
        print("Unrecognised keycode {}".format(c))


class Shape:

    def __init__(self, gl, positions, texture_coords):
        self.position_buffer = gl.createBuffer()
        gl.bindBuffer(gl.ARRAY_BUFFER, self.position_buffer)
        gl.bufferData(gl.ARRAY_BUFFER, sum(positions, []), gl.STATIC_DRAW)
        self.position_item_size = len(positions[0])

        self.num_vertices = len(positions)

        self.texture_coord_buffer = gl.createBuffer()
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, self.texture_coord_buffer)
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, sum(texture_coords, []), gl.STATIC_DRAW)
        self.texture_coord_item_size = len(texture_coords[0])



def load_world(gl): 
    with open(os.path.join(os.path.dirname(__file__), "world.txt"), "r") as f:
        lines = [line for line in f]
    vertex_positions = []
    vertex_texture_coords = []
    for line in lines:
        vals = line.strip().split()
        if len(vals) == 5 and vals[0] != "//":
            vertex_positions.append([
                float(vals[0]),
                float(vals[1]),
                float(vals[2]),
            ])
            vertex_texture_coords.append([
                float(vals[3]),
                float(vals[4]),
            ])

    shape = Shape(gl, vertex_positions, vertex_texture_coords)
    return World(shape)


def draw_scene(egl, gl, shader_program, world, texture):
    gl.bindFramebuffer(gl.FRAMEBUFFER, 0)
    gl.viewport(0, 0, egl.width, egl.height)

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

    p_matrix = perspective(45, float(egl.width.value) / float(egl.height.value), 0.1, 100.0)

    mv_matrix = np.identity(4)
    mv_matrix = mv_matrix * rotate(-world.pitch, np.array([1, 0, 0]))
    mv_matrix = mv_matrix * rotate(-world.yaw, np.array([0, 1, 0]))
    mv_matrix = mv_matrix * translate([world.xPos, world.yPos, world.zPos])

    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.uniform1i(shader_program.sampler_uniform, 0)

    gl.bindBuffer(gl.ARRAY_BUFFER, world.shape.texture_coord_buffer)
    gl.vertexAttribPointer(
        shader_program.texture_coord_attribute,
        world.shape.texture_coord_item_size,
        gl.FLOAT, False, 0, 0
    )

    gl.bindBuffer(gl.ARRAY_BUFFER, world.shape.position_buffer)
    gl.vertexAttribPointer(
        shader_program.vertex_position_attribute,
        world.shape.position_item_size,
        gl.FLOAT, False, 0, 0
    )
    
    gl.uniformMatrix4fv(shader_program.p_matrix_uniform, False, p_matrix)
    gl.uniformMatrix4fv(shader_program.mv_matrix_uniform, False, mv_matrix)

    gl.drawArrays(gl.TRIANGLES, 0, world.shape.num_vertices)

    egl.swap_buffers()


def animate(world):
    pass


def update_curses_display(stdscr, world):
    stdscr.clear()

    stdscr.refresh()


def init_curses_display(stdscr):
    stdscr.nodelay(1)


def main(stdscr):
    init_curses_display(stdscr)
    egl = EGL()
    gl = egl.get_context()
    shader_program = init_shaders(gl)
    texture = init_texture(gl)
    world = load_world(gl)
    gl.clearColor(0.0, 0.0, 0.0, 1.0)
    gl.enable(gl.DEPTH_TEST)
    while True:
        start = time.time()
        handle_keys(stdscr, world)
        update_curses_display(stdscr, world)
        draw_scene(egl, gl, shader_program, world, texture)
        animate(world)
        remainder = (1/60.0) - (time.time() - start)
        if remainder > 0:
            time.sleep(remainder)


    
if __name__ == "__main__":
    curses.wrapper(main)
