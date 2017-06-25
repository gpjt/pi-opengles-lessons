import curses
import numpy as np
from PIL import Image
from random import random
import time

from pygl.egl import EGL
from matrix_utils import perspective, rotate, translate



FRAGMENT_SHADER = """
    precision mediump float;

    varying vec2 vTextureCoord;

    uniform sampler2D uSampler;

    uniform vec3 uColor;

    void main(void) {
        vec4 textureColor = texture2D(uSampler, vec2(vTextureCoord.s, vTextureCoord.t));
        gl_FragColor = textureColor * vec4(uColor, 1.0);
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
    shader_program.color_uniform = gl.getUniformLocation(
        shader_program.gl_program, "uColor"
    )

    return shader_program


def init_texture(gl):
    image = Image.open("star.gif")
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



def init_buffers(gl): 
    shape = Shape(
        gl,
        [
            [-1.0, -1.0,  1.0],
            [ 1.0, -1.0,  1.0],
            [ 1.0,  1.0,  1.0],
            [-1.0,  1.0,  1.0],
        ],
        texture_coords = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ], 
    )

    return shape


class World(object):

    def __init__(self):
        self.tilt = 90
        self.zoom = -15
        self.spin = 0
        self.twinkle = False
        self.stars = []


class Star(object):

    def __init__(self, starting_distance, rotation_speed):
        self.angle = 0
        self.dist = starting_distance
        self.rotation_speed = rotation_speed

        self.randomise_colors()


    def animate(self):
        self.angle += self.rotation_speed
        self.dist -= 0.01
        if self.dist < 0:
            self.dist += 5
        self.randomise_colors()


    def randomise_colors(self):
        self.r = random()
        self.g = random()
        self.b = random()

        self.twinkleR = random()
        self.twinkleG = random()
        self.twinkleB = random()


    def draw(
        self, gl, shader_program, p_matrix, mv_matrix, shape, texture, tilt, spin, twinkle
    ):
        mv_matrix = mv_matrix * rotate(self.angle, np.array([0.0, 1.0, 0.0]))
        mv_matrix = mv_matrix * translate([self.dist, 0.0, 0.0])
        mv_matrix = mv_matrix * rotate(-self.angle, np.array([0.0, 1.0, 0.0]))
        mv_matrix = mv_matrix * rotate(-tilt, np.array([1.0, 0.0, 0.0]))

        if twinkle:
            gl.uniform3f(
                shader_program.color_uniform,
                self.twinkleR, self.twinkleG, self.twinkleB
            )
            draw_star(gl, shader_program, p_matrix, mv_matrix, shape, texture)

        mv_matrix = mv_matrix * rotate(spin, np.array([0.0, 0.0, 1.0]))
        gl.uniform3f(shader_program.color_uniform, self.r, self.g, self.b)
        draw_star(gl, shader_program, p_matrix, mv_matrix, shape, texture)


def draw_star(gl, shader_program, p_matrix, mv_matrix, shape, texture):
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.uniform1i(shader_program.sampler_uniform, 0)
    gl.bindBuffer(gl.ARRAY_BUFFER, shape.texture_coord_buffer)
    gl.vertexAttribPointer(
        shader_program.texture_coord_attribute,
        shape.texture_coord_item_size, gl.FLOAT, False, 0, 0
    )
    gl.bindBuffer(gl.ARRAY_BUFFER, shape.position_buffer)
    gl.vertexAttribPointer(
        shader_program.vertex_position_attribute,
        shape.position_item_size, gl.FLOAT, False, 0, 0
    )
    gl.uniformMatrix4fv(shader_program.p_matrix_uniform, False, p_matrix)
    gl.uniformMatrix4fv(shader_program.mv_matrix_uniform, False, mv_matrix)
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, shape.num_vertices)


def init_world_objects():
    world = World()
    num_stars = 50
    for i in range(num_stars):
        world.stars.append(Star((i / float(num_stars)) * 5.0, i / float(num_stars)))
    return world


def draw_scene(egl, gl, shader_program, world, shape, texture):
    gl.bindFramebuffer(gl.FRAMEBUFFER, 0)
    gl.viewport(0, 0, egl.width, egl.height)

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

    p_matrix = perspective(45, float(egl.width.value) / float(egl.height.value), 0.1, 100.0)

    gl.blendFunc(gl.SRC_ALPHA, gl.ONE)
    gl.enable(gl.BLEND)

    mv_matrix = np.identity(4)
    mv_matrix = mv_matrix * translate([0.0, 0.0, world.zoom])
    mv_matrix = mv_matrix * rotate(world.tilt, np.array([1, 0, 0]))

    for star in world.stars:
        star.draw(
            gl, shader_program, p_matrix, mv_matrix, shape, texture,
            world.tilt, world.spin, world.twinkle
        )
        world.spin += 0.1

    egl.swap_buffers()


def handle_keys(stdscr, world):
    c = stdscr.getch()
    if c == 339:
        # Page Up
        world.zoom -= 0.1
    elif c == 338:
        # Page Down
        world.zoom += 0.1
    elif c == 259:
        # Up cursor key
        world.tilt += 2
    elif c == 258:
        # Down cursor key
        world.tilt -= 2
    elif c == 32:
        world.twinkle = not world.twinkle
    elif c != -1:
        print("Unrecognised keycode {}".format(c))


def animate(world):
    for star in world.stars:
        star.animate()


def update_curses_display(stdscr, world):
    stdscr.clear()

    stdscr.addstr(2, 1, "Twinkle (space to toggle): {}".format(world.twinkle))

    stdscr.addstr(3, 1, "Zoom: {}".format(world.zoom))
    stdscr.addstr(4, 1, "Tilt: {}".format(world.tilt))

    stdscr.refresh()


def init_curses_display(stdscr):
    stdscr.nodelay(1)


def main(stdscr):
    init_curses_display(stdscr)
    egl = EGL()
    gl = egl.get_context()
    shader_program = init_shaders(gl)
    shape = init_buffers(gl)
    texture = init_texture(gl)
    world = init_world_objects()
    gl.clearColor(0.0, 0.0, 0.0, 1.0)
    while True:
        start = time.time()
        handle_keys(stdscr, world)
        update_curses_display(stdscr, world)
        draw_scene(egl, gl, shader_program, world, shape, texture)
        animate(world)
        remainder = (1/60.0) - (time.time() - start)
        if remainder > 0:
            time.sleep(remainder)


    
if __name__ == "__main__":
    curses.wrapper(main)
