import curses
import numpy as np
from PIL import Image
import time

from pygl.egl import EGL
from matrix_utils import perspective, rotate, translate



FRAGMENT_SHADER = """
    precision mediump float;

    varying vec2 vTextureCoord;
    varying vec3 vLightWeighting;

    uniform sampler2D uSampler;

    void main(void) {
        vec4 textureColor = texture2D(uSampler, vec2(vTextureCoord.s, vTextureCoord.t));
        gl_FragColor = vec4(textureColor.rgb * vLightWeighting, textureColor.a);
        gl_FragColor = vec4(1.0, 1.0, 1.0, textureColor.a);
    }
"""

VERTEX_SHADER = """
    attribute vec3 aVertexPosition;
    attribute vec3 aVertexNormal;
    attribute vec2 aTextureCoord;

    uniform mat4 uMVMatrix;
    uniform mat4 uPMatrix;
    uniform mat3 uNMatrix;

    uniform vec3 uAmbientColor;

    uniform vec3 uLightingDirection;
    uniform vec3 uDirectionalColor;

    uniform bool uUseLighting;

    varying vec2 vTextureCoord;
    varying vec3 vLightWeighting;

    void main(void) {
        gl_Position = uPMatrix * uMVMatrix * vec4(aVertexPosition, 1.0);
        vTextureCoord = aTextureCoord;

        if (!uUseLighting) {
            vLightWeighting = vec3(1.0, 1.0, 1.0);
        } else {
            vec3 transformedNormal = uNMatrix * aVertexNormal;
            float directionalLightWeighting = max(dot(transformedNormal, uLightingDirection), 0.0);
            vLightWeighting = uAmbientColor + uDirectionalColor * directionalLightWeighting;
        }
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

    shader_program.vertex_normal_attribute = gl.getAttribLocation(
        shader_program.gl_program, "aVertexNormal"
    )
    gl.enableVertexAttribArray(shader_program.vertex_normal_attribute)

    shader_program.p_matrix_uniform = gl.getUniformLocation(
        shader_program.gl_program, "uPMatrix"
    )
    shader_program.mv_matrix_uniform = gl.getUniformLocation(
        shader_program.gl_program, "uMVMatrix"
    )
    shader_program.n_matrix_uniform = gl.getUniformLocation(
        shader_program.gl_program, "uNMatrix"
    )
    gl.check_for_error()
    shader_program.sampler_uniform = gl.getUniformLocation(
        shader_program.gl_program, "uSampler"
    )
    gl.check_for_error()
    shader_program.use_lighting_uniform = gl.getUniformLocation(
        shader_program.gl_program, "uUseLighting"
    )
    gl.check_for_error()
    shader_program.use_lighting_uniform = gl.getUniformLocation(
        shader_program.gl_program, "uUseLighting"
    )
    gl.check_for_error()
    shader_program.ambient_color_uniform = gl.getUniformLocation(
        shader_program.gl_program, "uAmbientColor"
    )
    gl.check_for_error()
    shader_program.lighting_direction_uniform = gl.getUniformLocation(
        shader_program.gl_program, "uLightingDirection"
    )
    gl.check_for_error()

    return shader_program


def init_texture(gl):
    image = Image.open("crate.gif")
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image_data = np.array(image.convert("RGBA"))

    texture = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.RGBA, image.width, image.height,
        0, gl.RGBA, gl.UNSIGNED_BYTE, image_data
    )
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_NEAREST)
    gl.generateMipmap(gl.TEXTURE_2D)

    gl.bindTexture(gl.TEXTURE_2D, None)

    return texture


class Shape:

    def __init__(self, gl, positions, indices, texture_coords, normals):
        self.position_buffer = gl.createBuffer()
        gl.bindBuffer(gl.ARRAY_BUFFER, self.position_buffer)
        gl.bufferData(gl.ARRAY_BUFFER, sum(positions, []), gl.STATIC_DRAW)
        self.position_item_size = len(positions[0])

        self.num_vertices = len(positions)

        self.index_buffer = gl.createBuffer()
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, self.index_buffer)
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW)
        self.num_indices = len(indices)

        self.texture_coord_buffer = gl.createBuffer()
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, self.texture_coord_buffer)
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, sum(texture_coords, []), gl.STATIC_DRAW)
        self.texture_coord_item_size = len(texture_coords[0])

        self.normal_buffer = gl.createBuffer()
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, self.normal_buffer)
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, sum(normals, []), gl.STATIC_DRAW)
        self.normal_item_size = len(normals[0])

        self.xRot = 0
        self.xSpeed = 0
        self.yRot = 0
        self.ySpeed = 0

        self.z = -5.0

        self.lighting = True



def init_buffers(gl): 
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
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],

            # Back face
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],

            # Top face
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],

            # Bottom face
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],

            # Right face
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],

            # Left face
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ], 
        normals = [
            # Front face
            [ 0.0,  0.0,  1.0],
            [ 0.0,  0.0,  1.0],
            [ 0.0,  0.0,  1.0],
            [ 0.0,  0.0,  1.0],

            # Back face
            [ 0.0,  0.0, -1.0],
            [ 0.0,  0.0, -1.0],
            [ 0.0,  0.0, -1.0],
            [ 0.0,  0.0, -1.0],

            # Top face
            [ 0.0,  1.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [ 0.0,  1.0,  0.0],

            # Bottom face
            [ 0.0, -1.0,  0.0],
            [ 0.0, -1.0,  0.0],
            [ 0.0, -1.0,  0.0],
            [ 0.0, -1.0,  0.0],

            # Right face
            [ 1.0,  0.0,  0.0],
            [ 1.0,  0.0,  0.0],
            [ 1.0,  0.0,  0.0],
            [ 1.0,  0.0,  0.0],

            # Left face
            [-1.0,  0.0,  0.0],
            [-1.0,  0.0,  0.0],
            [-1.0,  0.0,  0.0],
            [-1.0,  0.0,  0.0],
        ]
    )

    return cube_shape


class LightData(object):

    def __init__(self):
        self.on = True
        self.ambient_r = 0.2
        self.ambient_g = 0.2
        self.ambient_b = 0.2
    
        self.direction_x = -0.25
        self.direction_y = -0.25
        self.direction_z = -1.0

        self.directional_r = 0.8
        self.directional_g = 0.8
        self.directional_b = 0.8


def init_lighting():
    return LightData()


def draw_scene(egl, gl, shader_program, cube_shape, texture, lighting):
    gl.bindFramebuffer(gl.FRAMEBUFFER, 0)
    gl.viewport(0, 0, egl.width, egl.height)

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

    p_matrix = perspective(45, float(egl.width.value) / float(egl.height.value), 0.1, 100.0)
    mv_matrix = np.identity(4)

    mv_matrix = mv_matrix * translate([0.0, 0.0, cube_shape.z])
    mv_matrix = mv_matrix * rotate(cube_shape.xRot, np.array([1, 0, 0]))
    mv_matrix = mv_matrix * rotate(cube_shape.yRot, np.array([0, 1, 0]))
    gl.bindBuffer(gl.ARRAY_BUFFER, cube_shape.position_buffer)
    gl.vertexAttribPointer(
        shader_program.vertex_position_attribute,
        cube_shape.position_item_size, gl.FLOAT, False, 0, 0
    )
    gl.bindBuffer(gl.ARRAY_BUFFER, cube_shape.texture_coord_buffer)
    gl.vertexAttribPointer(
        shader_program.texture_coord_attribute,
        cube_shape.texture_coord_item_size, gl.FLOAT, False, 0, 0
    )
    gl.bindBuffer(gl.ARRAY_BUFFER, cube_shape.normal_buffer)
    gl.vertexAttribPointer(
        shader_program.vertex_normal_attribute,
        cube_shape.normal_item_size, gl.FLOAT, False, 0, 0
    )

    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.uniform1i(shader_program.sampler_uniform, 0)

    gl.uniform1i(shader_program.use_lighting_uniform, 1) # lighting.on)
    if lighting.on:
        gl.uniform3f(
            shader_program.ambient_color_uniform,
            lighting.ambient_r,
            lighting.ambient_g,
            lighting.ambient_b,
        )
        print("Ambient lighting", lighting.ambient_r, lighting.ambient_g, lighting.ambient_b)
        gl.check_for_error()
        
        lighting_direction = np.array([
            lighting.direction_x,
            lighting.direction_y,
            lighting.direction_z,
        ])
        norm = np.linalg.norm(lighting_direction)
        if norm != 0:
            lighting_direction /= norm
        lighting_direction *= -1
        print("Lighting direction: ", lighting_direction)
        gl.uniform3fv(shader_program.lighting_direction_uniform, lighting_direction)
        gl.check_for_error()

        gl.uniform3f(
            shader_program.ambient_color_uniform,
            lighting.directional_r,
            lighting.directional_g,
            lighting.directional_b,
        )
        print("Directional lighting: ", lighting.directional_r, lighting.directional_g, lighting.directional_b)
        gl.check_for_error()
    

    gl.uniformMatrix4fv(shader_program.p_matrix_uniform, False, p_matrix)
    gl.check_for_error()
    gl.uniformMatrix4fv(shader_program.mv_matrix_uniform, False, mv_matrix)
    gl.check_for_error()
    print("MV matrix", mv_matrix)
    normal_matrix = mv_matrix[:3,:3].I.T
    print("Normal matrix:", normal_matrix)
    gl.uniformMatrix3fv(shader_program.n_matrix_uniform, False, normal_matrix)
    gl.check_for_error()

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, cube_shape.index_buffer)
    gl.drawElements(gl.TRIANGLES, cube_shape.num_indices, gl.UNSIGNED_SHORT, 0)

    egl.swap_buffers()


def handle_keys(stdscr, cube_shape):
    c = stdscr.getch()
    if c == 339:
        # Page Up
        cube_shape.z -= 0.05
    elif c == 338:
        # Page Down
        cube_shape.z += 0.05
    elif c == 260:
        # Left cursor key
        cube_shape.ySpeed -= 1
    elif c == 261:
        # Right cursor key
        cube_shape.ySpeed += 1
    elif c == 259:
        # Up cursor key
        cube_shape.xSpeed -= 1
    elif c == 258:
        # Down cursor key
        cube_shape.xSpeed += 1
    elif c != -1:
        print("Unrecognised keycode {}".format(c))


def animate(cube_shape):
    cube_shape.xRot = (cube_shape.xRot - 0.8 * cube_shape.xSpeed) % 360
    cube_shape.yRot = (cube_shape.yRot - 0.8 * cube_shape.ySpeed) % 360


def main(stdscr):
    stdscr.nodelay(1)
    egl = EGL()
    gl = egl.get_context()
    shader_program = init_shaders(gl)
    cube_shape = init_buffers(gl)
    texture = init_texture(gl)
    lighting = init_lighting()
    gl.clearColor(0.0, 0.0, 0.0, 1.0)
    gl.enable(gl.DEPTH_TEST)
    while True:
        start = time.time()
        handle_keys(stdscr, cube_shape)
        draw_scene(egl, gl, shader_program, cube_shape, texture, lighting)
        animate(cube_shape)
        remainder = (1/60.0) - (time.time() - start)
        if remainder > 0:
            time.sleep(remainder)


    
if __name__ == "__main__":
    curses.wrapper(main)
