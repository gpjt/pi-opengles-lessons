import time

from pygl.egl import EGL


FRAGMENT_SHADER = """
    precision mediump float;

    varying vec2 vPosition;

    void main(void) {
        float cx = vPosition.x;
        float cy = vPosition.y;

        float hue;
        float saturation;
        float value;
        float hueRound;
        int hueIndex;
        float f;
        float p;
        float q;
        float t;


        float x = 0.0;
        float y = 0.0;
        float tempX = 0.0;
        int i = 0;
        int runaway = 0;
        for (int i=0; i < 100; i++) {
            tempX = x * x - y * y + float(cx);
            y = 2.0 * x * y + float(cy);
            x = tempX;
            if (runaway == 0 && x * x + y * y > 100.0) {
                runaway = i;
            }
        }

        if (runaway != 0) {
            hue = float(runaway) / 200.0;
            saturation = 0.6;
            value = 1.0;

            hueRound = hue * 6.0;
            hueIndex = int(mod(float(int(hueRound)), 6.0));
            f = fract(hueRound);
            p = value * (1.0 - saturation);
            q = value * (1.0 - f * saturation);
            t = value * (1.0 - (1.0 - f) * saturation);

            if (hueIndex == 0)
                gl_FragColor = vec4(value, t, p, 1.0);
            else if (hueIndex == 1)
                gl_FragColor = vec4(q, value, p, 1.0);
            else if (hueIndex == 2)
                gl_FragColor = vec4(p, value, t, 1.0);
            else if (hueIndex == 3)
                gl_FragColor = vec4(p, q, value, 1.0);
            else if (hueIndex == 4)
                gl_FragColor = vec4(t, p, value, 1.0);
            else if (hueIndex == 5)
                gl_FragColor = vec4(value, p, q, 1.0);

        } else {
            gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        }
    }
"""

VERTEX_SHADER = """
    attribute vec2 aVertexPosition;
    attribute vec2 aPlotPosition;

    varying vec2 vPosition;

    void main(void) {
        gl_Position = vec4(aVertexPosition, 1.0, 1.0);
        vPosition = aPlotPosition;
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

    shader_program.plot_position_attribute = gl.getAttribLocation(
        shader_program.gl_program, "aPlotPosition"
    )
    gl.enableVertexAttribArray(shader_program.plot_position_attribute)

    return shader_program


class Shape:

    def __init__(self, gl, vertices):
        self.buffer = gl.createBuffer()
        gl.bindBuffer(gl.ARRAY_BUFFER, self.buffer)
        gl.bufferData(gl.ARRAY_BUFFER, sum(vertices, []), gl.STATIC_DRAW)
        self.item_size = len(vertices[0])
        self.num_items = len(vertices)


def init_buffers(gl):
    shape = Shape(
        gl,
        [
            [ 1.0,  1.0],
            [-1.0,  1.0],
            [ 1.0, -1.0],
            [-1.0, -1.0],
        ]
    )
    shape.zoom = 1.0
    shape.zoom_center_x = 0.28693186889504513
    shape.zoom_center_y = 0.014286693904085048
    shape.center_offset_x = 0
    shape.center_offset_y = 0
    return shape


BASE_CORNERS = [
    ( 0.7,  1.2),
    (-2.2,  1.2),
    ( 0.7, -1.2),
    (-2.2, -1.2),
];

def draw_scene(egl, gl, shader_program, shape):
    gl.bindFramebuffer(gl.FRAMEBUFFER, 0)
    gl.viewport(0, 0, egl.width, egl.height)

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

    gl.bindBuffer(gl.ARRAY_BUFFER, shape.buffer)
    gl.vertexAttribPointer(
        shader_program.vertex_position_attribute,
        shape.item_size, gl.FLOAT, False, 0, 0
    )

    plot_position_buffer = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, plot_position_buffer)
    corners = []
    for x, y in BASE_CORNERS:
        corners.append(x / shape.zoom + shape.center_offset_x)
        corners.append(y / shape.zoom + shape.center_offset_y)
    gl.bufferData(gl.ARRAY_BUFFER, corners, gl.STATIC_DRAW);
    gl.vertexAttribPointer(
        shader_program.plot_position_attribute,
        2, gl.FLOAT, False, 0, 0
    )

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)

    gl.deleteBuffer(shader_program.plot_position_attribute)

    egl.swap_buffers()



def animate(shape):
    shape.zoom *= 1.02
    if shape.center_offset_x != shape.zoom_center_x:
        shape.center_offset_x += (shape.zoom_center_x - shape.center_offset_x) / 20

    if shape.center_offset_y != shape.zoom_center_y:
        shape.center_offset_y += (shape.zoom_center_y - shape.center_offset_y) / 20


def main():
    egl = EGL()
    gl = egl.get_context()
    shader_program = init_shaders(gl)
    shape = init_buffers(gl)
    gl.clearColor(0.0, 0.0, 0.0, 1.0)
    while True:
        start = time.time()
        draw_scene(
            egl, gl, shader_program,
            shape
        )
        animate(shape)
        remainder = (1/60.0) - (time.time() - start)
        if remainder > 0:
            time.sleep(remainder)


if __name__ == "__main__":
    main()

