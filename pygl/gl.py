import ctypes
import numpy as np

opengles = ctypes.CDLL('libGLESv2.so')

import pygl.gl2


class _BaseGL:

    def __getattr__(self, attr_name):
        constant_name = "GL_{}".format(attr_name)
        if hasattr(pygl.gl2, constant_name):
            return getattr(pygl.gl2, constant_name)

        gl_func_name = "gl{}{}".format(attr_name[0].upper(), attr_name[1:])
        if hasattr(opengles, gl_func_name):
            return getattr(opengles, gl_func_name)

        raise AttributeError(attr_name)



class GL:

    def __init__(self):
        self.base_gl = _BaseGL()


    def check_for_error(self):
        e = self.getError()
        if e:
            raise Exception("GL error {}".format(hex(e)))


    def clearColor(self, r, g, b, a):
        if isinstance(r, float):
            r = ctypes.c_float(r)
        if isinstance(g, float):
            g = ctypes.c_float(g)
        if isinstance(b, float):
            b = ctypes.c_float(b)
        if isinstance(a, float):
            a = ctypes.c_float(a)
        self.base_gl.clearColor(r, g, b, a)


    def getProgramParameter(self, program, parameter):
        result = ctypes.c_int()
        self.base_gl.getProgramiv(program, parameter, ctypes.byref(result))
        return result.value


    def getProgramInfoLog(self, program):
        N = 1024
        log = (ctypes.c_char*N)()
        loglen = ctypes.c_int()
        self.base_gl.getProgramInfoLog(program, N, ctypes.byref(loglen), ctypes.byref(log))
        return log.value.decode("utf-8")


    def shaderSource(self, shader, source):
        if isinstance(source, str):
            source = source.encode("utf-8")
        self.base_gl.shaderSource(shader, 1, ctypes.byref(ctypes.c_char_p(source)), 0)


    def getShaderParameter(self, shader, parameter):
        result = ctypes.c_int()
        self.base_gl.getShaderiv(shader, parameter, ctypes.byref(result))
        return result.value


    def getShaderInfoLog(self, shader):
        N = 1024
        log = (ctypes.c_char*N)()
        loglen = ctypes.c_int()
        self.base_gl.getShaderInfoLog(shader, N, ctypes.byref(loglen), ctypes.byref(log))
        return log.value.decode("utf-8")


    def getAttribLocation(self, program, attrib):
        if isinstance(attrib, str):
            attrib = attrib.encode("utf-8")
        return self.base_gl.getAttribLocation(program, attrib)


    def getUniformLocation(self, program, uniform):
        if isinstance(uniform, str):
            uniform = uniform.encode("utf-8")
        return self.base_gl.getUniformLocation(program, uniform)


    def createBuffer(self):
        result = ctypes.c_int()
        self.base_gl.genBuffers(1, ctypes.byref(result))
        return result


    def bufferData(self, target, data, usage):
        data_type = None
        for item in data:
            if data_type is None:
                data_type = type(item)
            elif type(item) != data_type:
                raise Exception(
                    "Found element of type {} when first was {}".format(type(item), data_type)
                )

        if data_type == float:
            converted_data = (ctypes.c_float*len(data))(*data)
        elif data_type == int:
            converted_data = (ctypes.c_short*len(data))(*data)
        else:
            raise Exception("I don't know how to convert type {}".format(data_type))

        self.base_gl.bufferData(
            target,
            ctypes.sizeof(converted_data), ctypes.byref(converted_data),
            usage
        )


    def uniformMatrix4fv(self, location, transpose, value):
        if transpose != False:
            raise ValueError("Transpose must be False -- see OpenGL spec")
        if isinstance(value, np.ndarray):
            value = np.asfortranarray(value, dtype=np.float32).ctypes.data
        self.base_gl.uniformMatrix4fv(location, 1, transpose, value)


    def __getattr__(self, attr_name):
        return getattr(self.base_gl, attr_name)
