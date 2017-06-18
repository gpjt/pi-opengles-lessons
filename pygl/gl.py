import ctypes 

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


    def shaderSource(self, shader, source):
        if isinstance(source, str):
            source = source.encode("utf-8")
        self.base_gl.shaderSource(shader, 1, ctypes.byref(ctypes.c_char_p(source)), 0)


    def getShaderParameter(self, shader, parameter):
        result = ctypes.c_int()
        self.base_gl.getShaderiv(shader, parameter, ctypes.byref(result))
        return result


    def getShaderInfoLog(self, shader):
        N = 1024
        log = (ctypes.c_char*N)()
        loglen = ctypes.c_int()
        self.base_gl.getShaderInfoLog(shader, N, ctypes.byref(loglen), ctypes.byref(log))
        return log.value.decode("utf-8")


    def __getattr__(self, attr_name):
        return getattr(self.base_gl, attr_name)
