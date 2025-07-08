import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Obtener configuración OpenCV usando pkg-config
import subprocess

def get_opencv_flags():
    libs = subprocess.check_output(['pkg-config', '--libs', 'opencv4']).decode('utf-8').strip().split()
    cflags = subprocess.check_output(['pkg-config', '--cflags', 'opencv4']).decode('utf-8').strip().split()
    return libs, cflags

class get_pybind_include:
    def __str__(self):
        import pybind11
        return pybind11.get_include()

opencv_libs, opencv_cflags = get_opencv_flags()

ext_modules = [
    Extension(
        'enhanced_stitcher',
        ['ImageClasses/enhanced_stitcher.cpp'],
        include_dirs=[
            get_pybind_include(),
            '/usr/include/opencv4'  # Ajusta la ruta según tu sistema
        ],
        extra_compile_args=['-std=c++11'] + opencv_cflags,
        extra_link_args=opencv_libs,
        language='c++'
    ),
]

setup(
    name='enhanced_stitcher',
    version='0.1',
    author='Capstone Team',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)