from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    # Extension("vectorization", ["vectorization.pyx"], include_dirs=[numpy.get_include()])
    Extension("yado_vectorization", ["yado_vectorization.pyx"], include_dirs=[numpy.get_include()])
]

setup(
    name='yado_vectorization',
    ext_modules=cythonize(extensions),
)
