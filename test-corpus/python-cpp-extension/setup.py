
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "fast_math",
        ["src/fast_math.cpp"],
        extra_compile_args=["-O3", "-march=native"],
    ),
]

setup(
    name="fast_math",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
