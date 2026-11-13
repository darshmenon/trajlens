"""Optional C++ speed-up (trajlens._fast). If pybind11 or a compiler is missing, the build skips it and trajlens runs
on numpy alone."""
from setuptools import setup

ext_modules = []
try:
    import pybind11
    from setuptools import Extension

    ext_modules = [Extension("trajlens._fast", ["cpp/fast_grad.cpp"], include_dirs=[pybind11.get_include()],
                             extra_compile_args=["-O3", "-std=c++17"], optional=True)]
except Exception:
    pass

setup(ext_modules=ext_modules)
