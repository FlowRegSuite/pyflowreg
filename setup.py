from skbuild import setup
from setuptools import find_packages


setup(
    name="pyflowreg",
    version="0.1.0",
    packages=find_packages(),
    cmake_install_dir="pyflowreg",
    cmake_generator="Visual Studio 17 2022",
    cmake_args=[
    #    "-GMinGW Makefiles",
    #    "-DCMAKE_MAKE_PROGRAM=C:/mingw64/bin/mingw32-make.exe",
    #    "-DCMAKE_C_COMPILER=C:/mingw64/bin/gcc.exe",
    #    "-DCMAKE_CXX_COMPILER=C:/mingw64/bin/g++.exe",
    #    "-DCMAKE_SH=NOTFOUND",
        #"-DSKBUILD_NO_INSTALL=ON",
        #"-DSKBUILD_SKIP_INSTALL=ON",
        "-G", "Visual Studio 17 2022",  # Specify your Visual Studio version
        "-A", "x64",  # Ensure 64-bit architecture
        "-Dpybind11_DIR=C:/Users/Philipp/anaconda3/envs/pyflowreg/Lib/site-packages/pybind11/share/cmake/pybind11"
    ]
)
