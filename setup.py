import numpy
import sys
from setuptools import setup, Extension

ext_modules = Extension("imufusion", ["Python/Python-C-API/imufusion.c",
                                      "Fusion/FusionAhrs.c",
                                      "Fusion/FusionCompass.c",
                                      "Fusion/FusionOffset.c"],
                        include_dirs=[numpy.get_include()],
                        define_macros=[("FUSION_USE_NORMAL_SQRT", None)],
                        libraries=(["m"] if "linux" in sys.platform else []))  # link math library for Linux

setup(ext_modules=[ext_modules])
