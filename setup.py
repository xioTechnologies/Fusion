import sys

import numpy
from setuptools import Extension, setup

ext_modules = Extension(
    "imufusion",
    [
        "Fusion/FusionAhrs.c",
        "Fusion/FusionBias.c",
        "Fusion/FusionCompass.c",
        "Fusion/FusionHardIron.c",
        "Fusion/FusionResult.c",
        "Python/imufusion/imufusion.c",
    ],
    include_dirs=[numpy.get_include()],
    define_macros=[("FUSION_USE_NORMAL_SQRT", None)],
    libraries=(["m"] if "linux" in sys.platform else []),
)  # link math library for Linux

setup(
    ext_modules=[ext_modules],
    packages=["imufusion-stubs"],
    package_dir={"imufusion-stubs": "Python/imufusion-stubs"},
)
