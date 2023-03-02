import numpy
import os
import shutil
import sys
from setuptools import setup, Extension

for folder in ["build", "dist", "imufusion.egg-info"]:
    if os.path.exists(folder) and os.path.isdir(folder):
        shutil.rmtree(folder)

if len(sys.argv) == 1:  # if this script was called without arguments
    sys.argv.append("install")
    sys.argv.append("--user")

github_url = "https://github.com/xioTechnologies/Fusion"

ext_modules = Extension("imufusion", ["Python/Python-C-API/imufusion.c",
                                      "Fusion/FusionAhrs.c",
                                      "Fusion/FusionCompass.c",
                                      "Fusion/FusionOffset.c"],
                        include_dirs=[numpy.get_include()],
                        libraries=(["m"] if "linux" in sys.platform else []))  # link math library for Linux

setup(name="imufusion",
      version="1.1.0",
      description="Fusion Python package",
      long_description="See [github](" + github_url + ") for documentation and examples.",
      long_description_content_type='text/markdown',
      url=github_url,
      author="x-io Technologies Limited",
      author_email="info@x-io.co.uk",
      license="MIT",
      classifiers=["Programming Language :: Python :: 3.8",
                   "Programming Language :: Python :: 3.9",
                   "Programming Language :: Python :: 3.10",
                   "Programming Language :: Python :: 3.11"],  # versions shown by pyversions badge in README
      ext_modules=[ext_modules])
