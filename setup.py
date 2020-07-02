#!/usr/bin/env python3

from setuptools import setup, find_namespace_packages
from mt.opencv.version import version

setup(name='mtopencv',
      version=version,
      description="Minh-Tri Pham's extra modules using OpenCV",
      author=["Minh-Tri Pham"],
      packages=find_namespace_packages(include=['mt.*']),
      install_requires=[
          'mtbase>=0.4.1',
          # opencv is detected at run-time for flexibility
      ]
      )
