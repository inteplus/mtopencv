#!/usr/bin/env python3

from setuptools import setup, find_namespace_packages
from mt.opencv.version import version

setup(name='mtopencv',
      version=version,
      description="Minh-Tri Pham's extra modules using OpenCV",
      author=["Minh-Tri Pham"],
      packages=find_namespace_packages(include=['mt.*']),
      scripts=[
          'scripts/immview.py',
      ],
      install_requires=[
          'mtbase>=1.2.6',
          'mtgeo>=0.7.5',
          'opencv-python',
          'ascii_magic', # for displaying images on the terminal
          'pyturbojpeg', # for encoding/decoding images
      ]
      )
