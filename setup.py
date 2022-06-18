#!/usr/bin/env python3

from setuptools import setup, find_namespace_packages
from mt.opencv.version import version

setup(name='mtopencv',
      version=version,
      description="Minh-Tri Pham's extra modules using OpenCV",
      author=["Minh-Tri Pham"],
      packages=find_namespace_packages(include=['mt.*']),
      scripts=[
          'scripts/immview',
      ],
      install_requires=[
          # 'h5py>=3', # for pdh5 file format. Lazy import because TX2 may not need it.
          # 'opencv-python', # let them install opencv-python or opencv-python-headless or whatever
          'ansicolors',  # for displaying images on the terminal
          'pyturbojpeg',  # for encoding/decoding images
          'mtbase>=2.17',
          'mtgeo>=0.9',
      ]
      )
