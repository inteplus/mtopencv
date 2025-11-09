#!/usr/bin/env python3

import os
from setuptools import setup, find_namespace_packages

VERSION_FILE = os.path.join(os.path.dirname(__file__), "VERSION.txt")

setup(
    name="mtopencv",
    description="Minh-Tri Pham's extra modules using OpenCV",
    author=["Minh-Tri Pham"],
    packages=find_namespace_packages(include=["mt.*"]),
    scripts=[
        "scripts/draw_chessboard_corners",
    ],
    install_requires=[
        # 'h5py>=3', # for pdh5 file format. Lazy import because TX2 may not need it.
        # 'opencv-python', # let them install opencv-python or opencv-python-headless or whatever
        "ansicolors",  # for displaying images on the terminal
        "pyturbojpeg",  # for encoding/decoding images
        "mtbase>=4.33.2",  # just updating
        "mtgeo>=1.1.13",  # just updating
    ],
    setup_requires=["setuptools-git-versioning<2"],
    setuptools_git_versioning={
        "enabled": True,
        "version_file": VERSION_FILE,
        "count_commits_from_version_file": True,
        "template": "{tag}",
        "dev_template": "{tag}.dev{ccount}+{branch}",
        "dirty_template": "{tag}.post{ccount}",
    },
)
