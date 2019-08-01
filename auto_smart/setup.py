#!/usr/bin/env python
# coding: utf-8

import setuptools
from Cython.Build import cythonize

setuptools.setup(
    name='AutoSmart',
    version='0.0.1',
    author='DeepBlueAI',
    author_email='1229991666@qq.com',
    url='https://github.com/DeepBlueAI/AutoSmart',
    description=u'The 1st place solution for KDD Cup 2019 AutoML Track',
    packages=setuptools.find_packages(),
    install_requires=[
        "hyperopt",
        "lightgbm",
        "joblib",
        "pandas",
        ],
    ext_modules = cythonize("ac.pyx"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
)
