#!/usr/bin/env python3
from setuptools import find_packages, setup


setup(
    name="semantic_decision_py_pkg",
    version="0.1.0",
    package_dir={"": "scripts"},
    packages=find_packages("scripts"),
)
