from setuptools import find_packages, setup


setup(
    name="semantic_mllm_py_pkg",
    version="0.1.0",
    package_dir={"": "scripts"},
    packages=find_packages("scripts"),
)
