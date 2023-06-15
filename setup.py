# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from os.path import abspath, dirname, join

this_dir = abspath(dirname(__file__))

with open(join(this_dir, "LICENSE")) as f:
    license = f.read()

with open(join(this_dir, "README.md"), encoding="utf-8") as file:
    long_description = file.read()

with open(join(this_dir, "requirements.txt")) as f:
    requirements = f.read().split("\n")


setup(
    name="grad_june",
    version="0.1.4",
    description="Differentiable implementation of the JUNE model.",
    url="https://github.com/arnauqb/GradABM-JUNE",
    long_description_content_type="text/markdown",
    long_description=long_description,
    author="Arnau Quera-Bofarull",
    author_email="arnauq@protonmail.com",
    license="MIT License",
    install_requires=requirements,
    packages=find_packages(exclude=["docs"]),
    include_package_data=True,
)
