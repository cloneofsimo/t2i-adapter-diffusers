import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="t2i_adapters",
    py_modules=["t2i_adapters"],
    version="0.1.0",
    description="T2I-Adapters, compatible with diffusers",
    author="Simo Ryu",
    packages=find_packages(),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
)
