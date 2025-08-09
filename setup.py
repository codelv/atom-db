"""
Copyright (c) 2019-2021, Jairus Martin.

Distributed under the terms of the MIT License.

The full license is in the file LICENSE.text, distributed with this software.

Created on Feb 21, 2019

@author: jrm
"""
import re
from setuptools import Extension, setup, find_packages

def find_version():
    with open("atomdb/__init__.py") as f:
        for line in f:
            m = re.search(r'version = [\'"](.+)["\']', line)
            if m:
                return m.group(1)
    raise Exception("Could not find version in atomdb/__init__.py")

ext_module = Extension(
    "atomdb.ext",
    ["src/ext.c"],
    include_dirs=["src"],
    language="c",
)


setup(
    name="atom-db",
    version=find_version(),
    author="CodeLV",
    author_email="frmdstryr@gmail.com",
    url="https://github.com/codelv/atom-db",
    description="Database abstraction layer for atom objects",
    license="MIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    requires=["atom"],
    python_requires=">=3.7",
    install_requires=["atom>=0.7.0"],
    optional_requires=[
        "sqlalchemy<2",
        "aiomysql",
        "aiopg",
        "aiosqlite",  # sql database support
        "motor",  # nosql database support
    ],
    ext_modules=[ext_module],
    packages=find_packages(),
    package_data={'atomdb': ["py.typed"]}
)
