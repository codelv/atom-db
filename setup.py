"""
Copyright (c) 2019, Jairus Martin.

Distributed under the terms of the MIT License.

The full license is in the file LICENSE.text, distributed with this software.

Created on Feb 21, 2019

@author: jrm
"""
from setuptools import setup, find_packages

setup(
    name='atom-db',
    version='0.2.0',
    author='CodeLV',
    author_email='frmdstryr@gmail.com',
    url='https://github.com/codelv/atom-db',
    description='Database abstraction layer for atom objects',
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    requires=['atom'],
    install_requires=['atom'],
    optional_requires=[
        'sqlalchemy', 'aiomysql', 'aiopg',  # sql database support
        'motor', 'txmongo'  # nosql database support
    ],
    packages=find_packages(),
)

