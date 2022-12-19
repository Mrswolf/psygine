# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r", errors="ignore", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="psygine",
    version="0.01",
    author="swolf",
    author_email="swolfforever@gmail.com",
    description="The best engine for cyberpsychos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages("psygine"),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)