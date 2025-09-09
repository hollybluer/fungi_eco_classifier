#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      holly
#
# Created:     19/08/2025
# Copyright:   (c) holly 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from setuptools import setup, find_packages

setup(
    name = "fungi_model",
    version = "0.1.0",
    author = "Bluer",
    author_email = "holly.bluer@gmail.com"
    description = "A machine learning pipeline to identify dominant fungal species in a community based on environmental data."
    long_description = open("README.md").read(),
    long_description_content_type = "text/markdown",
    url = "https://github.com/hollybluer/fungi_model",
    packages = find_packages(),
    install_requires = [
    "pandas >= 1.5.0",
    "numpy >= 1.25.0",
    "scikit-learn >= 1.3.0".
    "joblib >= 1.3.0"
],
classifiers = [
    "Programming language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating system :: OS independent"
],
python_requires = ">= 3.10",
)