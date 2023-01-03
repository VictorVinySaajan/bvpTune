# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="bvpTune",
    version="0.1.0",
    description="Library for fine tuning the numerical settings of boundary value problem solvers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://bvpTune.readthedocs.io/",
    author="Viny Saajan Victor",
    author_email="viny.saajan.victor@itwm.fraunhofer.de",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7.11",
        "Operating System :: OS Independent"
    ],
    packages=["code"],
    include_package_data=True,
    install_requires=["numpy", "pandas", "scikit-learn", "lightgbm", "keras", "tensorflow", "pickleshare", "optuna", "plotly"]
)
