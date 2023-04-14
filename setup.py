import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "kursl",
    version = "0.1.0",
    author = "Dawid Laszuk",
    author_email = "laszukdawid@gmail.com",
    description = ("Implementation of KurSL method"),
    license = "",
    keywords = "scientific signal-processing numerical",
    url = "https://github.com/laszukdawid/kursl",
    packages=['kursl', 'tests'],
    install_requires=['numpy (>=1.13.3,<1.20.0)', 'scipy (>=0.19.1,<1.0)', "emcee (>=2.2.1,<3.0)", 'matplotlib (>=1.0,<2.0)'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
