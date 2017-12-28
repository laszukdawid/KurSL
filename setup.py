import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "kursl",
    version = "0.0.1",
    author = "Dawid Laszuk",
    author_email = "laszukdawid@gmail.com",
    description = ("Implementation of KurSL method"),
    license = "",
    keywords = "example documentation tutorial",
    url = "http://packages.python.org/an_example_pypi_project",
    packages=['kursl', 'tests'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
    ],
)
