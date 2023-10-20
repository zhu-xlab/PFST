import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "RSI-DASS",
    version = "1.0.0",
    author = "Fahong Zhang",
    author_email = "fahongzhang@gmail.com",
    description = ("Code libraries for remote sensing images-based domain adaptive semantic segmentation."),
    license = "BSD",
    keywords = "Remote Sensing Image-based Domain Adaptive Semantic Segmentatioon",
    url = "http://packages.python.org/RSI-DASS",
    packages=['rsiseg', 'tests'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
