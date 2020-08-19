import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gspan_mining",
    version="0.1",
    author="Naazish Sheikh",
    author_email="Naazish.Sheikh@gmail.com",
    description="Implementation of frequent subgraph mining algorithm closeGraph",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NaazS03/CloseGraph",
    packages=['gspan_mining'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
