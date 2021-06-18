import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cgspan_mining",
    version="0.1",
    author="Naazish Sheikh",
    author_email="Naazish.Sheikh@gmail.com",
    description="Implementation of closed frequent subgraph mining algorithm cgSpan",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NaazS03/cgSpan",
    packages=['cgspan_mining'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
