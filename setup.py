from setuptools import setup, find_packages

setup(
    name="anPysotropy",
    version="0.1.0",
    description="Toolkit for modelling anisotropy (velocity and attenuation) for fluid-filled fracture rock physics models.",
    author="Joseph Asplet, Mark Chapman",
    author_email="joseph.asplet@earth.ox.ac.uk",
    url="https://github.com/Jasplet/anPysotropy",
    # Automatically find all packages (folders with __init__.py)
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.26.4",
        "numba",
        "obspy>=1.4.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
    license="BSD-3-Clause",
    keywords="seismology anisotropy rock physics",
    licence_files=("LICENSE"),
    test_suite="tests"
)