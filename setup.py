from setuptools import setup, find_packages

# Read version
version = {}
with open("ml_lib/version.py") as f:
    exec(f.read(), version)

setup(
    name="ml_lib",
    version=version["__version__"],
    author="Mohammad Sahil Bhat",
    description="Machine Learning library built from scratch",
    packages=find_packages(),
    python_requires=">=3.14.0",
    install_requires=[
        'numpy>=2.3.5',
        'pandas>= 2.3.3',
        'matplotlib>=3.10.8'
    ],
)

