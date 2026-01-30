import setuptools

with open("README.MD", "r") as fh:
    long_description = fh.read()

# Read the requirements from requirements.txt
with open("requirements.txt", "r") as req_file:
    requirements = req_file.read().splitlines()

setuptools.setup(
    name="dz_lib",
    version="1.5.2",
    author="Ryan Nielsen",
    author_email="nielrya4@isu.edu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nielrya4/dz_lib",
    packages=setuptools.find_packages(),
    install_requires=requirements,  # Automatically populates from requirements.txt
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
