from setuptools import setup, find_packages

with open("README.md", "r") as fh:  # If you have a README
    long_description = fh.read()

setup(
    name="principal-distortions",
    version="0.1.0",
    author="Jenelle Feather, David Lipshutz",
    author_email="jenellefeather@gmail.com",
    description="Principal Distortions, ICLR 2025",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jfeather/principal-distortions",
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        "plenoptic==1.1.0",
        "torch==2.5.1",
        "timm==1.0.11",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={
        "principal_distortions": ["imagenet_classes.txt", "images/*"],
    },
    python_requires='>=3.7',  # Specify minimum Python version
)
