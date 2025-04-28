from setuptools import setup, find_packages

setup(
    name="pcm-spaghetti",  
    version="0.1.0",  
    author="Richard (Zhi Fei) Dong, Chris McIntosh, Gregory W. Schwartz",
    author_email="gregory.schwartz@uhn.ca",
    description="A PyTorch implementation of the SPAGHETTI model for phase-contrast microscopy image transformation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/schwartzlab-methods/spaghetti",  
    packages=find_packages(exclude=["tutorials*"]),  
    entry_points={
        "console_scripts": [
            "spaghetti=spaghetti.cli_inference:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9", 
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "numpy>=1.26.4",
        "Pillow>=10.3.0",
        "scikit-image>=0.18.3",
        "pytorch-lightning>=2.3.3",
        "tqdm>=4.66.4",
    ],
)
