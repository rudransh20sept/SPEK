# setup.py
from setuptools import setup, find_packages

setup(
    name="SPEK",
    version="0.0.4",
    author="Rudransh Joshi",
    author_email="rudransh20septmber@gmail.com",
    description="SPEK: Simple Python Extraction Kit - Easy YOLOv8 Object Detection",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "opencv-python",
        "ultralytics>=8.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video :: Capture",
        "Topic :: Multimedia :: Graphics :: Capture :: Digital Camera",
    ],
    entry_points={
        "console_scripts": [
            "spek=spek.__main__:main",
        ],
    },
)
