"""
Setup script for mineru_client package
"""

from setuptools import setup, find_packages

with open("README_MINERU.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mineru-client",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Python client for MinerU document parsing API with bidirectional coordinate mapping",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mineru-client",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
)
