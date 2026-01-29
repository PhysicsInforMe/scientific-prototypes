"""Setup script for timeseries_toolkit package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="timeseries_toolkit",
    version="0.1.0",
    author="Scientific Prototypes",
    description="A comprehensive toolkit for time series analysis and forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/timeseries-toolkit",
    packages=find_packages(exclude=["tests", "notebooks", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "statsmodels>=0.13.0",
    ],
    extras_require={
        "viz": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
        "models": [
            "lightgbm>=3.0.0",
            "hmmlearn>=0.2.7",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black",
            "isort",
            "mypy",
        ],
        "all": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "lightgbm>=3.0.0",
            "hmmlearn>=0.2.7",
        ],
    },
)
