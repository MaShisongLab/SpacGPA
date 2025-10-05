from pathlib import Path
from setuptools import setup, find_packages

README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="SpacGPA",  
    version="0.1.0",
    description="Spatial and single-cell Gene Program Analysis (SpacGPA)",
    long_description=README,
    long_description_content_type="text/markdown",
    author="MaShisongLab",
    url="https://github.com/MaShisongLab/SpacGPA",
    license="BSD-3-Clause",
    packages=find_packages(include=["SpacGPA", "SpacGPA.*"]),
    include_package_data=True,
    package_data={
        # SpacGPA/Ref_for_Enrichment/*
        "SpacGPA": ["Ref_for_Enrichment/*"],
    },
    python_requires=">=3.8",
    install_requires=[
        # Core scientific stack
        "numpy",
        "scipy",
        "pandas",
        "anndata",
        "scanpy",
        "matplotlib",
        "seaborn",
        "statsmodels",
        "scikit-learn",
        "h5py",

        # Graph & clustering
        "networkx",
        "igraph>=0.10",          
        "leidenalg>=0.10",
        "python-louvain>=0.16",  

        # Annotation utilities
        "mygene",

        # torch and related
        "torch",
        "torchvision",
        "torchaudio",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)