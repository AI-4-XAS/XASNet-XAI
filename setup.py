import io
import os
from setuptools import setup, find_packages

def read(
    fname,
):
    with io.open(
        os.path.join(
            os.path.dirname(__file__),
            fname,
        ),
        encoding="utf-8",
    ) as f:
        return f.read()

setup(
    name="XASNet-XAI",
    packages=find_packages("src"),
    package_dir={"": "src"},
    version="1.0.0",
    author="Amir Kotobi",
    scripts=[],
    include_package_data=True,
    install_requires=[
        "torch>=1.12",
        "torch_geometric",
        "numpy",
        "pandas",
        "scipy",
        "ase>=3.21",
        "scikit-learn",
        "tqdm",
        "rdkit",
    ],
)