from setuptools import setup, find_packages

setup(
    name="bol_wandb",
    version="0.1.0",
    description="Weights & Biases integration for the Blocks of Life (BoL) test suite",
    author="juiceb0xc0de",
    packages=find_packages(),
    install_requires=[
        "wandb",
        "transformers",
        "torch",
        "numpy",
        "tqdm",
        "scipy"
    ],
)
