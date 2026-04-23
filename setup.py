from setuptools import setup, find_packages

setup(
    name="lucky_pick_scheduler",
    version="0.1.0",
    description="Weights & Biases integration for the Blocks of Life (BoL) test suite + Scheduler",
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
