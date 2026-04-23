from setuptools import setup, find_packages

setup(
    name="lucky_pick_scheduler",
    version="0.1.0",
    description="Trainer-agnostic BoL scan suite with W&B logging",
    author="juiceb0xc0de",
    packages=find_packages(),
    py_modules=["bol_scans"],
    install_requires=[
        "wandb",
        "transformers",
        "torch",
        "peft",
        "numpy",
        "tqdm",
        "scipy"
    ],
)
