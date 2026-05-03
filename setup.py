from setuptools import setup, find_packages

setup(
    name="deep_chaos_scheduler",
    version="0.1.0",
    description="Sticky-topology chaos scheduler for transformer fine-tuning, with BoL diagnostic scans",
    author="juiceb0xc0de",
    license="MIT",
    packages=find_packages(),
    py_modules=["bol_scans"],
    python_requires=">=3.10",
    install_requires=[
        "wandb",
        "transformers",
        "torch",
        "peft",
        "numpy",
        "tqdm",
        "scipy"
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
