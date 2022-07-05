from setuptools import setup

with open("README.md", mode="r", encoding="utf-8") as fp:
    long_description = fp.read()


setup(
    name="pytorch-autotrainer",
    version="0.1.0",
    description="Wrapper for PyTorch model training.",
    long_description=long_description,
    author="Dani El-Ayyass",
    author_email="dayyass@yandex.ru",
    license_files=["LICENSE"],
    url="https://github.com/dayyass/pytorch-autotrainer",
    packages=["pytorch_autotrainer"],
    install_requires=[
        "numpy==1.21.6",
        "tensorboard==2.9.1",
        "torch==1.12.0",
        "tqdm==4.64.0",
    ],
)
