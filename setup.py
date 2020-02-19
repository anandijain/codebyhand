from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="codebyhand",
    version="0.01",
    description="hate typing?",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Anand Jain",
    author_email="anandj@uchicago.edu",
    packages=["codebyhand"],  # same as name
    url="https://github.com/anandijain/codebyhand",
    install_requires=[
        "numpy",
        "flask",
        "torch",
        "torchaudio",
        "torchvision",
        "Pillow",
    ],  # external packages as dependencies
    python_requires=">=3.6",
)
