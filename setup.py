from setuptools import setup, find_packages

setup(
    name="lifelong-vdm",
    version="1.0.0",
    description='Code for "Lifelong Learning of Video Diffusion Models From a Single Video Stream"',
    packages=find_packages(),
    python_requires=">=3.10",
    # The full pinned dependency list lives in requirements.txt (and
    # requirements-jedi.txt for the separate JEDi evaluation environment).
    install_requires=["torch", "numpy", "blobfile", "tqdm"],
)
