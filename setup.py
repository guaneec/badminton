from setuptools import setup, find_packages

setup(
    name="shuttle",
    version="0.0.1",
    python_requires=">=3.8, <3.9",
    packages=find_packages(where="src"),
    install_requires=["opencv-contrib-python", "tensorflow", "tqdm"],
)
