import setuptools

from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="qsolve",
    version="0.3.2",
    url = "https://github.com/jfmennemann/qsolve",
    author="Jan-Frederik Mennemann",
    author_email="jfmennemann@gmx.de",
    description="Numerical framework for the simulation and optimization of ultracold atom experiments",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[]
)

