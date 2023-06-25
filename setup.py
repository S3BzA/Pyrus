from setuptools import setup, find_packages

with open('config/requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='Pyrus',
    version='1.0.0',
    author='Sebastian Guimaraens',
    description='A collection of niche python utility scripts.',
    packages=find_packages(),
    install_requires=requirements
)
