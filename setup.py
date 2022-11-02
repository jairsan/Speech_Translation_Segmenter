from setuptools import setup, find_packages
setup(
    name='segmenter',
    version='pre0.2.0',
    packages=find_packages(exclude=["examples", "scripts"]),
    url='https://github.com/jairsan/Speech_Translation_Segmenter',
    license='Apache License 2.0',
    author='Javier Iranzo-Sanchez',
    description='RNN-based streaming segmenter for Speech Translation'
)
