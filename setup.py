from setuptools import setup, find_packages
setup(
    name='segmenter',
    version='0.2.0a1',
    packages=find_packages(exclude=["examples", "scripts"]),
    url='https://github.com/jairsan/Speech_Translation_Segmenter',
    license='Apache License 2.0',
    author='Javier Iranzo-Sanchez',
    description='RNN-based streaming segmenter for Speech Translation',
    install_requires=[
        "torch==1.7.0",
        "scikit-learn==0.21.3",
        "datasets==2.8.0"]
)
