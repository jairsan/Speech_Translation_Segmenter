from setuptools import setup, find_packages
setup(
    name='segmenter',
    version='1.0.0',
    packages=find_packages(exclude=["examples", "scripts"]),
    url='https://github.com/jairsan/Speech_Translation_Segmenter',
    license='Apache License 2.0',
    author='Javier Iranzo-Sanchez',
    description='RNN-based streaming segmenter for Speech Translation',
    install_requires=[
        "torch==1.7.0",
        "scikit-learn==0.21.3",
        "datasets==2.8.0",
        "transformers==4.24.0",
        "sentencepiece==0.1.97"]
)
