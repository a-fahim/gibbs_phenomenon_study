from setuptools import setup, find_packages

setup(
    name='gibbsphen',
    version='0.1',
    author='Ali Fahim',
    author_email='a.fahim@gmail.com',
    description='A package for studying the Gibbs phenomenon using Neural Networks',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'pandas',
    ],
    python_requires='>=3.6',
)
