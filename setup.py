from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='confidenceset',
    install_requires=[
        'numpy',
        'scipy',
    ],
    version = '0.0.2',
    license='MIT',
    author='Howon Ryu',
    download_url='https://github.com/HowonRyu/ConfidenceSet',
    author_email='howonryu@ucsd.edu',
    url='https://github.com/HowonRyu/ConfidenceSet',
    long_description=long_description,
    description='Python toolbox functions for FDR controlled confidence sets',
    keywords='FDR, Confidence Sets',
    packages=find_packages(),
    python_requires='>=3',
)
