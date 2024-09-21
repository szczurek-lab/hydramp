from setuptools import find_packages
from setuptools import setup

setup(
    name='HydrAMP',
    version='1.2.0',
    description='Python package for peptide generation',
    author='Paulina Szymczak',
    author_email='szymczak.pau@gmail.com',
    url='https://hydramp.mimuw.edu/',
    packages=find_packages(),
    install_requires=[
        'tensorflow~=2.2.1',
        'tensorflow-probability~=0.10.0',
        'Keras~=2.3.1',
        'Keras-Applications~=1.0.8',
        'Keras-Preprocessing~=1.1.2',
        'cloudpickle~=1.4.1',
        'numpy~=1.18.5',
        'pandas~=1.1.4',
        'scikit-learn~=0.23.2',
        'modlamp~=4.2.3',
        'matplotlib~=3.3.2',
        'protobuf~=3.14.0',
        'seaborn~=0.11.0',
        'setuptools~=50.3.1',
        'joblib~=0.17.0',
        'argparse',
        'tqdm~=4.51.0',
        'biopython==1.83',
        'gdown>=5.2.0'
    ],
    setup_requires=['wheel']
)
