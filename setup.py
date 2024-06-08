
from setuptools import setup, find_packages

setup(
    name='gee_mgwr',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'geemap',
        'earthengine-api',
        'matplotlib',
        'mgwr'
    ],
    author='Neil Linehan',
    author_email='neiledwardlinehan@gmail.com',
    description='A package for performing a multiscale MGWR analysis on Google Earth Engine images',
    url='https://github.com/neillinehan/gee-mgwr',
)
