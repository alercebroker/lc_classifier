from setuptools import setup, find_packages

setup(
    name="late_classifier",
    version="0.1",
    packages=find_packages(),
    author="ALeRCE Team",
    install_requires=[
        'pandas',
        'numpy',
        'astropy'
    ]
)
