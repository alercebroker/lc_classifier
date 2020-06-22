from setuptools import setup

with open("requirements.txt") as f:
    required_packages = f.readlines()

required_packages = [r for r in required_packages if "-e" not in r]

setup(
    name="late_classifier",
    version="0.1",
    description='Scripts for ALeRCE late classifier.',
    author="ALeRCE Team",
    author_email='contact@alerce.online',
    packages=['late_classifier'],
    scripts=["scripts/lateclassifier"],
    install_requires=required_packages,
    build_requires=required_packages
)
