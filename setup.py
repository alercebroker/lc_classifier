from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    required_packages = f.readlines()

required_packages = [r for r in required_packages if "-e" not in r] 

setup(
    name="lc_classifier",
    version="2.0.1",
    description="ALeRCE light curve classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ALeRCE Team",
    author_email="contact@alerce.online",
    packages=find_packages(),
    install_requires=required_packages,
    build_requires=required_packages,
)
