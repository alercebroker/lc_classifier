![unittest](https://github.com/alercebroker/late_classifier/workflows/unittest/badge.svg?branch=main&event=push)
[![codecov](https://codecov.io/gh/alercebroker/lc_classifier/branch/main/graph/badge.svg?token=5VNGJTSOCK)](undefined)

# Light Curve Classifier Library

## Installing late_classifier

From PyPI stable version:

```
pip install numpy Cython
pip install -e git+https://git@github.com/alercebroker/turbo-fats#egg=turbofats
pip install -e git+https://git@github.com/alercebroker/mhps#egg=mhps
pip install -e git+https://git@github.com/alercebroker/P4J#egg=P4J
pip install lc-classifier
```

For development:

```
git clone https://github.com/alercebroker/lc_classifier.git
python -m pip install -e .
```

## How to use the library?

Check the available Jupyter notebooks in the *examples* directory.

## Functionalities

### Feature computation
This library provides an extensive number of feature extractors for astronomical
light curves, including period computation, autoregresive models, parametric models,
statistical features, etc. We also provide tools to transform your data into 
the format that this library expects (Pandas dataframes).

### Augmentation
If you want more samples you can use our ShortTransientAugmenter class.
More data augmentation techniques will be implemented in further releases.

### Classifier
Two classifiers are available: A traditional Random Forest model, and a hierarchical
model made from 4 Random Forest classifiers.

### Preprocessing for ZTF data:
Before computing features, we preprocess the time series with filters 
and boundary conditions:
- Drop duplicate observations.
- Discard noisy detections.
- Discard bogus.
- Filter time series with more than 5 detections.
- Discard invalid values (like nans and infinite).  


## How can I add my own feature extractors to the library?
Feature extractors extend the following classes:
- `FeatureExtractor`
- `FeatureExtractorSingleBand`. This type of feature extractor runs independently 
on each available band.
  
Check out the existent feature extractors in the directory 
*lc_classifier/features/extractors*.


## Profile functionalities
The easiest way to profile a step is using cProfile, for this we just have to run the step with the following command:

```bash
python -m cProfile -o <outputfile> profiling/<functionality>.py
```

After that you can run `snakeviz` (first install it).

```
snakeviz <outputfile>
```

## Test functionalities
You must first install the following packages:

```bash
pip install coverage pytest
```

All scripts of tests must be in `tests` folder. For run all tests:

```bash
coverage run --source lc_classifier -m pytest -x -s tests/
```

If you want run a specify functionality you can run:

```bash
coverage run --source lc_classifier -m pytest -x -s tests/<functionality>
```

After that you can see a report of tests:

```bash
coverage report
```

# Run a container

This repository comes with a Dockerfile to test the model.

To build the image run
```
docker build -t alerce/lc_classifier
```
Then run the container
```
docker run --rm -p 8888:8888 alerce/lc_classifier
```
The container comes with a jupyter notebook and some examples in `http://localhost:8888`

# Reference

If you use this library, please cite our work:

```
@inproceedings{sanchez2020alert,
  title={Alert Classification for the ALeRCE Broker System: The Light Curve Classifier},
  author={S{\'a}nchez-S{\'a}ez, P and Reyes, I and Valenzuela, C and F{\"o}rster, F and Eyheramendy, S and Elorrieta, F and Bauer, FE and Cabrera-Vives, G and Est{\'e}vez, PA and Catelan, M and others},
  year={2020}
}
```
