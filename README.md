![unittest](https://github.com/alercebroker/late_classifier/workflows/unittest/badge.svg?branch=main&event=push)
[![codecov](https://codecov.io/gh/alercebroker/late_classifier/branch/main/graph/badge.svg?token=5VNGJTSOCK)](undefined)


# Late Classifier Library

## Installing late_classifier

From PyPI stable version:

```
pip install 
```

For development:

```
pip install -e .
```

# Functionalities

## Augmentation

## Classifier
The classifier code contains BaseClassifier (a simple random forest) and HierarchicalRandomForest (a random forest with internal hierarchy), both with methods for fit and predict.

## Features
For train the models, we use features of astronomical time series. The features are obtained from our different extractors, these receive a DataFrame with detections indexed by `oid`.   

### Preprocess:
Before to get features, we preprocess the time series with filters and boundary conditions:
- Drop duplicates.
- Discard noisy detections.
- Discord bogus.
- Filter time series with more than 5 detections.
- Discard invalid values (like nans and infinite).  

### Extractors:
The extractors are the portion of code with the logic to extract features from time series. Each extractor do only one task, after that our CustomHierarchicalExtractor merge all extractors for get features to train the model.

##### How can I add extractors to library? 
You can use inheritance from `base extractors` and use it for create your own extractor. For now you can inherit:
- `FeatureExtractor` is a generic extractor only fill methods.
- `FeatureExtractorSingleBand` is a extractor that compute features by band.


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
coverage run --source late_classifier -m pytest -x -s tests/
```

If you want run a specify functionality you can run:

```bash
coverage run --source late_classifier -m pytest -x -s tests/<functionality>
```

After that you can see a report of tests:

```bash
coverage report
```

# Reference

If you use this library, please cite our work:

```
@inproceedings{sanchez2020alert,
  title={Alert Classification for the ALeRCE Broker System: The Light Curve Classifier},
  author={S{\'a}nchez-S{\'a}ez, P and Reyes, I and Valenzuela, C and F{\"o}rster, F and Eyheramendy, S and Elorrieta, F and Bauer, FE and Cabrera-Vives, G and Est{\'e}vez, PA and Catelan, M and others},
  year={2020}
}
```
