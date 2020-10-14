[![Build Status](https://travis-ci.com/alercebroker/late_classifier.svg?token=FuwtsLbsSNgHY1qXBVmB&branch=paper_paula)](https://travis-ci.com/alercebroker/late_classifier)
[![codecov](https://codecov.io/gh/alercebroker/late_classifier/branch/paper_paula/graph/badge.svg?token=5VNGJTSOCK)](https://codecov.io/gh/alercebroker/late_classifier)


# Late Classifier Library

## Installing late_classifier
For development:

```
pip install -e .
```


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


## Augmentation

## Classifier

## Features

### Core:

- Base:
- Decorators:
- Helpers:

### Preprocess:

- Base:
- Preprocess ZTF:

### Extractors:

- Color feature:
- Galactic coordinates:
- Hierarchical:
- IQR:
- MHPS:
- Real bogus:
- SG score:
- SN detections:
- SN non-detections:
- Turbofats:

#### How can I add extractors to library? 
You can use inheritance from `base extractors` and use it for create your own extractor. For now you can inherit:
- `FeatureExtractor` is a generic extractor only fill methods.
- `FeatureExtractorSingleBand` is a extractor that compute features by band.

