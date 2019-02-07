# ShapeData

[![PyPI version](https://badge.fury.io/py/shapedata.svg)](https://badge.fury.io/py/shapedata) [![Build Status](https://travis-ci.com/justusschock/shapedata.svg?token=GsT2RFaJJMxpqLAN3xuh&branch=master)](https://travis-ci.com/justusschock/shapedata) [![Documentation Status](https://readthedocs.org/projects/shapedata/badge/?version=master)](https://shapedata.readthedocs.io/en/master/?badge=master) [![codecov](https://codecov.io/gh/justusschock/shapedata/branch/master/graph/badge.svg?token=PeQndVRdEQ)](https://codecov.io/gh/justusschock/shapedata) ![LICENSE](https://img.shields.io/github/license/justusschock/shapedata.svg)

This repository contains image classes to perform transformations on images with landmarks (similar to [menpo](https://github.com/menpo/menpo) but with much less dependencies). It also provides some basic Datasets for [delira](https://github.com/justusschock/delira)

## Installation
This package can be installed via `pip install shapedata`

## Basic Usage
To load a single image with landmarks you can simply do

```python
import shapedata
img = shapedata.SingleShapeSingleImage2D.from_files("./example_files/lenna.png")
```

and to view this image do
```python
from matplotlib import pyplot as plt
img.view(view_landmarks=True)
plt.show()
```

To augment the image you can use `img.transform()` to transform the image with the origin as transformation base or `img.transform_about_centre()` to use the images's center as transformation base.
Transformations as `img.translate()`, `img.rotate()`, `img.rescale()` or `img.resize()` are also implemented and will fall back on `img.transform()` or `img.transform_about_centre()`

To access the image itself you can do `img.img` and accessing the landmarks works with `img.lmk` for instances of `SingleShapeSingleImage`
For further usage have a look at the datasets and docstrings.

