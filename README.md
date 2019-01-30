# ShapeNet-Data

This repository contains image classes to perform transformations on images with landmarks (similar to [menpo](https://github.com/menpo/menpo) but with much less dependencies). It also provides some basic Datasets for [delira](https://github.com/justusschock/delira)

## Installation
This package can be installed via `pip install git+https://github.com/justusschock/shapenet_data.git`

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

To access the image itself you can do `img.img` and accessing the landmarks works with `img.lmk` for instances of `SingleShapeSingleImage` and `img.lmks` for instances of `MultiShapesSingleImage`
For furhter usage have a look at the datasets and docstrings.

## Requirements
Requirements and their versions are listed in the [setup.py](./setup.py) and include:
* [numpy](http://www.numpy.org/)
* [scikit-image](https://scikit-image.org/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [tqdm](https://github.com/tqdm/tqdm)
* [torch>=0.4](https://pytorch.org/)
* [matplotlib](https://matplotlib.org)
* [delira](https//github.com/justusschock/delira)
