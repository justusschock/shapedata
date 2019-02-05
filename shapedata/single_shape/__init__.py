# author: Justus Schock (justus.schock@rwth-aachen.de)

"""
Module to provide Dataloading for images with single shapes

"""

from .data_processing import DataProcessing as SingleShapeDataProcessing
from .data_processing import SingleImage2D as SingleShapeSingleImage2D
from .dataset import ShapeDataset as SingleShapeDataset
