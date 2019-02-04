from shapedata.single_shape import SingleShapeDataset
import os
import pytest

@pytest.mark.parametrize("crop,rotate,cached,random_offset,random_scale",
[
    (0.1, 45, True, 5, 0.1),
    (0.1, 45, True, 5, 0.0),
    (0.1, 45, True, 0, 0.1),
    (0.1, 45, True, 0, 0.0),
    (0.1, 45, False, 5, 0.1),
    (0.1, 45, False, 5, 0.0),
    (0.1, 45, False, 0, 0.1),
    (0.1, 45, False, 0, 0.0),
    (0.1, None, True, 5, 0.1),
    (0.1, None, True, 5, 0.0),
    (0.1, None, True, 0, 0.1),
    (0.1, None, True, 0, 0.0),
    (0.1, None, False, 5, 0.1),
    (0.1, None, False, 5, 0.0),
    (0.1, None, False, 0, 0.1),
    (0.1, None, False, 0, 0.0),
    (None, 45, True, 5, 0.1),
    (None, 45, True, 5, 0.0),
    (None, 45, True, 0, 0.1),
    (None, 45, True, 0, 0.0),
    (None, 45, False, 5, 0.1),
    (None, 45, False, 5, 0.0),
    (None, 45, False, 0, 0.1),
    (None, 45, False, 0, 0.0),
    (None, None, True, 5, 0.1),
    (None, None, True, 5, 0.0),
    (None, None, True, 0, 0.1),
    (None, None, True, 0, 0.0),
    (None, None, False, 5, 0.1),
    (None, None, False, 5, 0.0),
    (None, None, False, 0, 0.1),
    (None, None, False, 0, 0.0)
]
)
def test_dataset(crop, rotate, cached, random_offset, random_scale):

    dset = SingleShapeDataset(os.path.join(
        os.path.dirname(os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)))), 
        "example_files"), 224, 
        crop=crop, extension=".ljson", 
        rotate=rotate, cached=cached, 
        random_offset=random_offset, 
        random_scale=random_scale)

    assert dset[0]
    assert "data" in dset[0]
    assert "label" in dset[0]

    try:
        dset[1]
        assert False, "Should raise Indexerror,\
            because index 1 is out of bounds"
    except IndexError:
        assert True