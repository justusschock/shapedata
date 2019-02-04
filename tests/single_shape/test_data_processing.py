from shapedata.single_shape import SingleShapeDataProcessing
import os
import warnings

def test_data_processing():
    data = SingleShapeDataProcessing.from_dir(os.path.join(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
        "example_files"))

    assert data.samples
    data.resize((224, 224))
    assert data[0].img.shape[: -1] == (224, 224)

    assert data.images
    assert data.landmarks
    assert data.lmk_pca(True, True).shape == (2, 68, 2)
    
    data[0] = 5
    assert data[0] == 5

    try:
        data[1] = 500
        assert False, "Should raise IndexError since index 1 is out\
         of bound"
    except IndexError:
        assert True