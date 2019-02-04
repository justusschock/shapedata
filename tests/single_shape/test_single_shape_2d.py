from shapedata.single_shape import SingleShapeSingleImage2D
import os
import warnings

def test_single_image_2d():
    img = SingleShapeSingleImage2D.from_files(os.path.join(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
        "example_files", "lenna.png"))
    
    assert not img.is_gray
    assert img.is_cartesian
    assert not img.is_homogeneous

    assert img.resize((500, 500)).img.shape[:-1] == (500, 500)

    assert img.resize((200, 500)).rotate(
        90, output_shape=(500, 200)).img.shape[:-1] == (500, 200)

    assert img.lmk.shape == (68, 2)

    assert img.rescale((1, 5))
    assert img.crop_to_landmarks()
    assert img.homogeneous_coordinates().is_homogeneous

    assert img.homogeneous_coordinates().cartesian_coordinates().is_cartesian
    assert img.cartesian_coordinates().is_cartesian

