from shapedata.io import ljson_exporter, ljson_importer
from shapedata.io import pts_exporter, pts_importer
import os
import numpy as np 


def test_io():
    lmks = np.loadtxt(os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 
        "example_files", "lenna.txt"))

    ljson_exporter(lmks, "./lmks.ljson")
    assert os.path.isfile("./lmks.ljson")

    pts_exporter(lmks, "./lmks.pts")
    assert os.path.isfile("./lmks.pts")

    lmks_ljson = ljson_importer("./lmks.ljson")
    assert (lmks==lmks_ljson).all()

    lmks_pts = pts_importer("./lmks.pts")
    assert (lmks==lmks_ljson).all()

    os.remove("./lmks.ljson")
    os.remove("./lmks.pts")