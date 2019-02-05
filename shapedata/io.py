# functions adopted from menpo.io.input.landmark.py

import numpy as np
import json
import itertools
from collections import OrderedDict


def _ljson_parse_null_values(points_list):
    filtered_points = [np.nan if x is None else x
                       for x in itertools.chain(*points_list)]
    return np.array(filtered_points,
                    dtype=np.float).reshape([-1, len(points_list[0])])


def _parse_ljson_v1(lms_dict):
    all_points = []
    labels = []  # label per group
    labels_slices = []  # slices into the full pointcloud per label
    offset = 0
    connectivity = []
    for group in lms_dict['groups']:
        lms = group['landmarks']
        labels.append(group['label'])
        labels_slices.append(slice(offset, len(lms) + offset))
        # Create the connectivity if it exists
        conn = group.get('connectivity', [])
        if conn:
            # Offset relative connectivity according to the current index
            conn = offset + np.asarray(conn)
            connectivity += conn.tolist()
        for p in lms:
            all_points.append(p['point'])
        offset += len(lms)

    # Don't create a PointUndirectedGraph with no connectivity
    points = _ljson_parse_null_values(all_points)
    return points


def _parse_ljson_v2(lms_dict):

    points = _ljson_parse_null_values(lms_dict['landmarks']['points'])

    return points


_ljson_parser_for_version = {
    1: _parse_ljson_v1,
    2: _parse_ljson_v2
}


def ljson_importer(filepath):
    """
    Importer for the Menpo JSON format. This is an n-dimensional
    landmark type for both images and meshes that encodes semantic labels in
    the format.
    Landmark set label: JSON
    Landmark labels: decided by file

    Parameters
    ----------
    filepath : str
        Absolute filepath of the file.

    Returns
    -------
    np.ndarray
        loaded landmarks
    
    """
    with open(str(filepath), 'r') as f:
        # lms_dict is now a dict rep of the JSON
        lms_dict = json.load(f, object_pairs_hook=OrderedDict)
    v = lms_dict.get('version')
    parser = _ljson_parser_for_version.get(v)

    if parser is None:
        raise ValueError("{} has unknown version {} must be "
                         "1, or 2".format(filepath, v))
    return parser(lms_dict)


def pts_importer(filepath, image_origin=True, z=False, **kwargs):
    """
    Importer for the PTS file format. Assumes version 1 of the format.
    Implementations of this class should override the :meth:`_build_points`
    which determines the ordering of axes. For example, for images, the
    `x` and `y` axes are flipped such that the first axis is `y` (height
    in the image domain).
    Note that PTS has a very loose format definition. Here we make the
    assumption (as is common) that PTS landmarks are 1-based. That is,
    landmarks on a 480x480 image are in the range [1-480]. As Menpo is
    consistently 0-based, we *subtract 1* off each landmark value
    automatically.
    If you want to use PTS landmarks that are 0-based, you will have to
    manually add one back on to landmarks post importing.
    Landmark set label: PTS

    Parameters
    ----------
    filepath : str
        Absolute filepath of the file.
    image_origin : `bool`, optional
        If ``True``, assume that the landmarks exist within an image and thus
        the origin is the image origin.
    **kwargs : `dict`, optional
        Any other keyword arguments.

    Returns
    -------
    np.ndarray
        imported points

    """
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f.readlines()]

    line = lines[0]
    while not line.startswith('{'):
        line = lines.pop(0)

    if not z:
        xs = []
        ys = []
        for line in lines:
            if not line.strip().startswith('}'):
                xpos, ypos = line.split()[:2]
                xs.append(xpos)
                ys.append(ypos)

        xs = np.array(xs, dtype=np.float).reshape((-1, 1))
        ys = np.array(ys, dtype=np.float).reshape((-1, 1))

        # PTS landmarks are 1-based, need to convert to 0-based (subtract 1)
        if image_origin:
            points = np.hstack([ys - 1, xs - 1])
        else:
            points = np.hstack([xs - 1, ys - 1])

    else:
        xs = []
        ys = []
        zs = []
        for line in lines:
            if not line.strip().startswith('}'):
                xpos, ypos, zpos = line.split()[:3]
                xs.append(xpos)
                ys.append(ypos)
                zs.append(zpos)

        xs = np.array(xs, dtype=np.float).reshape((-1, 1))
        ys = np.array(ys, dtype=np.float).reshape((-1, 1))
        zs = np.array(zs, dtype=np.float).reshape((-1, 1))

        # PTS landmarks are 1-based, need to convert to 0-based (subtract 1)
        if image_origin:
            points = np.hstack([zs -1, ys - 1, xs - 1])
        else:
            points = np.hstack([xs - 1, ys - 1, zs-1])

    return points

# adopted from menpo.io.output.landmark.py


def ljson_exporter(lmk_points, filepath, **kwargs):
    """
    Given a file handle to write in to (which should act like a Python `file`
    object), write out the landmark data. No value is returned.
    Writes out the LJSON format which is a verbose format that closely
    resembles the labelled point graph format. It describes semantic
    labels and connectivity between labels. The first axis of the format
    represents the image y-axis and is consistent with ordering within Menpo.

    Parameters
    ----------
    lmk_points : np.ndarray
        The shape to write out.
    filepath : str
        The file to write in to
    """

    lmk_points[np.isnan(lmk_points)] = None

    lmk_points = [list(_tmp) for _tmp in lmk_points]

    ljson = {
        'version': 2,
        'labels': [],
        'landmarks': {
            'points': lmk_points
        }
    }

    with open(filepath, "w") as file_handle:

        return json.dump(ljson, file_handle, indent=4, separators=(',', ': '),
                         sort_keys=True, allow_nan=False, ensure_ascii=False)


def pts_exporter(pts, file_handle, **kwargs):
    """
    Given a file handle to write in to (which should act like a Python `file`
    object), write out the landmark data. No value is returned.
    Writes out the PTS format which is a very simple format that does not
    contain any semantic labels. We assume that the PTS format has been created
    using Matlab and so use 1-based indexing and put the image x-axis as the
    first coordinate (which is the second axis within Menpo).
    Note that the PTS file format is only powerful enough to represent a
    basic pointcloud. Any further specialization is lost.

    Parameters
    ----------
    pts : np.ndarray
        points to save
    file_handle : `file`-like object
        The file to write in to
        
    """
    # Swap the x and y axis and add 1 to undo our processing
    # We are assuming (as on import) that the landmark file was created using
    # Matlab which is 1 based

    if len(pts.shape) == 2:
        pts = pts[:, [1, 0]] + 1
    else:
        pts = pts[:, [2, 1, 0]] + 1

    header = 'version: 1\nn_points: {}\n{{'.format(pts.shape[0])
    np.savetxt(file_handle, pts, delimiter=' ', header=header, footer='}',
               fmt='%.3f', comments='')