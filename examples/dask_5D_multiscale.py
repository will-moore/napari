"""
Display a 5D dask multiscale pyramid
"""

try:
    from dask import array as da
    from dask import delayed
except ImportError:
    raise ImportError("""This example uses a dask array but dask is not
    installed. To install try 'pip install dask'.""")

import numpy as np
import napari
from math import ceil


def get_tile(tile_name):
    """Return a tile for the given coordinates"""
    print('get_tile level, t, c, z, y, x, w, h', tile_name)
    level, t, c, z, y, x, w, h = [int(n) for n in tile_name.split(",")]

    def f2(x, y):
        # Try to return a tile that depends on level and z, c, t
        if c % 2 == 1:
            return (y + (2 * t) + (2 * z))
        else:
            return (x + ((level % 2) * y)) // 2

    plane_2d = np.fromfunction(f2, (h, w), dtype=np.int16)
    return plane_2d

lazy_reader = delayed(get_tile)


def get_lazy_plane(level, t, c, z, plane_y, plane_x, tile_shape):

    print('get_lazy_plane: level, t, c, z, plane_y, plane_x', level, t, c, z, plane_y, plane_x)

    tile_w, tile_h = tile_shape
    rows = ceil(plane_y / tile_h)
    cols = ceil(plane_x / tile_w)
    print('rows', rows, 'cols', cols)

    lazy_rows = []
    for row in range(rows):
        lazy_row = []
        for col in range(cols):
            x = col * tile_w
            y = row * tile_h
            w = min(tile_w, plane_x - x)
            h = min(tile_h, plane_y - y)
            tile_name = "%s,%s,%s,%s,%s,%s,%s,%s" % (level, t, c, z, y, x, w, h)
            lazy_tile = da.from_delayed(lazy_reader(tile_name), shape=(h, w), dtype=np.int16)
            lazy_row.append(lazy_tile)
        lazy_row = da.concatenate(lazy_row, axis=1)
        print('lazy_row.shape', lazy_row.shape)
        lazy_rows.append(lazy_row)
    return da.concatenate(lazy_rows, axis=0)


def get_pyramid_lazy(shape, tile_shape, levels):
    """Get a pyramid of rgb dask arrays, loading tiles from OMERO."""

    size_t, size_c, size_z, size_y, size_x = shape

    pyramid = []
    plane_x = size_x
    plane_y = size_y
    for level in range(levels):
        print('level', level)
        t_stacks = []
        for t in range(size_t):
            c_stacks = []
            for c in range(size_c):
                z_stack = []
                for z in range(size_z):
                    lazy_plane = get_lazy_plane(level, t, c, z, plane_y, plane_x, tile_shape)
                    z_stack.append(lazy_plane)
                c_stacks.append(da.stack(z_stack))
            t_stacks.append(da.stack(c_stacks))
        pyramid.append(da.stack(t_stacks))
        plane_x = plane_x // 2
        plane_y = plane_y // 2

    print ('pyramid...')
    for level in pyramid:
        print(level.shape)
    return pyramid


shape = (10, 2, 5, 3000, 5000)
tile_shape = (256, 256)
levels = 4
pyramid = get_pyramid_lazy(shape, tile_shape, levels)

with napari.gui_qt():
    viewer = napari.view_image(pyramid, channel_axis=1)