"""
This module is aimed to classify tweets location based on its content.
"""

from __future__ import print_function

import itertools
import numpy as np
import settings
import preprocess


def _get_bounds(coords):
    """
    Return a bounding box coordinates for a collection of points from dataframe.
    """
    return ((coords.min()[settings.CSV.INPUT.LATITUDE], 
             coords.max()[settings.CSV.INPUT.LATITUDE]),
            (coords.min()[settings.CSV.INPUT.LONGITUDE], 
             coords.max()[settings.CSV.INPUT.LONGITUDE]))


def in_bounds(dataframe, bounds):
    """
    Return the dataframe of the points that are in the map bounding box.
    """
    return dataframe.query(settings.CSV.INPUT.LATITUDE + '>=' + str(bounds[0][0])) \
                    .query(settings.CSV.INPUT.LATITUDE + '<=' + str(bounds[0][1])) \
                    .query(settings.CSV.INPUT.LONGITUDE + '>=' + str(bounds[1][0])) \
                    .query(settings.CSV.INPUT.LONGITUDE + '<=' + str(bounds[1][1]))


def _to_cells(coords, bounding_box, pattern):
    """
    Split a dataframe with geocoordinates into four cells.
    """
    batches = []
    cells = []

    ((lat_0, lat_n), (lon_0, lon_n)) = bounding_box

    lats = np.linspace(lat_0, lat_n, pattern + 1)
    lons = np.linspace(lon_0, lon_n, pattern + 1)

    cells = list(itertools.product(
        zip(lats[:], lats[1:]),
        zip(lons[:], lons[1:])
    ))
    batches = [in_bounds(coords, cell) for cell in cells]
    return batches, cells


# stop criteria
def _stop_by_depth(self):
    return self.depth >= self.max_depth


def _stop_by_point_number(self):
    return len(self.coords) <= self.max_per_grid


def _to_dec(number, base):
    """
    Convert a number to decimal from 'base'
    """
    assert type(number) is list
    return sum([(int(v) * base**i) for i, v in enumerate(number[::-1])])


class QuadTree:
    """Quad-tree class which recursively subdivide the space into quadrants"""

    def __init__(self, coords, depth=0, 
                 max_per_grid=settings.CLUSTER.MAXIMUM_PER_GRID,
                 max_depth=settings.CLUSTER.MAXIMUM_DEPTH,
                 pattern=settings.CLUSTER.PATTERN,
                 bounds=None, stop_criteria=None):
        self.coords = coords.sort_values([settings.CSV.INPUT.LATITUDE, 
                                          settings.CSV.INPUT.LONGITUDE])
        self.children = []
        self.bounds = _get_bounds(coords) if not bounds else bounds
        self.depth = depth
        self.max_depth = max_depth
        self.max_per_grid = max_per_grid
        self.stop_criteria = stop_criteria
        self.pattern = pattern
        self.height = self.depth

        # stop if no point anymore
        if len(self.coords) == 0:
            return

        # stop by chosen criteria
        if stop_criteria(self):
            return
        else:
            batches, cells = _to_cells(coords, self.bounds, pattern)
            for batch, cell in zip(batches, cells):
                self.children.append(QuadTree(
                    batch, depth=depth + 1, bounds=cell, stop_criteria=stop_criteria, pattern=pattern))
    
        # calculation the height of the whole tree
        self.height = max(self.height, max(child.height for child in self.children))
        

    def assign_labels(self, grid_labels, centroids, path):
        """
        DFS procudure to assign labels to every point.
        """

        if not self.children:
            class_id = "G" + str(_to_dec(path, self.pattern ** 2))
            for key in self.coords.index:
                grid_labels[key] = class_id

            centroids[class_id] = self.coords.mean()
        else:
            for i, child in enumerate(self.children):
                child.assign_labels(grid_labels, centroids, path + [i])


def by_grid(input_filename, output_filename, map_bounds, by_depth):
    """
    The entry point of the module.
    """

    dataframe = preprocess.read_dataframe(input_filename)
    # fliter points that are not in the bounds
    dataframe = in_bounds(dataframe, map_bounds)
    coords = dataframe[[settings.CSV.INPUT.LATITUDE, settings.CSV.INPUT.LONGITUDE]]

    stop_criteria = _stop_by_depth if by_depth else _stop_by_point_number
    tree = QuadTree(coords, stop_criteria=stop_criteria)

    # write output csv file
    centroids = {}
    grid_labels = {key: None for key in list(dataframe.index)}
    tree.assign_labels(grid_labels, centroids, [])
    label_values = [grid_labels[key] for key in dataframe.index]
    dataframe[settings.CSV.OUTPUT.ACTUAL_GRID] = label_values
    output_dataframe = \
        dataframe.loc[:, (settings.CSV.INPUT.TEXT, 
                          settings.CSV.INPUT.LATITUDE, 
                          settings.CSV.INPUT.LONGITUDE, 
                          settings.CSV.OUTPUT.ACTUAL_GRID)]

    output_dataframe = output_dataframe.rename(index=str, columns={
        settings.CSV.INPUT.LATITUDE: settings.CSV.OUTPUT.ACTUAL_LATITUDE,
        settings.CSV.INPUT.LONGITUDE: settings.CSV.OUTPUT.ACTUAL_LONGITUDE
    })
    output_dataframe.to_csv(output_filename, sep='\t', header=True)
    return tree, centroids


if __name__ == "__main__":
    raise RuntimeError("This module is not supposed to be called.")
