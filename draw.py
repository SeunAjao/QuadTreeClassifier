"""
Module for drawing tweets and quadtree on the map.
"""

import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt
import settings


def draw_bounds(bmap, tree, max_depth):
    """
    Draw a proper bounding rectangle of quadtree on basemap
    """

    bounds_lons = [tree.bounds[1][0], tree.bounds[1][0],
                   tree.bounds[1][1], tree.bounds[1][1], tree.bounds[1][0]]
    bounds_lats = [tree.bounds[0][0], tree.bounds[0][1],
                   tree.bounds[0][1], tree.bounds[0][0], tree.bounds[0][0]]
    bmap.plot(bounds_lons, bounds_lats, linewidth=0.2, color='b', zorder=3)

    if not tree.children or tree.depth >= max_depth:
        return
    else:
        for child in tree.children:
            draw_bounds(bmap, child, max_depth)


def draw_result(labels, tree, max_depth, map_bounds):
    """
    Draw a geographic map and the resulting tree with tweets on it.
    """

    bmap = Basemap(projection='cyl', llcrnrlon=map_bounds[1][0], llcrnrlat=map_bounds[0][0],
                   urcrnrlon=map_bounds[1][1], urcrnrlat=map_bounds[0][1],
                   resolution='l')

    # draw continents, countries, states
    bmap.drawcoastlines()
    bmap.drawcountries(linewidth=1.5)
    bmap.fillcontinents(
        color=settings.MAP.LAND_COLOR, lake_color=settings.MAP.WATER_COLOR)
    bmap.drawmapboundary(fill_color=settings.MAP.WATER_COLOR)

    y_test, y_pred = labels

    # draw tweet points
    dataframe = tree.coords.join(y_pred == y_test)
    test_df = dataframe.dropna()
    train_df = dataframe.ix[ dataframe[settings.CSV.OUTPUT.ACTUAL_GRID].isnull().nonzero() ]
    train_points = np.array([point for _, point in train_df.iterrows()])
    true_df = test_df.loc[test_df[settings.CSV.OUTPUT.ACTUAL_GRID] == True]
    false_df = test_df.loc[test_df[settings.CSV.OUTPUT.ACTUAL_GRID] == False]
    true_points = np.array([point for _, point in true_df.iterrows()])
    false_points = np.array([point for _, point in false_df.iterrows()])

    handle_0 = bmap.scatter(train_points[:, 1], train_points[:, 0],
                 marker='o', color='gray', zorder=2, s=0.5)
    handle_1 = bmap.scatter(false_points[:, 1], false_points[:, 0],
                 marker='o', color='r', zorder=2, s=1)
    handle_2 = bmap.scatter(true_points[:, 1], true_points[:, 0],
                 marker='o', color='g', zorder=2, s=1)
    

    plt.xlabel('')
    plt.ylabel('')
    plt.title('')
    plt.legend((handle_0, handle_1, handle_2), 
               ('Train points', 'False prediction', 'True prediction'))
    
    # draw quadtree regions layout
    draw_bounds(bmap, tree, max_depth)
