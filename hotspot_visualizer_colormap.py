"""
Colormap visualization for mesh topology.

Supports an arbitrary NxN mesh with any traffic pattern.

Use parse_data to first parse a trace file produced by garnet and dump data into a .pkl file.
The filename at the beginning of main can be changed to visualize a different sim.

Hotspots are visualized by a color interpolation of router activity to colormap JET seen here
https://docs.opencv.org/3.4.12/d3/d50/group__imgproc__colormap.html
Color coding is absolute - for instance the darkest red indicates incoming flits at every port every cycles.

Each displayed column represents a single router, and is in ascending order from left to right
according to the router labeling used in Garnet.

This is a consolidation of the mesh visualizer code into a single picture. From top to bottom,
each column shows the color interpolated router activity for the chosen window size across all 
windows. For N simulation cycles and window size M, there are N-M+1 individual color strips
concatenated together in a column. This can be thought of as a 1D convolution for each column.

This visualization method does better to visualize how router activtity changes over the course
of a simulation, and allows for pattern recogonition and comparison of activity
variance across different traffic patterns for the same topology by the user.

Alan Kittel
ECE 6115
4/16/2021
"""

from hotspot_visualizer_mesh import create_heat_maps
from parse_data import parseData, load
from hotspot_functions import create_colormap, trackbar_nothing

import cv2 as cv
import numpy as np

def main():
    # change these for the current .csv file
    filename = r'data\XY_Mesh_4x4_UniformRandom_50.pkl'

    window_size = 100

    heat_map, _, _, _ = load(filename)
    heat_map /= 4.0

    sim_cycles = heat_map.shape[0]
    color_map = create_colormap(heat_map, window_size)

    cv.namedWindow('Colormap', cv.WINDOW_NORMAL)
    cv.resizeWindow('Colormap', 1000, 1000)
    cv.createTrackbar('Window Size', 'Colormap', 100, min(sim_cycles//10 - 1, 500), trackbar_nothing)

    window_size = cv.getTrackbarPos('Window Size', 'Colormap')

    old_window_size = window_size

    while(1):
        cv.imshow('Colormap', color_map)
        
        k = cv.waitKey(1) & 0xFF # wait for 1ms
        if k == 27: # hit escape to end the program
            break
        
        window_size = cv.getTrackbarPos('Window Size', 'Colormap')

        if window_size == 0:
            window_size = 1
            cv.setTrackbarPos('Window Size', 'Colormap', window_size)

        if window_size != old_window_size:
            color_map = create_colormap(heat_map, window_size)
        
        old_window_size = window_size

if __name__ == "__main__":
    main()