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