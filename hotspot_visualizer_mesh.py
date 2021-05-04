"""
Hotspot visualizer for mesh topology.

Supports an arbitrary NxN mesh with any traffic pattern.

Use parse_data to first parse a trace file produced by garnet and dump data into a .pkl file.
The filename at the beginning of main can be changed to visualize a different sim.

Hotspots are visualized by a color interpolation of router activity to colormap JET seen here
https://docs.opencv.org/3.4.12/d3/d50/group__imgproc__colormap.html
Color coding is absolute - for instance the darkest red indicates incoming flits at every port every cycles.

The displayed GUI window has trackbars for several configurable options:
Window size -> Changes the number of cycles to average router activity over when interpolating a color.
               Higher window sizes give a better representation of general router activity, while
               smaller windows capture more of router activity changing throughout a sim, but
               are more susceptible to noise.
Window offset -> cycle offset of the chosen window size, this is like sliding the window over the sim
                 and is mathematically equivalent to a convolution.
Most active router -> draws an X through the top N most active routers
Toggle router/Port view -> changes view between color coding entire routers, or individual ports

Alan Kittel
ECE 6115
4/16/2021
"""

import sys
import cv2 as cv
import numpy as np
from parse_data import parseData, load

from hotspot_functions import create_heat_maps, heat_map_window, trackbar_nothing

num_ports = 4 # 2D mesh, 4 ports - excluding local port

def draw_mesh(heat_map, topology_info, n, router_display):
    """ Creates an image of a mesh network, with routers color coded.

        Inputs:
            heat_map - array of shape (sim_cycles//window_size, num_routers) representing the average arrival rate of
                       flits at each router for each window. Normalized between 0 (no flits) and 1 (flits arrive at every port every cycle).
            topology_info - list of topology information
        Outputs:
            img - ~1000x1000 image with the drawn mesh topology, to be displayed in the openCV GUI
    """

    img_size = 1000 # set img size as 1000x1000
    num_rows = topology_info[1]
    router_width = img_size//2//num_rows # pixel width for a drawn router

    img = np.full((img_size+router_width,img_size+router_width,3), 255, np.uint8)

    if router_display == 0:
        # apply JET colormap to heat map to assign colors to routers
        intensities = np.flip(np.array(heat_map*255, dtype=np.uint8).reshape(num_rows,num_rows), axis=0).flatten()
        colors = np.array(cv.applyColorMap(intensities, cv.COLORMAP_JET).reshape(num_rows,num_rows,3), dtype=float)
        most_active = np.argsort(-intensities, kind='stable')[0:n]

        for i in range(num_rows):
            for j in range(num_rows):
                # draw routers w/ interpolated color
                top_left_x = router_width + router_width*i*2
                top_left_y = router_width + router_width*j*2
                cv.rectangle(img, (top_left_x, top_left_y), (top_left_x+router_width, top_left_y+router_width), tuple(colors[j][i][:]), 2)

                # draw X through routers among the top n most active
                if np.any(j*num_rows + i == most_active):
                    cv.line(img, (top_left_x, top_left_y), (top_left_x+router_width, top_left_y+router_width), tuple(colors[j][i][:]), 2)
                    cv.line(img, (top_left_x, top_left_y+router_width), (top_left_x+router_width, top_left_y), tuple(colors[j][i][:]), 2)
    else: # router_display = 1
        # assign color to each port
        intensities = np.flip(np.array(heat_map*255, dtype=np.uint8).reshape(num_rows*num_rows,num_ports).reshape(num_rows,num_rows,num_ports), axis=0).flatten()
        colors_ports = np.array(cv.applyColorMap(intensities, cv.COLORMAP_JET).reshape(num_rows,num_rows,num_ports,3), dtype=float)
        router_intensities = np.array(np.sum(intensities.reshape(num_rows*num_rows, num_ports), axis=1).flatten()/num_ports, dtype=np.uint8)

        # reformat condensed heat map into one for the routers
        colors_routers = np.array(cv.applyColorMap(router_intensities, cv.COLORMAP_JET).reshape(num_rows,num_rows,3), dtype=float)

        # most active routers
        most_active = np.argsort(-router_intensities, kind='stable')[0:n]

        for i in range(num_rows):
            for j in range(num_rows):
                top_left_x = router_width + router_width*i*2
                top_left_y = router_width + router_width*j*2
                for k in range(num_ports):
                    # draw ports w/ interpolated color. For now each port is hardcoded, so it's a bit messy.
                    if k == 0: # south port
                        cv.line(img, (top_left_x, top_left_y), (top_left_x+router_width, top_left_y), tuple(colors_ports[j][i][k][:]), 2)
                    elif k == 1: # east port
                        cv.line(img, (top_left_x+router_width, top_left_y), (top_left_x+router_width, top_left_y+router_width), tuple(colors_ports[j][i][k][:]), 2)
                    elif k == 2: # north port
                        cv.line(img, (top_left_x+router_width, top_left_y+router_width), (top_left_x, top_left_y+router_width), tuple(colors_ports[j][i][k][:]), 2)
                    else: # west port
                        cv.line(img, (top_left_x, top_left_y+router_width), (top_left_x, top_left_y), tuple(colors_ports[j][i][k][:]), 2)

                    # draw X through routers among the top n most active
                    if np.any(j*num_rows + i == most_active):
                        cv.line(img, (top_left_x, top_left_y), (top_left_x+router_width, top_left_y+router_width), tuple(colors_routers[j][i][:]), 2)
                        cv.line(img, (top_left_x, top_left_y+router_width), (top_left_x+router_width, top_left_y), tuple(colors_routers[j][i][:]), 2)

    # draw links as black lines
    # TODO add option to interpolate color for links as well
    for i in range(num_rows):
        for j in range(num_rows):
            top_left_x = router_width + router_width*i*2
            top_left_y = router_width + router_width*j*2

            if j != num_rows - 1: # North-South links
                cv.line(img, (top_left_x+router_width//2, top_left_y+router_width), (top_left_x+router_width//2, top_left_y+router_width*2), (0,0,0), 2)
            if i != num_rows - 1: # East-West links
                cv.line(img, (top_left_x+router_width, top_left_y+router_width//2), (top_left_x+router_width*2, top_left_y+router_width//2), (0,0,0), 2)
    return img

def main(filename, sim_length):
    """ main function """

    # change these for the current .csv file
    #filename = r'data/uniformrandom_90.pkl'
    #filename = r'data/bit_compl_deadlock.pkl'

    #TODO don't hardcode sim_cycles, output to .csv file in garnet
    #sim_cycles = 10000

    sim_cycles = int(sim_length)

    # initial window size and offset for display
    window_size = 1000
    window_offset = 1000

    heat_map_routers, heat_map_ports, topology_info, _ = load(filename)
    heat_map_routers /= 4.0

    # compute an initial heat map
    heat_map_routers_window = heat_map_window(heat_map_routers, window_size, window_offset, 0)
    heat_map_ports_window = heat_map_window(heat_map_ports, window_size, window_offset, 0)

    # create the GUI window and trackbars
    cv.namedWindow('Heatmap', cv.WINDOW_NORMAL)
    cv.resizeWindow('Heatmap', 1000, 1000) # default size is 1000x1000px, but it is resizable
    cv.createTrackbar('Window Size', 'Heatmap', window_size, sim_cycles, trackbar_nothing)
    cv.createTrackbar('Window Offset', 'Heatmap', window_offset, sim_cycles-1, trackbar_nothing)
    cv.createTrackbar('Most Active Routers', 'Heatmap', 0, topology_info[1]**2, trackbar_nothing)
    cv.createTrackbar('Toggle Router/Port View', 'Heatmap', 0, 1, trackbar_nothing)
    cv.createTrackbar('Toggle normalize for average flits', 'Heatmap', 0, 1, trackbar_nothing)

    # set tracked variables to their initial values for the first pass through below loop
    old_window_offset = cv.getTrackbarPos('Window Offset', 'Heatmap')
    old_window_size = cv.getTrackbarPos('Window Size', 'Heatmap')
    most_active = cv.getTrackbarPos('Most Active Routers', 'Heatmap')
    router_display = cv.getTrackbarPos('Toggle Router/Port View','Heatmap')
    normalize_opt = cv.getTrackbarPos('Toggle normalize for average flits','Heatmap')
    #print(normalize_opt)

    while(1):
        heat_map_routers_window = heat_map_window(heat_map_routers, window_size, window_offset, normalize_opt)
        heat_map_ports_window = heat_map_window(heat_map_ports, window_size, window_offset, normalize_opt)
        # draw the mesh, either with routers color coded, or the ports
        if router_display == 0:
            cv.imshow('Heatmap', draw_mesh(heat_map_routers_window, topology_info, most_active, router_display))
        else:
            cv.imshow('Heatmap', draw_mesh(heat_map_ports_window, topology_info, most_active, router_display))

        k = cv.waitKey(1) & 0xFF # wait for 1ms
        if k == 27: # hit escape to end the program
            break

        # get current positions of trackbars
        window_offset = cv.getTrackbarPos('Window Offset', 'Heatmap')
        window_size = cv.getTrackbarPos('Window Size', 'Heatmap')
        most_active = cv.getTrackbarPos('Most Active Routers', 'Heatmap')
        router_display = cv.getTrackbarPos('Toggle Router/Port View','Heatmap')
        normalize_opt = cv.getTrackbarPos('Toggle normalize for average flits','Heatmap')

        if window_size == 0:
            window_size = 1
            cv.setTrackbarPos('Window Size', 'Heatmap', window_size)

        # no way I could find to change the bound of a trackbar once you create it.
        # manually set the position to the maximum possible window_indedx given
        # the current window_size if the user slides it over the maximum.
        if window_offset > sim_cycles - window_size:
            window_offset = sim_cycles - window_size
            cv.setTrackbarPos('Window Offset', 'Heatmap', window_offset)

        # the user changes the window_size, so need to recompute the heat map.
        if window_size != old_window_size or window_offset != old_window_offset:
            heat_map_routers_window = heat_map_window(heat_map_routers, window_size, window_offset, normalize_opt)
            heat_map_ports_window = heat_map_window(heat_map_ports, window_size, window_offset, normalize_opt)

        old_window_size = window_size
        old_window_offset = window_offset

    cv.destroyAllWindows()

if __name__ == "__main__":
    if(len(sys.argv) < 3):
        print("usage: python hotspot_visualizer_mesh.py pklFile sim_length");
    else:
        main(sys.argv[1], sys.argv[2])
