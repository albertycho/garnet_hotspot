import cv2 as cv
import numpy as np
from parse_data import parseData

def create_heat_map(cycle_data, topology_info):
    """
    Parses the cycle data read from a .csv file into labeling of number of flits
    arriving at each router each cycle.

    Inputs:
        cycle_data - cycle_data output by parseData
        topology_info - topology information output by parseData
    Outputs:
        heat_map - (sim_cycles, num_routers) array representing the number of
                   flits arriving at each router at each cycle.
                   For mesh, there can be at most 5 each cycle (4 links + local port)
    """

    sim_cycles = cycle_data[-1]["cycle"]
    routers = cycle_data[cycle_data["unit"] == "InUnit"]
    #links = cycle_data[cycle_data["unit"] == "Link"]

    heat_map = np.zeros((sim_cycles, topology_info[0]))
    for i in range(topology_info[0]):
        router_cycles = routers[routers["unit_ID"] == i]["cycle"] - 1
        num_flits = np.bincount(router_cycles)
        num_flits = num_flits[np.nonzero(num_flits)]
        heat_map[np.unique(router_cycles),i] = num_flits
    return heat_map

def condense_heat_map(heat_map, time_window):
    """
    Condenses the heat map produced by create_heat_map from sim_cycles rows down to
    sim_cycles//time_window rows, averaging the values across each window.

    Average values are then truncated down to be in range [0-1] for display in draw_mesh.

    Inputs:
        heat_map - heat map produced by create_heat_map
        time_window - integer for number of cycles to average the heat map over.
                      Higher time_windows will produce a condensed heat map with fewer rows.
    Outputs:
        condense_heat_map - condensed heat map, with values truncated between 0 and 1 for display
    """

    num_windows = heat_map.shape[0]//time_window
    condense_heat_map = np.zeros((num_windows, heat_map.shape[1]))
    for i in range(num_windows):
        condense_heat_map[i,:] = np.sum(heat_map[i*time_window:i*time_window+time_window,:], axis=0)/time_window
    return condense_heat_map / 5

def draw_mesh(heat_map, topology_info, n=0):
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
    intensities = np.array(heat_map*255, dtype=np.uint8)

    # apply JET colormap to heat map to assign colors to routers
    colors = np.array(cv.applyColorMap(intensities, cv.COLORMAP_JET).reshape(num_rows,num_rows,3), dtype=float)

    most_active = np.argsort(-intensities, kind='stable')[0:n]

    for i in range(num_rows):
        for j in range(num_rows):
            # draw routers w/ interpolated color
            top_left_x = router_width + router_width*i*2
            top_left_y = router_width + router_width*j*2
            cv.rectangle(img, (top_left_x, top_left_y), (top_left_x+router_width, top_left_y+router_width), tuple(colors[j][i][:]), 2)
            if np.any(j*num_rows + i == most_active):
                cv.line(img, (top_left_x, top_left_y), (top_left_x+router_width, top_left_y+router_width), tuple(colors[j][i][:]), 2)
                cv.line(img, (top_left_x, top_left_y+router_width), (top_left_x+router_width, top_left_y), tuple(colors[j][i][:]), 2)

            # draw links as black lines
            # TODO add option to interpolate color for links as well
            if j==0 or j%(num_rows - 1) != 0:
                cv.line(img, (top_left_x+router_width//2, top_left_y+router_width), (top_left_x+router_width//2, top_left_y+router_width*2), (0,0,0), 2)
            if i==0 or i%(num_rows - 1) != 0:
                cv.line(img, (top_left_x+router_width, top_left_y+router_width//2), (top_left_x+router_width*2, top_left_y+router_width//2), (0,0,0), 2)
    return img

def trackbar_nothing(val):
    """ Trackbars require a callback function. Feature not used, so use this
        function which does nothing.
    """
    pass

def main():
    """ main function """

    # change these for the current .csv file
    filename = r'traceFiles\XY_Mesh_4x4_UniformRandom_50.csv'
    topology = "mesh"
    window_size = 100
    sim_cycles = 10000

    # parse the .csv file
    cycle_data, _, topology_info = parseData(filename, topology)

    # compute an initial heat map
    heat_map = create_heat_map(cycle_data, topology_info)
    heat_map_condense = condense_heat_map(heat_map, window_size)

    # create the GUI window and trackbars
    cv.namedWindow('Heatmap', cv.WINDOW_NORMAL)
    cv.resizeWindow('Heatmap',800,800) # default size is 800x800px, but it is resizable
    cv.createTrackbar('Window Size','Heatmap',window_size-1,int(sim_cycles//10-1),trackbar_nothing)
    cv.createTrackbar('Window Index','Heatmap',0,int((sim_cycles//10-1)//10),trackbar_nothing)
    cv.createTrackbar('Most Active Routers','Heatmap',0,topology_info[1]**2,trackbar_nothing)

    # set tracked variables to their initial values for the first pass through below loop
    window_index = cv.getTrackbarPos('Window Index', 'Heatmap')
    old_window_size = cv.getTrackbarPos('Window Size', 'Heatmap') + 1
    most_active = cv.getTrackbarPos('Most Active Routers', 'Heatmap')

    while(1):
        # draw the mesh
        cv.imshow('Heatmap', draw_mesh(heat_map_condense[window_index,:], topology_info, most_active))

        k = cv.waitKey(1) & 0xFF # wait for 1ms
        if k == 27: # hit escape to end the program
            break

        # get current positions of trackbars
        window_index = cv.getTrackbarPos('Window Index', 'Heatmap')
        window_size = cv.getTrackbarPos('Window Size', 'Heatmap') + 1
        most_active = cv.getTrackbarPos('Most Active Routers', 'Heatmap')

        # no way I could find to change the bound of a trackbar once you create it.
        # manually set the position to the maximum possible window_indedx given
        # the current window_size if the user slides it over the maximum.
        if window_index > int(sim_cycles/window_size-1):
            window_index = int(sim_cycles/window_size-1)
            cv.setTrackbarPos('Window Index', 'Heatmap', window_index)

        # the user changes the window_size, so need to recompute the heat map.
        if window_size != old_window_size:
            heat_map_condense = condense_heat_map(heat_map, window_size)

        old_window_size = window_size

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()