import numpy as np
from scipy.signal import fftconvolve
import cv2 as cv

num_ports = 4 # 2D mesh, 4 ports - excluding local port

def create_heat_maps(cycle_data, topology_info):
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

    port_directions = ["North", "East", "South", "West"]
    sim_cycles = cycle_data[-1]["cycle"]
    routers = cycle_data[cycle_data["unit"] == "InUnit"]

    heat_map_routers = np.zeros((sim_cycles, topology_info[0]))
    heat_map_ports = np.zeros((sim_cycles, topology_info[0]*num_ports))

    for i in range(topology_info[0]):
        # calculate heat map for routers
        router_cycles = routers[np.logical_and(routers["unit_ID"] == i, routers["direction"] != "Local")]["cycle"] - 1
        num_flits = np.bincount(router_cycles)
        num_flits = num_flits[np.nonzero(num_flits)]
        heat_map_routers[np.unique(router_cycles),i] = num_flits

        # calculate heat map for ports
        for j in range(len(port_directions)):
            dir_cycles = routers[np.logical_and(routers["unit_ID"] == i, routers["direction"] == port_directions[j])]["cycle"] - 1
            heat_map_ports[dir_cycles,(i*num_ports)+j] = 1
    return heat_map_routers, heat_map_ports

def heat_map_window(heat_map, time_window, window_offset, normalize_opt):
    """
    Computes the average flit activity from a heat map over a window.

    Inputs:
        heat_map - heat map for routers, ports, or links
        time_window - number of cycles to average over
        window_offset - cycle offset for position of window within the heat_map array
    Outputs:
        average flit activity for each router, port, or link over the window
    """
    t_list=np.sum(heat_map[window_offset:window_offset+time_window], axis=0)
    #t_array=np.sum(heat_map[window_offset:window_offset+time_window], axis=0)

    #t_list=t_array.tolist()

    max_flit=1.0
    for i in range(len(t_list)):
        if t_list[i]>max_flit:
            max_flit=t_list[i]

    #TODO: we want to have both options
    if normalize_opt==0:
        return np.sum(heat_map[window_offset:window_offset+time_window,:], axis=0) / time_window
    else:
        return np.sum(heat_map[window_offset:window_offset+time_window,:], axis=0) / max_flit

def heat_map_window_all(heat_map, time_window):
    """
    Computes the average flit activity from a heat map over a sliding window.

    Inputs:
        heat_map - heat map for routers, ports, or links
        time_window - number of cycles to average over
    Outputs:
        average flit activity for each router, port, or link over the sliding window
    """

    mask = np.ones((time_window, heat_map.shape[1]))

    heat_map_all = fftconvolve(heat_map, mask, axes=0, mode='valid') / time_window

    return heat_map_all

def create_colormap(heat_map, window_size=100):
    """
    Computes the colormap as an interpolation of router activity to the colormap JET from the
    heat map generated from heat_map_window_all.

    Inputs:
        heat_map - heat map for routers, ports, or links
        window_size - number of cycles to average over
    Outputs:
        image with interpolated color representing router activity
    """

    num_routers = heat_map.shape[1]
    scale=50

    heat_map_all = cv.resize(heat_map_window_all(heat_map, window_size), (num_routers, 1000))
    heat_map_all_len = heat_map_all.shape[0]

    color_map = cv.applyColorMap(np.array(heat_map_all*255, dtype=np.uint8), cv.COLORMAP_JET)
    color_map = (np.tile(color_map[:,:,None,:], (1,1,scale,1))).reshape(heat_map_all_len,scale*num_routers,3)

    return color_map

def trackbar_nothing(val):
    """ Trackbars require a callback function. Feature not used, so use this
        function which does nothing.
    """
    pass
