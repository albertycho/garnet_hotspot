"""
Parses the .csv file produced by running a Garnet simulation.

Alan Kittel
4/13/2021
Version 1.0: Initial implementation
"""

import numpy as np
import pickle

from hotspot_functions import create_heat_maps

# data type for the cycle data
dtype = [
    ("cycle", "i4"),
    ("unit", "<U255"),
    ("unit_ID", "i4"),
    ("direction", "<U255"),
    ("flit_ID", "i4"),
    ("flit_type", "i4"),
    ("flit_vnet", "i4"),
    ("flit_vc", "i4"),
    ("flit_src", "i4"),
    ("flit_dst", "i4"),
    ("flit_enqueue", "i4"),
]

def parseData(filename, topology):
    """
    Parses the .csv file produced by running a Garnet simulation.
    For now, only meshes are supported.

    Inputs:
        filename - the relative path to the .csv file
        topology - string identifying the topology type, for only "mesh" is supported
    Outputs:
        cycle_data - cycle-by-cycle data for flit presence in buffers and on links
        total_router_activity - list of total flits passing through each router for whole simulation
        topology_info - extracted topology information, varies by topology type
                        For mesh, this is [num_routers, num_rows, vcs_per_vnet, m_virtual_networks]
    """

    # open trace file and read lines
    traceFile = open(filename)
    lines = traceFile.readlines()
    lines = [line.split(',')[0:-1] for line in lines]

    # get topology information
    topology_info = np.array([int(lines[0][i]) for i in range(len(lines[0]))])

    # line number of the end of sim printing
    end_sim_idx = np.argwhere([line[0] == "End of sim" for line in lines])[0][0]

    # parse the cycle data into a strucutred numpy array
    cycle_data = lines[1:end_sim_idx]
    for i in range(len(cycle_data)):
        cycle_data[i] = (int(cycle_data[i][0]), cycle_data[i][1], int(cycle_data[i][2]), cycle_data[i][3], \
            int(cycle_data[i][5]), int(cycle_data[i][6]), int(cycle_data[i][7]), int(cycle_data[i][8]), \
            int(cycle_data[i][9]), int(cycle_data[i][10]), int(cycle_data[i][11]))
    cycle_data = np.array(cycle_data, dtype=dtype)

    # parse the total router activity
    end_sim = lines[end_sim_idx+2:]
    total_router_activity = np.array([int(router[1]) for router in end_sim])
    
    return cycle_data, total_router_activity, topology_info

def load_and_save(loadfile, savefile, topology):
    save_data = {}

    cycle_data, router_activity, topology_info = parseData(loadfile, topology)

    heat_map_routers, heat_map_ports = create_heat_maps(cycle_data, topology_info)

    save_data["heat_map_routers"] = heat_map_routers
    save_data["heat_map_ports"] = heat_map_ports
    save_data["topology_info"] = topology_info
    save_data["router_activity"] = router_activity

    with open(savefile, 'wb') as f:
        pickle.dump(save_data, f)

def load(loadfile):
    with open(loadfile, 'rb') as f:
        data = pickle.load(f)
    
    return data["heat_map_routers"], data["heat_map_ports"], data["topology_info"], data["router_activity"]