# Garnet Hotspot Detector

This is the repository for garnet hotspot detection.

Alan Kittel, alan.kittel@verizon.net

Albert Cho, albertycho90@gmail.com

## Setup
This is intended to be run alongside Garnet2.0. Copy garnet2.0/ to /src/mem/ruby/network in your gem5 repository and replace the contents. Copy garnet_synth_traffic.py to /configs/example and replace the file there.

The following Python packages will need to be installed, with known working versions in parentheses. A known working verison of Python is 64-bit 3.9.1.
* numpy (1.20.2)
* opencv (4.5.1)
* scipy (1.6.2)

### Hotspot detection output within Garnet
We enable garnet hotpost tracking output (non visualization part) by default. \
We take additional options - "hotspot-cutoff" for number of hotspots to track and "hotspot_period" for sampling period. \
Tracking output will be written to hotspotStatFile.txt every run. 

Content of hotspotStatFile.txt will look like this:\
at cycle 1000\
Router_id, flit_count\
Router_id, flit_count\
at cycle 2000\
Router_id, flit_count\
Router_id, flit_count

for hotspot-cutoff==2 and hotspot-period==1000

### Hotspot detection output for visualization
To visualize a new simulation, run a Garnet simulation from the command line as done normally. At the base gem5/ folder, a .csv file called "LoupeFile.csv" will be produced. Copy this file to the /traceFiles folder in this repository, and call load_and_save in parse_data.py with this file's filename as input, the desired location and name of the output file (suggested to put it under the data/ folder), and a string of the topology type. This will parse the data and dump the router and port activity arrays into a .pkl file. For future calls to hotspot_visualizer_colormap.py and hotspot_visualizer_mesh, this .pkl file can be used as input alongside the load function in parse_data.py. Parsing the data from the .csv file long sims can take some time, but loading from the .pkl file is nearly instantaneous. 

## Code Overview
### Running the Code
Currently, our code only supports mesh topologies. Minor adjustments will need to be made to generalize the code for other topolgies, and the topology drawing will need to be manually created for each new topology.

To see the topology visualization, run hotspot_visualizer_mesh.py as a script, with no arguments. Change the filename defined by the "file" variable at the beginning of main to visualize a different simulation.

To see the colormap visualization, the process is the same as for topology visualization, but with the hotspot_visualizer_colormap.py file.

### Hotspot visualizer mesh
<img src="pictures\8x8_topology_visualization_router.PNG" alt="topology_router" width="400"/>

The topology visualization will draw the topology on screen, with routers as squares, connected by straight lines as the links. Above is an example GUI window popup from running hotspot_visualizer_mesh. The trackbars are user adjustable with click and dragging. 

Routers or ports are colored by an interpolation on the JET colormap as seen below.

<img src="pictures\colormap_jet.PNG" alt="jet" width="300"/>

where 0 represents no flits arriving at the router or port for every cycle over all cycles in the window, and 1 represents the maximum number of flits arriving at the router or port for every cycle over all cycles in the window. For routers, there can be at most 4 flits arrving (1 for each of the 4 ports in a mesh), and for ports, at most 1 flit.

We define activity as the quantity of flits arriving at a router or port at each cycle. The window size parameter changes how many cycles to take an average of activity over. In finding hotspots, it is not helpful to examine the network cycle-by-cycle, as activity is highly volatile at this granularity. Instead, we take an average of a number of cycles to get a better picture of relative activity among routers. Setting the window size to 1 will examine activity cycle-by-cycle. We find window sizes between 100 and 1000 to be the best. In this range, the window size is large enough than random spikes or dips in flits arriving at routers for a cycle or two is averaged out, but not so large that general changes in router activity over the simulation are avraged out.

The window offset specifies how many cycles to offset the averaging window from cycle 1 by. This changes where in the simulation the hotspots are visualizaed.

The most active parameters trackbar will draw X's through the most active routers.

The toggle router/port view will change the hotspot visualization to show port activity instead.

<img src="pictures\8x8_topology_visualization_port.PNG" alt="topology_port" width="400"/>

### Colormap visualization

<img src="pictures\8x8_colormap_visualization.gif" alt="colormap" width="800"/>

Above is an example colormap visualization output. Each dinstinctly colored column represents a single router, ordered by router ID left to right. The above is for an 8x8 mesh, so there are 64 distinct columns.

The Y axis (vertical) represents the cycle number, starting at 1 at the top of the picture. For each router, the colors shown are the color interpolation of activity at that router, as done before with the topology visualizer, throughout the entire simulation with the specified window size.

For example, with window size 100, and this simulation being 10,000 cycles long, the activity over the following cycles is color interpolated:
* 1-100
* 2-101
* 3-102

...
* 9,899-9,998
* 9,900-9,999
* 9,901-10,000

This equates to 9,901 separate color interpolations concatenated into a single column and visualized. In the more general case, the output size is M - N + 1, where M is the simulation length and N is the window size, in cycles. 

This operation is equivalent to a 1D convolution, where the normalized cycle-by-cycle arriving flit count for a router is colvolved with a unit pulse signal of length N. This is how it is calculated in the code (see heat_map_window all in hotspot_functions.py).

## Runcmd Example
Build and run Garnet to generate LoupeTraceFile.csv\
Run parse_data.py with the .csv file to generate .pkl file\
Run hotspot_visualizer (Mehs or colorview) with the .pkl file\
.pkl file is the output from parse_data, and input to hotspot_visualizer\
Script are split into 2 parts because parse_data could take long, and we want to avoid running it everytime we want to visualize the same load

ex)
./build/Garnet_standalone/gem5.opt configs/example/garnet_synth_traffic.py --network=garnet2.0 --num-cpus=64 --num-dirs=64 --topology=Mesh --mesh-rows=8 --sim-cycles=10000 --inj-vnet=0 --routing-algorithm=random_oblivious --vcs-per-vnet=16 --injectionrate=0.90 --garnet-deadlock-threshold=1000 --synthetic=transpose\
python parse_data.py LoupeTraceFile.csv trace.pkl\
python hotspot_visualizer_mesh.py trace.pkl\
\
(consult SETUP section of this document for required libraries)


## Sample Traces
Sample .csv files and .pkl files can be found in sample_data branch of this git repository. \
url:\
https://github.com/albertycho/garnet_hotspot/tree/sample_data


## Future Work
Some improvements that could be made
* Output and parse the topology type from Garnet rather than hard coding it.
* Output and parse the port quantity of routers from Garnet rather than hard coding it.
* Integrate this visualization tool with Garnet to automatically parse the simulation output and visualize it, instead of having to run it externally.
* Implement topology visualization for other topologies, or if time not permitting, generalize it to work for any topology (i.e. visualize the routers in a deterministically ordered grid regardless of the topology, and don't draw the links).
* Add axes labeling to the colormap visualization.
