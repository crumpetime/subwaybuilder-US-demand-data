"""
TODOs:
- curl the files directly?
  - Example paths:
    - https://lehd.ces.census.gov/data/lodes/LODES8/fl/od/fl_od_main_JT01_2022.csv.gz
    - https://lehd.ces.census.gov/data/lodes/LODES8/fl/fl_xwalk.csv.gz
  - then gunzip
- use config file for bbox and list of state(s)
"""

import sys, os
import csv
import glob
import json
from multiprocessing import Pool
import numpy as np
import geopandas as gpd
import osmnx as ox
import networkx as nx

MAXPOPSIZE = 200

#states = glob.glob('./*')
#states = [s for s in states if os.path.isdir(s)]
#for state in states:
state = './md'
print("Processing", state, flush=True)
bbox = (-76.88, 38.993, -76.387, 39.431) #minlon, minlat, maxlon, maxlat

# Set up OSM graph
print("Initializing OSM drive network graph")
G = ox.graph_from_bbox(bbox, network_type='drive')#, simplify=False)
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)

print("Finding data files")
#fxwalk = glob.glob(os.path.join(state, '*_xwalk.csv'))
fjobs  = glob.glob(os.path.join(state, '*_main_JT01_*.csv'))
ftract = glob.glob(os.path.join(state, 'tl_????_??_tract.shp'))
#assert len(fxwalk) == 1, "More than 1 xwalk file found for "+state
assert len(fjobs)  == 1, "More than 1 JT01 file found for "+state
assert len(ftract)  == 1, "More than 1 tract file found for "+state
#fxwalk = fxwalk[0]
fjobs  = fjobs[0]
ftract = ftract[0]

# Read crosswalk file - xwalk - from LODES https://lehd.ces.census.gov/data/
# Not needed unless using zcta for higher-level aggregation...which might be necessary 
"""
print("Loading crosswalk data")
with open(fxwalk, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    xwalk_header = next(reader)
    xwalk_rows = np.array([row for row in reader], dtype=str)

xwalk_id = xwalk_rows[:,0].astype(int)
xwalk_trct = xwalk_rows[:,6].astype(int)
#xwalk_lat = xwalk_rows[:,-3].astype(float)
#xwalk_lon = xwalk_rows[:,-2].astype(float)

xwalk_keep = (xwalk_lon >= bbox[0]) * (xwalk_lat >= bbox[1]) * \
             (xwalk_lon <= bbox[2]) * (xwalk_lat >= bbox[3])

"""

# Read jobs file - JT01 - from LODES
print("Loading jobs data")
with open(fjobs, 'r') as foo:
    jobs_header = np.array(foo.readlines()[0].strip().split(','))
jobs = np.loadtxt(fjobs, delimiter=',', skiprows=1, dtype=int)
work = jobs[:,0]
home = jobs[:,1]
pops = jobs[:,2]
work_tracts = work.astype('<U11').astype(int)
home_tracts = home.astype('<U11').astype(int)

# Read census tracts data file - https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2022&layergroup=Census+Tracts
print("Loading census tract data")
tracts = gpd.read_file(ftract) 
tracts_ids = tracts.GEOID.to_numpy().astype(int)
# Change to an equal-area projection, then calculate centroid in this projection
tracts_proj = tracts.to_crs(epsg=2163)
tracts_proj_centroid = tracts_proj.geometry.centroid
# Convert back to lat/lon locations for the game map
tracts_latlon = tracts_proj_centroid.to_crs(epsg=4326)
tracts_proj_lat = tracts_latlon.geometry.y.to_numpy()
tracts_proj_lon = tracts_latlon.geometry.x.to_numpy()

# Remove any tracts not within the bbox
ikeep = (tracts_proj_lon >= bbox[0]) * (tracts_proj_lat >= bbox[1]) * \
        (tracts_proj_lon <= bbox[2]) * (tracts_proj_lat <= bbox[3])
tracts_ids = tracts_ids[ikeep]
tracts_proj_lat = tracts_proj_lat[ikeep]
tracts_proj_lon = tracts_proj_lon[ikeep]
itracts = np.arange(tracts_ids.size, dtype=int)

ikeep2 = np.array([w in tracts_ids for w in work_tracts]) * \
         np.array([h in tracts_ids for h in home_tracts])
pops = pops[ikeep2]
work_tracts = work_tracts[ikeep2]
home_tracts = home_tracts[ikeep2]

# Go through each tract - log number of pops and their workplace tract
demand_data = {'total' : 0}
for it,t in enumerate(tracts_ids):
    #if not it % 10:
    print("  Processing tract", it+1, "/", tracts_ids.size, flush=True, end='\r')
    demand_data[t] = {}
    demand_data[t]['total'] = 0
    itract = home_tracts == t
    work_locations = np.unique(work_tracts[itract])
    for w in work_locations:
        ipops = work_tracts[itract] == w
        demand_data[t][w] = pops[itract][ipops].sum()
        demand_data[t]['total'] += demand_data[t][w]
    demand_data['total'] += demand_data[t]['total']
print("")
# demand file consists of points, and pops

# point format:
#      "id": "2",
#      "location": [-78.9975, 43.863],
#      "jobs": 834,
#      "residents": 292,
#      "popIds": ["27528", "148", "390", "1", "32657", "5717", "5837", "13734", "14651"]

# pop format:
#      "id": "31582",
#      "size": 165,
#      "residenceId": "12053",
#      "jobId": "9405",
#      "drivingSeconds": 847,
#      "drivingDistance": 9435

demand = {"points" : [], "pops" : []}
for itract in itracts:
    #if not itract % 100:
    print("  Processing point", itract+1, "/", itracts.size, flush=True, end='\r')
    point = {
        "id" : str(itract+1),
        "location" : [float(tracts_proj_lon[itract]), float(tracts_proj_lat[itract])],
        "jobs" : 0,
        "residents" : 0,
        "popIds" : []
    }
    demand["points"].append(point)
print("")

def process_job(args):
    i, j, residence_id, home_node = args
    
    if j == 'total':
        return []  # skip
    
    job_id = itracts[tracts_ids == j][0]
    work_node = ox.nearest_nodes(G, Y=tracts_proj_lat[job_id], X=tracts_proj_lon[job_id])
    
    try:
        distance_in_meters = nx.shortest_path_length(G, home_node, work_node, weight='length')
        travel_time_in_seconds = nx.shortest_path_length(G, home_node, work_node, weight='travel_time')
    except:
        distance_in_meters = 0
        travel_time_in_seconds = 0
    
    niter = int(np.ceil(demand_data[i][j] / MAXPOPSIZE))
    pops = []
    for n in range(niter):
        pop = {
            "residenceId": str(residence_id + 1),
            "jobId": str(job_id + 1),
            "drivingSeconds": int(travel_time_in_seconds),
            "drivingDistance": int(np.ceil(distance_in_meters)),
        }
        if n < niter - 1:
            pop["size"] = MAXPOPSIZE
        else:
            pop["size"] = int(demand_data[i][j]) % MAXPOPSIZE
        pops.append((residence_id, job_id, pop))
    return pops


ipop = 1
home_ids = list(demand_data.keys())
home_ids.remove('total')

for ihome, i in enumerate(home_ids):
    print("  Processing home tract", ihome+1, "/", len(home_ids), flush=True, end='\r')
    residence_id = itracts[tracts_ids == i][0]
    home_node = ox.nearest_nodes(G, Y=tracts_proj_lat[residence_id], X=tracts_proj_lon[residence_id])
    
    # Prepare arguments for parallel jobs
    tasks = [(i, j, residence_id, home_node) for j in demand_data[i].keys()]
    
    with Pool() as pool:
        results = pool.map(process_job, tasks)
    
    # Flatten results and update demand
    for pops in results:
        for residence_id, job_id, pop in pops:
            pop["id"] = str(ipop)
            demand["pops"].append(pop)
            demand["points"][residence_id]["residents"] += pop["size"]
            demand["points"][residence_id]["popIds"].append(pop["id"])
            demand["points"][job_id]["jobs"] += pop["size"]
            demand["points"][job_id]["popIds"].append(pop["id"])
            ipop += 1
print("")

# Old, non-parallel code:
"""
ipop = 1
home_ids = list(demand_data.keys())
home_ids.remove('total')
for ihome, i in enumerate(home_ids):
    #if not ihome % 100:
    print("  Processing pop", ihome+1, "/", len(home_ids), flush=True, end='\r')
    residence_id = itracts[tracts_ids == i][0]
    home_node = ox.nearest_nodes(G, Y=tracts_proj_lat[residence_id], X=tracts_proj_lon[residence_id])
    for j in demand_data[i].keys():
        if j == 'total':
            continue
        
        job_id = itracts[tracts_ids == j][0]
        # Calculate driving distance and time
        work_node = ox.nearest_nodes(G, Y=tracts_proj_lat[job_id], X=tracts_proj_lon[job_id])
        try:
            distance_in_meters = nx.shortest_path_length(G, home_node, work_node, weight='length')
            travel_time_in_seconds = nx.shortest_path_length(G, home_node, work_node, weight='travel_time')
        except:
            distance_in_meters = 0
            travel_time_in_seconds = 0
        # Limit pops to max of 200
        niter = int(np.ceil(demand_data[i][j] / MAXPOPSIZE))
        for n in range(niter):
            pop = {
                "id": str(ipop),
                #"size": int(demand_data[i][j]), # Handled below
                "residenceId": str(residence_id + 1),
                "jobId": str(job_id + 1),
                "drivingSeconds": int(travel_time_in_seconds),
                "drivingDistance": int(np.ceil(distance_in_meters))
            }
            if n < niter - 1:
                # More than MAXPOPSIZE pops remain - cap at MAXPOPSIZE
                pop["size"] = MAXPOPSIZE
            else:
                # Less than MAXPOPSIZE remains - put all into this pop
                pop["size"] = int(demand_data[i][j]) % MAXPOPSIZE
            demand["pops"].append(pop)
            demand["points"][residence_id]["residents"] += pop["size"]
            demand["points"][residence_id]["popIds"].append(pop["id"])
            demand["points"][job_id]["jobs"] += pop["size"]
            demand["points"][job_id]["popIds"].append(pop["id"])
            ipop += 1
print("")
"""

# Save out demand file
filename = os.path.join(state, 'demand_data.json')
with open(filename, "w") as json_file:
    json.dump(demand, json_file, indent=4)


