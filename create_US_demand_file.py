"""
This script is meant to be run like
    ./create_US_demand_file.py Rochester.json

The input JSON file must have the following fields defined:
    city : string, the city you're modeling.  
           Example: "Rochester"
    airport : list of strings, IATA codes for the local airport 
                               Note: The first listed airport is used here to uniquely identify a city.  
              Example: ["ROC"]
    states : string or list of strings, two-letter code for the state(s) your map covers.  
            Example: "ny"
            Example: ["md", "dc", "va"]
    year : int, the year you want to use for the LODES data.  At the time of writing, it must be within 2002-2023.  
           Example: 2022
    bbox : list of ints, the [min_lon, min_lat, max_lon, max_lat] boundary for the city.  
           Example: [-77.8216, 43.0089, -77.399, 43.3117],
    cbd_bbox : list of ints, exactly like `bbox` except for the Central Business District.  
                             Can be used to reduce clustering for the downtown area.  
                             To disable, set this to null.
               Example: null

    HUMAN_READABLE : (optional) bool, determines whether to make the output demand_data.json file have indentation 
                                      structure for readability (true) or not to minimize file size (false).
                      Default: false
    MAX_WORKERS : (optional) int, sets the number of workers to use for parallel processing.
                   Default: None (total number of CPU threads)
    
    MAXPOPSIZE : int, maximum size any pop can be.  
                      Pops larger than this value are split into multiple pops to follow this setting.
                 Example: 200
    CALCULATE_ROUTES : bool, determines whether to calculate commuting routes.  
                             Recommended to set this to false when initially testing out boundaries and clustering.
                       Example: true
    SMALL_THRESHOLD : int, maximum pop size considered for agglomerative clustering.  
                           Pops smaller than this size will be merged with other nearby pops that work at the same location.
                      Example: 100
    DISTANCE_THRESHOLD_NONCBD : float, distance threshold in degrees to consider when clustering.
                                       This is applied outside the CBD, or if the CBD is not used, then applied everywhere.
                                       Note that you will get demand points separated by less than this value, so do not 
                                       use this value as a "minimum separation" parameter.
                                Example: 0.1
    DISTANCE_THRESHOLD_CBD : float, like `DISTANCE_THRESHOLD_NONCBD` but applied within the CBD defined by `cbd_bbox`.
                             Example: 0.05
    DEMAND_FACTOR: float, multiply all LODES pop sizes by this factor.
                          Example: 2
    
    point_locs_to_move : list of list of floats, coordinates in [lon, lat] of demand points that you want to move 
                                                 for whatever reason.  
                                                 Must correspond exactly to the order used in `moved_point_locs`.
                                                 To not use this, set it to []
                         Example: [[-77.69260, 43.29925], [-77.69280, 43.28780], [-77.74163, 43.30533], 
                                   [-77.76616, 43.29830], [-77.75377, 43.29501], [-77.73190, 43.29221], 
                                   [-77.71047, 43.28571], [-77.53833, 43.22158]]
    moved_point_locs : list of list of floats, coordinates in [lon, lat] where you want to move the demand points to.
                                               Must correspond exactly to the order used in `point_locs_to_move`.
                       Example: [[-77.69253, 43.29669], [-77.69224, 43.28529], [-77.73431, 43.30351], 
                                 [-77.76969, 43.29365], [-77.75209, 43.29262], [-77.72819, 43.29165], 
                                 [-77.71078, 43.28141], [-77.54152, 43.22132]]
    
    airport_daily_passengers : list of ints, number of daily passengers at the city's airports.
                               Example: [7000] 
    airport_loc : list of list of floats, coordinates in [lon, lat] of the city's airports.
                  Example: [[-77.67166, 43.12919]]
    airport_required_locs : list of list of list of floats, coordinates in [lon, lat] where you want airports' travelers 
                                                    to reside.  One pop will be placed at the demand bubble closest to 
                                                    each specified coordinate.
                                                    If you don't care to set this, then use [] for each airport (e.g., 
                                                    use [[], []] for 2 airports) and the code will decide automatically.
                            Example: [[[-77.61298,  43.15729], [-77.60688,  43.15614], [-77.58936,  43.1547 ],
                                       [-77.59342,  43.15564], [-77.6741 ,  43.21029], [-77.61647,  43.10564],
                                       [-77.61391,  43.08771], [-77.55086,  43.11299], [-77.57981,  43.19774],
                                       [-77.4567 ,  43.2146 ], [-77.44227,  43.21617], [-77.68496,  43.18599],
                                       [-77.64286,  43.0601 ], [-77.65179,  43.05802], [-77.44922,  43.01093],
                                       [-77.51514,  43.09333]]]
    air_pop_size_req : list of ints, size of airports' pops assigned by `airport_required_locs`.  
                            Note that if this exceeds `MAXPOPSIZE` then each pop will be split into multiple smaller pops.
                       Example: [200]
    air_pop_size_remain : list of ints, size of airports' pops assigned automatically by the code.
                          Note that if this exceeds `MAXPOPSIZE` then each pop will be split into multiple smaller pops.
                          Example: [150]
    
    universities : list of strings, 2-4 letter identifier for each university considered.
                                    All subsequent university-related parameters must correspond exactly to this ordering.
                   Example: ["UR", "RIT", "SJF", "NU", "RWU"],
    univ_loc : list of list of floats, coordinates for each university's demand bubble.
               Example: [[-77.62668, 43.12989], [-77.67629, 43.08389], [-77.51239, 43.11575], 
                         [-77.51873, 43.10218], [-77.79857, 43.12568]]
    univ_merge_within : list of ints, distance in meters to merge any nearby demand points into the new university demand point.
                   Example: [0, 350, 300, 0, 0]
    students : list of ints, number of students that attend each campus.
               Example: [11946, 17166, 4000, 2500, 1500]
    perc_oncampus : list of floats, percentage of students that live in on-campus housing for each university.
                    Example: [0.45, 0.4, 0.33, 0.5, 0.6]
    univ_pop_size : list of ints, size of each pop created for each university.
                    Example: [75, 75, 75, 75, 75]
    univ_perc_travel : list of list of floats, fraction of students that live [on campus, off campus] that travel on an average day.   
                                Default: [0.3, 0.5]

    entertainment : list of strings, short identifiers for each entertainment location
    ent_loc : list of list of floats, coordinates in [lon, lat] for each entertainment location 
    ent_req_residences : list of list of list of floats, like `airport_required_locs` but for entertainment locations
    ent_size : list of ints, number of daily visitors to each entertainment location
    ent_pop_size : list of ints, size of each pop created for each entertainment location

TODOs:
- automatically determine list of state(s) based on lat/lon?
"""

import sys, os
import copy
import csv
import glob
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import requests
import time
import numpy as np
import geopandas as gpd
from shapely.ops import unary_union, polygonize
from shapely.geometry import Point
from tqdm import tqdm
import osmnx as ox
import networkx as nx
import platform

from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

FOODIR = os.path.dirname(__file__)
from lib import utils as U

np.random.seed(42)

###############################################################################

if __name__ == "__main__":
    # Load the configuration file
    with open(sys.argv[1], 'r') as fcfg:
        cfg = json.load(fcfg)

    # Defines for preparing the demand file
    MAXPOPSIZE = cfg['MAXPOPSIZE']
    CALCULATE_ROUTES = cfg['CALCULATE_ROUTES']
    SMALL_THRESHOLD = cfg['SMALL_THRESHOLD']
    DISTANCE_THRESHOLD_NONCBD = cfg['DISTANCE_THRESHOLD_NONCBD']
    DISTANCE_THRESHOLD_CBD    = cfg['DISTANCE_THRESHOLD_CBD']
    DEMAND_FACTOR = cfg['DEMAND_FACTOR']
    try:
        HUMAN_READABLE = cfg['HUMAN_READABLE']
    except:
        HUMAN_READABLE = False
    try:
        MAX_WORKERS = cfg['MAX_WORKERS']
    except:
        MAX_WORKERS = None
    
    # Map info
    bbox = cfg['bbox']
    cbd_bbox = cfg['cbd_bbox']
    city = cfg['city']
    airport = cfg['airport']
    if not isinstance(airport, list):
        airport = [airport]
    airport = [iata.upper() for iata in airport]
    states = cfg['state']
    if not isinstance(states, list):
        states = [states]
    states = [state.lower() for state in states]
    year = cfg['year']
    if (year < 2002) or (year > 2023):
        raise ValueError("'year' must be in the range 2002-2023.\nReceived: "+str(year))  

    # Points that are not in the spot you want
    point_locs_to_move = cfg['point_locs_to_move']
    moved_point_locs = cfg['moved_point_locs']

    # Airport data
    airport_daily_passengers = cfg['airport_daily_passengers']
    if not isinstance(airport_daily_passengers, list):
        airport_daily_passengers = [airport_daily_passengers]
    assert len(airport) == len(airport_daily_passengers), str(len(airport))+" airports provided, but "+str(len(airport_daily_passengers))+" daily passenger values provided.  There must be one daily passenger value provided per airport specified."

    airport_loc = cfg['airport_loc']
    if not isinstance(airport_loc[0], list):
        airport_loc = [airport_loc]
    assert len(airport) == len(airport_loc), str(len(airport))+" airports provided, but "+str(len(airport_loc))+" airport locations provided.  There must be one [lon, lat] coordinate value provided per airport specified."

    try:
        airport_required_locs = cfg['airport_required_locs']
    except:
        print("airport_required_locs not specified/understood.  All airport pops will be placed according to the code's simple model.")
        airport_required_locs = [[] for i in range(len(airport))]
    else:
        if not len(airport_required_locs):
            airport_required_locs = [[] for i in range(len(airport))]

    try:
        air_pop_size_req = cfg['air_pop_size_req']
        if not isinstance(air_pop_size_req, list):
            air_pop_size_req = [air_pop_size_req for i in range(len(airport))]
    except:
        print("air_pop_size_req not specified/understood.  Any required locations for airport pops will have MAXPOPSIZE people.")
        air_pop_size_req = [MAXPOPSIZE for i in range(len(airport))]
    try:
        air_pop_size_remain = cfg['air_pop_size_remain']
        if not isinstance(air_pop_size_remain, list):
            air_pop_size_remain = [air_pop_size_remain for i in range(len(airport))]
    except:
        print("air_pop_size_remain not specified/understood.  Using MAXPOPSIZE ("+str(MAXPOPSIZE)+") for airport pops assigned by the code.")
        air_pop_size_remain = [MAXPOPSIZE for i in range(len(airport))]

    # University data
    universities = cfg['universities']
    if not isinstance(universities, list):
        universities = [universities]

    univ_loc = cfg['univ_loc']
    if not isinstance(univ_loc[0], list):
        univ_loc = [univ_loc]
    assert len(universities) == len(univ_loc), str(len(universities))+" universities provided, but "+str(len(univ_loc))+" university locations provided.  There must be one [lon, lat] coordinate value provided per university specified."

    try:
        univ_merge_within = cfg['univ_merge_within']
    except:
        print("univ_merge_within not specified/understood.  No bubbles will be merged around the universities.")
        univ_merge_within = [0 for i in range(len(universities))]
    assert len(universities) == len(univ_merge_within), str(len(universities))+" universities provided, but "+str(len(univ_merge_within))+" merge distances provided.  There must be one merge distance value provided per university specified."

    students = cfg['students']
    if not isinstance(students, list):
        students = [students]
    assert len(universities) == len(students), str(len(universities))+" universities provided, but "+str(len(students))+" student counts provided.  There must be one student count value provided per university specified."

    perc_oncampus = cfg['perc_oncampus']
    if not isinstance(perc_oncampus, list):
        perc_oncampus = [perc_oncampus]
    assert len(universities) == len(perc_oncampus), str(len(universities))+" universities provided, but "+str(len(perc_oncampus))+" % on campus values provided.  There must be one % on campus value provided per university specified."


    try:
        univ_pop_size = cfg['univ_pop_size']
    except:
        print("univ_pop_size not specified/understood.  Using MAXPOPSIZE ("+str(MAXPOPSIZE)+") for university pops.")
        univ_pop_size = [MAXPOPSIZE for i in range(len(universities))]
    assert len(universities) == len(univ_pop_size), str(len(universities))+" universities provided, but "+str(len(univ_pop_size))+" pop sizes provided.  There must be one pop size per university specified."

    try:
        univ_perc_travel = cfg['univ_perc_travel']
    except:
        print("Assuming that 30% of on-campus students and 50% of off-campus students travel daily.")
        univ_perc_travel = [0.3, 0.5]
    assert len(univ_perc_travel) == 2, "univ_pop_size must be a list of 2 values.\nFormat: [% on-campus students that travel daily, % off-campus students that travel daily]"

    # Entertainment data
    try:
        entertainment = cfg['entertainment']
        if not isinstance(entertainment, list):
            entertainment = [entertainment]
        ent_loc = cfg['ent_loc']
        if not isinstance(ent_loc[0], list):
            ent_loc = [ent_loc]
        assert len(ent_loc) == len(entertainment), str(len(entertainment))+" entertainment locations specified, but "+str(len(ent_loc))+" entertainment locations were provided."
        try:
            ent_req_residences = cfg['ent_req_residences']
        except:
            print("ent_req_residences not specified/understood.  All entertainment pops will be placed according to the code's simple model.")
            ent_req_residences = [[] for i in range(len(entertainment))]
        else:
            if not len(ent_req_residences):
                ent_req_residences = [[] for i in range(len(entertainment))]
            assert len(ent_req_residences) == len(entertainment), str(len(entertainment))+" entertainment locations specified, but "+str(len(ent_req_residences))+" groups of required entertainment residences were provided."
        
        ent_size = cfg['ent_size']
        assert len(ent_size) == len(entertainment), str(len(entertainment))+" entertainment locations specified, but "+str(len(ent_size))+"entertainment demand sizes were provided."
        
        try:
            ent_pop_size = cfg['ent_pop_size']
        except:
            print("ent_pop_size not specified/understood.  Using MAXPOPSIZE ("+str(MAXPOPSIZE)+") for entertainment pops.")
            ent_pop_size = [MAXPOPSIZE for i in range(len(entertainment))]
        else:
            assert len(ent_pop_size) == len(entertainment), str(len(entertainment))+" entertainment locations specified, but "+str(len(ent_pop_size))+" entertainment pop sizes were provided."
    except Exception as e:
        print("Entertainment data either not provided or missing required parameters.  Disabling entertainment pops.")
        entertainment = False

    ###############################################################################

    print("Building a demand file for", city)

    # Load needed data
    all_xwalk_ids = []
    all_xwalk_lat = []
    all_xwalk_lon = []
    all_work = []
    all_home = []
    all_pops = []

    for state in states:
        print("  Processing", state, "data for", year, flush=True)

        fxwalk = os.path.join(FOODIR, 'data', state, state+'_xwalk.csv')
        fjobs  = os.path.join(FOODIR, 'data', state, state+'_od_main_JT01_'+str(year)+'.csv')

        if not os.path.exists(fxwalk):
            print("    Downloading", year, "crosswalk data for", state)
            os.makedirs(os.path.dirname(fxwalk), exist_ok=True)
            response = requests.get(f"https://lehd.ces.census.gov/data/lodes/LODES8/{state}/{state}_xwalk.csv.gz")
            response.raise_for_status()
            content = gzip.decompress(response.content)
            with open(fxwalk, "wb") as f:
                f.write(content)

        if not os.path.exists(fjobs):
            print("    Downloading", year, "jobs data for", state)
            os.makedirs(os.path.dirname(fjobs), exist_ok=True)
            response = requests.get(f"https://lehd.ces.census.gov/data/lodes/LODES8/{state}/od/{state}_od_main_JT01_{year}.csv.gz")
            response.raise_for_status()
            content = gzip.decompress(response.content)
            with open(fjobs, "wb") as f:
                f.write(content)

        # Read crosswalk file - xwalk - from LODES https://lehd.ces.census.gov/data/
        print("    Loading crosswalk data")
        with open(fxwalk, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            xwalk_header = next(reader)
            xwalk_rows = np.array([row for row in reader], dtype=str)

        xwalk_ids = xwalk_rows[:,0].astype(int)
        xwalk_trct = xwalk_rows[:,6].astype(int)
        xwalk_lat = xwalk_rows[:,-3].astype(float)
        xwalk_lon = xwalk_rows[:,-2].astype(float)

        xwalk_keep = (xwalk_lon >= bbox[0]) * (xwalk_lat >= bbox[1]) * \
                     (xwalk_lon <= bbox[2]) * (xwalk_lat <= bbox[3])

        xwalk_ids = xwalk_ids[xwalk_keep]
        xwalk_trct = xwalk_trct[xwalk_keep]
        xwalk_lat = xwalk_lat[xwalk_keep]
        xwalk_lon = xwalk_lon[xwalk_keep]

        # Read jobs file - JT01 - from LODES
        print("    Loading jobs data")
        with open(fjobs, 'r') as foo:
            jobs_header = np.array(foo.readlines()[0].strip().split(','))
        jobs = np.loadtxt(fjobs, delimiter=',', skiprows=1, dtype=int)
        work = jobs[:,0]
        home = jobs[:,1]
        pops = jobs[:,2]

        ikeep = np.array([w in xwalk_ids for w in work]) * \
                np.array([h in xwalk_ids for h in home])
        work = work[ikeep]
        home = home[ikeep]
        pops = pops[ikeep]
        
        all_xwalk_ids.append(xwalk_ids)
        all_xwalk_lat.append(xwalk_lat)
        all_xwalk_lon.append(xwalk_lon)
        all_work.append(work)
        all_home.append(home)
        all_pops.append(pops)

    xwalk_ids = np.concatenate(all_xwalk_ids)
    xwalk_lat = np.concatenate(all_xwalk_lat)
    xwalk_lon = np.concatenate(all_xwalk_lon)
    work = np.concatenate(all_work)
    home = np.concatenate(all_home)
    pops = np.concatenate(all_pops)
    iblocks = np.arange(xwalk_ids.size, dtype=int)

    ###############################################################################

    # Go through each block - log number of pops and their workplace block
    block_data = {}

    def process_block(block):
        ret = {}
        iblock = home == block
        work_locations = np.unique(work[iblock])
        for w in work_locations:
            ipops = work[iblock] == w
            ret[w] = pops[iblock][ipops].sum()
        return block, ret

    print("  Processing block data")
    if platform.system() == "Windows":
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            results = ex.map(process_block, xwalk_ids)
    else:
        with Pool(processes=MAX_WORKERS) as pool:
            results = pool.map(process_block, xwalk_ids)
    for res in results:
        block_data[res[0]] = res[1]

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
    for iblock in iblocks:
        print("  Processing point", iblock+1, "/", iblocks.size, end='\r')
        point = {
            "id" : str(xwalk_ids[iblock]),
            "location" : [float(xwalk_lon[iblock]), float(xwalk_lat[iblock])],
            "jobs" : 0,
            "residents" : 0,
            "popIds" : []
        }
        demand["points"].append(point)
    print("")

    points_by_id = {p["id"]: p for p in demand["points"]}

    ipops = np.arange(len(pops), dtype=int)
    for ipop in ipops:
        print("  Processing pop", ipop+1, '/', ipops.size, end='\r')
        pop = {
            "id" : str(ipop+1),
            "residenceId": str(home[ipop]),
            "jobId": str(work[ipop]),
            "size" : int(pops[ipop]), 
            "drivingSeconds": 0,  # Will be filled in later
            "drivingDistance": 0, # after merging pops
        }
        points_by_id[pop["residenceId"]]["residents"] += pops[ipop]
        points_by_id[pop["jobId"      ]]["jobs"]      += pops[ipop]
        points_by_id[pop["residenceId"]]["popIds"].append(str(ipop+1))
        points_by_id[pop["jobId"      ]]["popIds"].append(str(ipop+1))
        demand["pops"].append(pop)
    print("")

    print("Initial points:", len(demand['points']))
    print("Initial pops:", len(demand['pops']))
    print("Initial total pop size:", np.sum([p['size'] for p in demand['pops']]))
    print("Initial workers:", np.sum([p['jobs'] for p in demand["points"]]))
    print("Initial residents:", np.sum([p['residents'] for p in demand["points"]]))

    ###############################################################################

    print("Agglomerating pops below a threshold size of", SMALL_THRESHOLD)
    flows_by_dest = defaultdict(list)
    for pop in demand['pops']:
        flows_by_dest[pop["jobId"]].append(pop)

    new_pops = []
    new_points = list(demand['points'])
    super_origin_counter = 0

    for dest_id, flows in flows_by_dest.items():
        orig_total = sum(f["size"] for f in flows)
        large_flows = [f.copy() for f in flows if f["size"] >= SMALL_THRESHOLD]
        small_flows = [f.copy() for f in flows if f["size"] < SMALL_THRESHOLD]
        new_flows_for_dest = []

        # cluster small origins spatially
        #origin_coords = []
        #small_indices_with_coords = []
        cbd_coords, cbd_indices = [], []
        noncbd_coords, noncbd_indices = [], []
        for i, f in enumerate(small_flows):
            loc = points_by_id[f["residenceId"]]["location"]
            if loc is not None:
                #origin_coords.append(loc)
                #small_indices_with_coords.append(i)
                if U.in_cbd(loc, cbd_bbox):
                    cbd_coords.append(loc)
                    cbd_indices.append(i)
                else:
                    noncbd_coords.append(loc)
                    noncbd_indices.append(i)
        #origin_coords = np.array(origin_coords) if origin_coords else np.empty((0, 2))
        cbd_coords = np.array(cbd_coords) if cbd_coords else np.empty((0, 2))
        noncbd_coords = np.array(noncbd_coords) if noncbd_coords else np.empty((0, 2))

        initial_clusters = []
        # Non-CBD regime
        if len(noncbd_indices) > 1:
            clustering_non = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=DISTANCE_THRESHOLD_NONCBD,  # larger threshold
                linkage="ward"
            ).fit(np.array(noncbd_coords))
            labels_non = clustering_non.labels_
            label_to_indices_non = defaultdict(list)
            for lbl, idx_in_valid in zip(labels_non, range(len(noncbd_indices))):
                label_to_indices_non[lbl].append(noncbd_indices[idx_in_valid])
            initial_clusters.extend(label_to_indices_non.values())
        elif len(noncbd_indices) == 1:
            initial_clusters.append([noncbd_indices[0]])
        
        # CBD regime
        if len(cbd_indices) > 1:
            clustering_cbd = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=DISTANCE_THRESHOLD_CBD,  # smaller threshold
                linkage="ward"
            ).fit(np.array(cbd_coords))
            labels_cbd = clustering_cbd.labels_
            label_to_indices_cbd = defaultdict(list)
            for lbl, idx_in_valid in zip(labels_cbd, range(len(cbd_indices))):
                label_to_indices_cbd[lbl].append(cbd_indices[idx_in_valid])
            initial_clusters.extend(label_to_indices_cbd.values())
        elif len(cbd_indices) == 1:
            initial_clusters.append([cbd_indices[0]])
        
        used = set(sum(initial_clusters, []))
        no_coord = [i for i in range(len(small_flows)) if i not in used]
        initial_clusters.extend([[i] for i in no_coord])
        

        # emit clusters as SuperOrigins
        for idxs in initial_clusters:
            cl_flows = [small_flows[i] for i in idxs]
            sizes = [f["size"] for f in cl_flows]
            seconds_vals  = [f["drivingSeconds" ] for f in cl_flows]
            distance_vals = [f["drivingDistance"] for f in cl_flows]
            agg_size = sum(sizes)
            agg_seconds = int(round(U.weighted_mean(seconds_vals, sizes)))
            agg_distance = int(round(U.weighted_mean(distance_vals, sizes)))
            centroid = U.compute_centroid([points_by_id[f["residenceId"]]["location"] \
                                           for f in cl_flows], 
                                          [points_by_id[f["residenceId"]]["residents"] + \
                                           points_by_id[f["residenceId"]]["jobs"]        \
                                           for f in cl_flows])
            residents_sum = np.sum([points_by_id[f["residenceId"]]["residents"] \
                                    for f in cl_flows])

            super_origin_counter += 1
            super_id = f"SO_{super_origin_counter}"
            pop_id = f"agg_{super_origin_counter}"

            new_flows_for_dest.append({
                "residenceId": super_id,
                "jobId": dest_id,
                "drivingSeconds": agg_seconds,
                "drivingDistance": agg_distance,
                "size": agg_size,
                "id": pop_id
            })
            # Update the destination point's pops
            new_points.append({
                "id": super_id,
                "location": centroid,
                "jobs": 0,
                "residents": agg_size, 
                "popIds": [pop_id]
            })

        new_flows_for_dest.extend(large_flows)

        # conservation check
        new_total = sum(f["size"] for f in new_flows_for_dest)
        if new_total != orig_total:
            raise ValueError(f"Commuter mismatch at destination {dest_id}: {orig_total} vs {new_total}")

        new_pops.extend(new_flows_for_dest)

    print("  Re-calculating points' residents and popIds")
    point_dict = {}
    for ip, p in enumerate(new_points):
        p['popIds'] = []
        p['residents'] = 0
        p['jobs'] = 0
        point_dict[p['id']] = ip

    for ip, p in enumerate(new_pops):
        new_points[point_dict[p['residenceId']]]['residents'] += p['size']
        new_points[point_dict[p['residenceId']]]['popIds'].append(p['id'])
        new_points[point_dict[p['jobId']]]['jobs'] += p['size']
        if p['residenceId'] != p['jobId']:
            new_points[point_dict[p['jobId']]]['popIds'].append(p['id'])


    demand['points'] = new_points
    demand['pops'] = new_pops
    print("  Current points:", len(new_points))
    print("  Current pops:", len(new_pops))

    ###############################################################################

    print("Clustering points based on Colin's method")
    merged_points = []
    pts = copy.deepcopy(demand['points'])
    counter = 0
    while counter < 5: # Fail-safe - usually in a ~steady state after this many rounds
        # Order points by size
        size_of_points = np.array([p["residents"] + p["jobs"] for p in pts])
        
        unique_locs     = np.empty((0, 2), dtype=float)#[]
        loc_assignments = []
        isort = np.argsort(size_of_points)[::-1] # Largest -> smallest
        sorted_points = [pts[ip] for ip in isort]
        
        size_of_points = size_of_points[isort] # Reorder to match
        
        # First go from largest -> smallest to figure out which are merged where
        for ipoint, p in enumerate(sorted_points):
            print("  ("+str(counter+1)+") Determining mergers:", ipoint+1, "/", len(sorted_points), end='\r')
            # Determine the buffer size for merging this point
            merge_buffer = U.bufferMeters[size_of_points[ipoint] <= U.maxPopThreshold][0]
            if not ipoint:
                #unique_locs.append(p['location'])
                unique_locs = np.vstack([unique_locs, p["location"]])
                loc_assignments.append(0)
            else:
                # Determine if any existing locations are close enough to this one
                dists = U.haversine(p['location'][0], p['location'][1], 
                                    unique_locs[:,0], unique_locs[:,1])
                iloc = dists.argmin()
                if dists[iloc] > merge_buffer:
                    # New point
                    loc_assignments.append(len(unique_locs))
                    #unique_locs.append(p['location'])
                    unique_locs = np.vstack([unique_locs, p["location"]])
                else:
                    # Existing point
                    loc_assignments.append(iloc)
        print("")
        
        merged_points = []
        pops_by_id = {p["id"]: p for p in demand["pops"]}
        
        # Then merge the points
        def merge_points(inps):
            ipoint, unique_loc = inps
            iloc = [ip==ipoint for ip in loc_assignments]
            these_points = [sorted_points[p] for p in range(len(sorted_points)) if iloc[p]]
            pids = [p['id'] for p in these_points]
            merged_id = these_points[size_of_points[iloc].argmax()]['id']
            if 'merged_' not in merged_id:
                # To make clear that this point had others merged into it
                merged_id = 'merged_' + merged_id
            merged_loc = U.compute_centroid([p['location'] for p in these_points], size_of_points[iloc])
            merged_jobs = int(np.sum([p['jobs'] for p in these_points]))
            merged_residents = int(np.sum([p['residents'] for p in these_points]))
            merged_popIds = []
            for p in these_points:
                merged_popIds += p['popIds']
            merged_popIds = np.unique(merged_popIds).tolist()
            merged_point = {
                "id": merged_id,
                "location": merged_loc,
                "jobs": merged_jobs,
                "residents": merged_residents,
                "popIds": merged_popIds
            }
            # Update pops
            updated_pops = []
            for popid in merged_popIds:
                p = copy.deepcopy(pops_by_id[popid])
                if p['residenceId'] in pids:
                    p['residenceId'] = merged_id
                if p['jobId'] in pids:
                    p['jobId'] = merged_id
                updated_pops.append(p)
            return merged_point, updated_pops

        print((len(str(counter+1)) + 4) * " " + " Merging")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            #merged_points = list(ex.map(merge_points, enumerate(unique_locs)))
            results = list(ex.map(merge_points, enumerate(unique_locs)))

        # Process it
        merged_points = []
        for merged_point, updated_pops in results:
            merged_points.append(merged_point)
            for pop in updated_pops:
                if pop['residenceId'] == merged_point['id']:
                    pops_by_id[pop['id']]['residenceId'] = merged_point['id']
                if pop['jobId'] == merged_point['id']:
                    pops_by_id[pop['id']]['jobId'] = merged_point['id']

        if len(merged_points) == len(pts):
            # It is already in a steady state, no need to continue
            break
        pts = copy.deepcopy(merged_points) # Update `pts` for next round
        counter += 1

    demand['points'] = merged_points

    ###############################################################################

    print("Ensuring all points are located on land")
    # Kronifer suggestion: use the water.geojson file in the raw_data
    fcoast = os.path.join("data", "coasts", "ne_10m_coastline.zip")
    if not os.path.exists(fcoast):
        os.makedirs(os.path.dirname(fcoast), exist_ok=True)
        print("Downloading coastline data")
        url = 'https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_coastline.zip'
        response = requests.get(url)
        response.raise_for_status()
        with open(fcoast, "wb") as f:
            f.write(response.content)

    coast = gpd.read_file("zip://"+fcoast+"!ne_10m_coastline.shp")
    coast_merged = unary_union(coast.geometry)
    coast_polygons = list(polygonize(coast_merged))
    land = gpd.GeoDataFrame(geometry=coast_polygons, crs=coast.crs)
    sindex = land.sindex
    def is_land(loc):
        lon, lat = loc
        pt = Point(lon, lat)
        # First: find candidate polygons using the spatial index
        candidates = list(sindex.intersection(pt.bounds))
        if not candidates:
            return False  # definitely water
        # Second: precise geometry test
        return land.iloc[candidates].contains(pt).any()

    pop_dict = {}
    for ip, p in enumerate(demand["pops"]):
        pop_dict[p['id']] = ip

    ip=0
    while ip < len(demand["points"]):
        print("  Checking point", ip+1, '/', len(demand['points']), end='\r')
        p = demand["points"][ip]
        if not is_land(p["location"]):
            # Over water - merge this into the nearest point
            # Find nearest point
            closest_point = min(
                                (point for point in demand['points'] if point["location"] != p["location"]),
                                key=lambda p: U.haversine(p    ["location"][0], p    ["location"][1], 
                                                          point["location"][0], point["location"][1])
                            )
            # Merge into it
            closest_point['jobs'] += p['jobs']
            closest_point['residents'] += p['residents']
            closest_point['popIds'] += p['popIds']
            closest_point['popIds'] = np.unique(closest_point['popIds']).tolist()
            # Update pops
            for popid in p['popIds']:
                if demand['pops'][pop_dict[popid]]['residenceId'] == p['id']:
                    demand['pops'][pop_dict[popid]]['residenceId'] = closest_point['id']
                if demand['pops'][pop_dict[popid]]['jobId'] == p['id']:
                    demand['pops'][pop_dict[popid]]['jobId'] = closest_point['id']
            del demand["points"][ip]
        else:
            ip += 1

    # Some points may still be over water - use the water.geojson file
    #fwater = '../subwaybuilder-patcher-gui-new/patcher/packages/mapPatcher/raw_data/ROC/water.geojson'
    #water_gdf = gpd.read_file(fwater)
    #for p in demand['points']:
    #    point = Point(*p['location'])
    #    if water_gdf.contains(point).any():
            # Point is over water

    # Move the points that are over water or placed poorly
    point_locs = np.array([p['location'] for p in demand['points']])
    for i, loc in enumerate(point_locs_to_move):
        iloc = U.haversine(loc[0], loc[1], 
                            point_locs[:,0], point_locs[:,1]).argmin()
        demand['points'][iloc]['location'] = moved_point_locs[i]

    ###############################################################################

    # Merge any pops that have identical homes and works
    print("Merging pops with identical commutes")
    index_map = defaultdict(list)
    for idx, entry in enumerate(demand['pops']):
        key = (entry["residenceId"], entry["jobId"])
        index_map[key].append(idx)
    #duplicates = {k: v for k, v in index_map.items() if len(v) > 1}
    new_pops = []
    keys = list(index_map.keys())
    points_by_id = {p["id"]: p for p in demand["points"]}
    for k in keys:
        imerge = index_map[k]
        nmerge = len(imerge)
        if nmerge > 1:
            pop = {
                "id" : demand['pops'][imerge[0]]["id"],
                "residenceId" : demand['pops'][imerge[0]]["residenceId"],
                "jobId" : demand['pops'][imerge[0]]["jobId"],
                "size" : int(np.sum([demand['pops'][imerge[i]]["size"] for i in range(nmerge)])),
                "drivingSeconds"  : 0,#U.weighted_mean([demand['pops'][imerge[i]]["drivingSeconds"] for i in range(nmerge)], 
                                      #                [demand['pops'][imerge[j]]["size"] for j in range(nmerge)]),
                "drivingDistance" : 0 #U.weighted_mean([demand['pops'][imerge[i]]["drivingDistance"] for i in range(nmerge)], 
                                      #                [demand['pops'][imerge[j]]["size"] for j in range(nmerge)])
            }
            # Update points to forget about old pops that no longer exist
            for i in range(1,nmerge):
                points_by_id[k[0]]["popIds"].remove(demand['pops'][imerge[i]]["id"])
                if k[0] != k[1]:
                    points_by_id[k[1]]["popIds"].remove(demand['pops'][imerge[i]]["id"])
        else:
            pop = demand['pops'][imerge[0]]
        new_pops.append(pop)
    demand['pops'] = new_pops

    print("  Current pops:", len(demand['pops']))

    ###############################################################################

    if DEMAND_FACTOR != 1:
        print("Applying a demand scaling factor of", DEMAND_FACTOR)

        points_by_id = {p["id"]: p for p in demand["points"]}

        for p in demand['pops']:
            size = p['size']
            new_size = int(size * DEMAND_FACTOR)
            addtl = new_size - size
            points_by_id[p['residenceId']]['residents'] += addtl
            points_by_id[p['jobId']]['jobs'] += addtl
            p['size'] = new_size

    ###############################################################################

    print("Adding airport demand to simulate travelers")
    air_points = []
    counter = 0
    for iair in range(len(airport)):
        print(" ", airport[iair])

        point = {
            "id": "AIR_"+airport[iair],
            "location": airport_loc[iair],
            "jobs": 0,
            "residents": 0,
            "popIds": []
        }

        point_locs = np.array([p['location'] for p in demand['points']])

        # Calculate where the pops will "live"
        # Required points - Find nearest points to these coords
        ilocs_air_req = np.zeros(len(airport_required_locs[iair]), dtype=int)
        for i in range(len(airport_required_locs[iair])):
            ilocs_air_req[i] = U.haversine(airport_required_locs[iair][i][0], airport_required_locs[iair][i][1], 
                                           point_locs[:,0], point_locs[:,1]).argmin()

        # And determine remaining number of points that will get pops
        ntarget_locs_air_remain = int((airport_daily_passengers[iair] - \
                                       (air_pop_size_req[iair] * len(airport_required_locs[iair]))) / \
                                      air_pop_size_remain[iair])
        size_of_points = np.array([p['residents'] for p in demand['points']])
        size_of_points[ilocs_air_req] = 0 # Don't consider these points
        ilocs_air_remain = np.random.choice(size_of_points.size, size=ntarget_locs_air_remain, replace=False, p=size_of_points/size_of_points.sum())

        # Make them
        for it in range(2):
            if not it:
                psize = air_pop_size_req[iair]
                locs_arr = ilocs_air_req
            else:
                psize = air_pop_size_remain[iair]
                locs_arr = ilocs_air_remain
            for i, iloc in enumerate(locs_arr):
                counter += 1
                pop = {
                        "id" : "AIR_"+str(counter),
                        "residenceId" : demand['points'][iloc]["id"],
                        "jobId" : point["id"],
                        "size" : psize,
                        "drivingSeconds"  : 0,
                        "drivingDistance" : 0
                }
                demand['pops'].append(pop)
                demand['points'][iloc]['residents'] += pop['size']
                point["jobs"] += pop['size']
                demand['points'][iloc]['popIds'].append(pop['id'])
                point['popIds'].append(pop['id'])
        air_points.append(point)

    demand['points'] += air_points

    ###############################################################################

    print("Adding university demand")
    univ_points = []
    for iuniv in range(len(universities)):
        print(" ", universities[iuniv], students[iuniv], perc_oncampus[iuniv])
        oncampus = int(perc_oncampus[iuniv] * students[iuniv]) # live on campus, "work" elsewhere
        offcampus = students[iuniv] - oncampus # "work" on campus, live elsewhere
        
        point = {
            "id": "UNI_" + universities[iuniv],
            "location": univ_loc[iuniv],
            "jobs": 0,
            "residents": 0,
            "popIds": []
        }
        
        if univ_merge_within[iuniv]:
            # Merge nearby points into this one
            point_locs = np.array([p['location'] for p in demand['points']])
            dists = U.haversine(point['location'][0], point['location'][1], 
                                point_locs[:,0], point_locs[:,1])
            iloc_merge = np.arange(len(demand['points']), dtype=int)[dists <= univ_merge_within[iuniv]][::-1] # largest to smallest
            pops_by_id = {p["id"]: p for p in demand["pops"]}
            for iloc in iloc_merge:
                point['jobs'] += demand['points'][iloc]['jobs']
                point['residents'] += demand['points'][iloc]['residents']
                point['popIds'] += demand['points'][iloc]['popIds']
                for popid in demand['points'][iloc]['popIds']:
                    if pops_by_id[popid]['residenceId'] == demand['points'][iloc]['id']:
                        pops_by_id[popid]['residenceId'] = point['id']
                    if pops_by_id[popid]['jobId'] == demand['points'][iloc]['id']:
                        pops_by_id[popid]['jobId'] = point['id']
                del demand['points'][iloc]
        
        # On-campus students
        point_locs = np.array([p['location'] for p in demand['points']])
        iloc_airport = [p['id'][:4] == "AIR_" for p in demand['points']]
        size_of_points = np.array([p['jobs'] for p in demand['points']])
        size_of_points[iloc_airport] = 0 # Don't consider the airport
        dist_of_points = U.haversine(point['location'][0], point['location'][1], 
                                     point_locs[:,0], point_locs[:,1])
        weight_of_points = size_of_points / dist_of_points**2 # Prefer places near campus
        ilocs = np.random.choice(weight_of_points.size, 
                                 size=int((oncampus * univ_perc_travel[0])//univ_pop_size[iuniv]), 
                                 p=weight_of_points/weight_of_points.sum())
        for i, iloc in enumerate(ilocs):
            pop = {
                    "id" : "UNI_" + universities[iuniv] + "_" + str(i+1),
                    "residenceId" : point["id"],
                    "jobId" : demand['points'][iloc]["id"],
                    "size" : int(univ_pop_size[iuniv]),
                    "drivingSeconds"  : 0,
                    "drivingDistance" : 0
            }
            demand['pops'].append(pop)
            demand['points'][iloc]['jobs'] += pop['size']
            point["residents"] += pop['size']
            demand['points'][iloc]['popIds'].append(pop['id'])
            point['popIds'].append(pop['id'])

        # Off-campus students
        size_of_points = np.array([p['residents'] for p in demand['points']])
        size_of_points[iloc_airport] = 0 # Don't consider the airport
        dist_of_points = U.haversine(point['location'][0], point['location'][1], 
                                     point_locs[:,0], point_locs[:,1])
        weight_of_points = size_of_points / dist_of_points
        ilocs = np.random.choice(weight_of_points.size, 
                                 size=int((offcampus * univ_perc_travel[1])//univ_pop_size[iuniv]), 
                                 p=weight_of_points/weight_of_points.sum())
        for j, iloc in enumerate(ilocs):
            pop = {
                    "id" : "UNI_" + universities[iuniv] + "_" + str(i+j+2),
                    "residenceId" : demand['points'][iloc]["id"],
                    "jobId" : point["id"],
                    "size" : int(univ_pop_size[iuniv]),
                    "drivingSeconds"  : 0,
                    "drivingDistance" : 0
            }
            demand['pops'].append(pop)
            demand['points'][iloc]['residents'] += pop['size']
            point["jobs"] += pop['size']
            demand['points'][iloc]['popIds'].append(pop['id'])
            point['popIds'].append(pop['id'])
        univ_points.append(point)

    demand['points'] += univ_points

    ###############################################################################

    if entertainment:
        print("Adding entertainment demand")
        ent_points = []
        counter = 0
        for ient in range(len(entertainment)):
            print(" ", entertainment[ient], ent_size[ient])
            point = {
                "id": "ENT_" + entertainment[ient],
                "location": ent_loc[ient],
                "jobs": 0,
                "residents": 0,
                "popIds": []
            }
            point_locs = np.array([p['location'] for p in demand['points']])

            # Calculate where the pops will "live"
            # Required points - Find nearest points to these coords
            ilocs_ent_req = np.zeros(len(ent_req_residences[ient]), dtype=int)
            for i in range(len(ent_req_residences[ient])):
                ilocs_ent_req[i] = U.haversine(ent_req_residences[ient][i][0], ent_req_residences[ient][i][1], 
                                               point_locs[:,0], point_locs[:,1]).argmin()

            # And determine remaining number of points that will get pops
            psize = ent_pop_size[ient]
            ntarget_locs_ent_remain = int((ent_size[ient] - (psize * len(ent_req_residences[ient]))) / \
                                          psize)
            size_of_points = np.array([p['residents'] for p in demand['points']])
            iloc_airport = [p['id'][:4] == "AIR_" for p in demand['points']]
            size_of_points[iloc_airport ] = 0 # Don't consider these points
            size_of_points[ilocs_ent_req] = 0 
            dist_of_points = U.haversine(point['location'][0], point['location'][1], 
                                         point_locs[:,0], point_locs[:,1])
            weight_of_points = size_of_points / dist_of_points
            
            ilocs_ent_remain = np.random.choice(size_of_points.size, size=ntarget_locs_ent_remain, 
                                                replace=False, 
                                                p=weight_of_points/weight_of_points.sum())
                                                #p=(size_of_points + weight_of_points) / \
                                                #  (size_of_points.sum() + weight_of_points.sum()))

            # Make them
            for it in range(2):
                if not it:
                    locs_arr = ilocs_ent_req
                else:
                    locs_arr = ilocs_ent_remain
                for i, iloc in enumerate(locs_arr):
                    counter += 1
                    pop = {
                            "id" : "ENT_"+str(counter),
                            "residenceId" : demand['points'][iloc]["id"],
                            "jobId" : point["id"],
                            "size" : psize,
                            "drivingSeconds"  : 0,
                            "drivingDistance" : 0
                    }
                    demand['pops'].append(pop)
                    demand['points'][iloc]['residents'] += pop['size']
                    point["jobs"] += pop['size']
                    demand['points'][iloc]['popIds'].append(pop['id'])
                    point['popIds'].append(pop['id'])
            ent_points.append(point)

        demand['points'] += ent_points


    ###############################################################################

    # Parallelize the route calculations over home nodes
    if CALCULATE_ROUTES:
        points_by_id = {p["id"]: p for p in demand["points"]}
        pops_by_id   = {p["id"]: p for p in demand["pops"  ]}
        
        # Set up OSM graph
        print("Initializing OSM drive network graph")
        G = ox.graph_from_bbox(bbox, network_type='drive')#, simplify=False)
        G = ox.truncate.largest_component(G, strongly=True)
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        
        def process_home_node(i):
            home_point = demand['points'][i]
            home_id = home_point['id']
            home_node = ox.nearest_nodes(G, Y=home_point['location'][1], X=home_point['location'][0])
            pops = [p for p in demand['pops'] if p['residenceId'] == home_id]
            for p in pops:
                job_id = p['jobId']
                job_point = points_by_id[job_id]
                try:
                    job_node = ox.nearest_nodes(G, Y=job_point['location'][1], X=job_point['location'][0])
                    path_nodes = nx.shortest_path(G, home_node, job_node, weight='travel_time')
                    distance_in_meters = nx.path_weight(G, path_nodes, weight='length')
                    travel_time_in_seconds = nx.path_weight(G, path_nodes, weight='travel_time')
                except:
                    try:
                        # Find closest road segment and project a point onto it
                        x, y = job_point['location']
                        u, v, key = ox.nearest_edges(G, Y=y, X=x)
                        edge_data = G[u][v][key]
                        line = edge_data['geometry']
                        point = Point(x, y)
                        nearest_point = line.interpolate(line.project(point))
                        new_node = max(G.nodes) + 1
                        G.add_node(new_node, x=nearest_point.x, y=nearest_point.y)
                        dist_to_u = Point(G.nodes[u]['x'], G.nodes[u]['y']).distance(nearest_point)
                        dist_to_v = Point(G.nodes[v]['x'], G.nodes[v]['y']).distance(nearest_point)
                        G.add_edge(new_node, u, length=dist_to_u)
                        G.add_edge(new_node, v, length=dist_to_v)
                        job_node = ox.nearest_nodes(G, X=x, Y=y)
                        path_nodes = nx.shortest_path(G, home_node, job_node, weight='travel_time')
                        distance_in_meters = nx.path_weight(G, path_nodes, weight='length')
                        travel_time_in_seconds = nx.path_weight(G, path_nodes, weight='travel_time')
                    except:
                        path_nodes = []
                        distance_in_meters = 0
                        travel_time_in_seconds = 0
                # Add time penalties for intersections + traffic: 5 seconds per intersection
                travel_time_in_seconds += len(path_nodes) * 5
                
                p['drivingSeconds']  = int(travel_time_in_seconds)
                p['drivingDistance'] = int(np.ceil(distance_in_meters))
            return pops
        
        # Prepare arguments for parallel jobs
        print("Calculating driving paths for each home node.  This may take a while.")

        if platform.system() == "Windows":
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                results = list(ex.map(process_home_node, [i for i in range(0,len(demand["points"]))]))
        else:
            with Pool(processes=MAX_WORKERS) as pool:
                #results = pool.map(process_home_node, range(len(demand['points'])))
                results = []
                for r in tqdm(pool.imap(process_home_node, range(len(demand['points']))), total=len(demand['points'])):
                    results.append(r)
        
        # Flatten results and update demand
        for ret in results:
            for pop in ret:
                pops_by_id[pop['id']]['drivingSeconds']  = pop['drivingSeconds']
                pops_by_id[pop['id']]['drivingDistance'] = pop['drivingDistance']

    ###############################################################################

    # Make sure that pops are <=200 in size
    print("Pops before enforcing size <="+str(MAXPOPSIZE)+":", len(demand['pops']))
    for p in demand['pops']:
        if p['size'] > MAXPOPSIZE:
            niter = int(np.ceil(p['size'] / MAXPOPSIZE))
            for n in range(1, niter):
                pop = copy.deepcopy(p)
                pop['id'] += "_"+str(n)
                if n < niter - 1:
                    # More than MAXPOPSIZE pops remain - cap at MAXPOPSIZE
                    pop["size"] = MAXPOPSIZE
                else:
                    # Less than MAXPOPSIZE remains - put all into this pop
                    pop["size"] = int(p['size']) % MAXPOPSIZE
                demand["pops"].append(pop)
            # Update the original pop
            p['size'] = MAXPOPSIZE

    print("Final points:", len(demand['points']))
    print("Final pops:", len(demand['pops']))
    print("Final total pop size:", np.sum([p['size'] for p in demand['pops']]))
    print("Final workers:", np.sum([p['jobs'] for p in demand['points']]))
    print("Final residents:", np.sum([p['residents'] for p in demand['points']]))

    ###############################################################################

    # Save out demand file
    os.makedirs(os.path.join(os.path.dirname(__file__), 'demand_data', city.replace(' ', '')), exist_ok=True)
    filename = os.path.join('demand_data', city.replace(' ', ''), 'demand_data.json')
    with open(filename, "w") as json_file:
        if HUMAN_READABLE:
            json.dump(demand, json_file, indent=4)
        else:
            json.dump(demand, json_file, indent=None, separators=(',', ':'))
