### This code allows you to calculate demand data for US cities using the real origin-destination pairs provided by the LODES data: https://lehd.ces.census.gov/data/

To make the resulting number of pops and points feasible for Subway Builder, a multi-step agglomeration 
process is used.  If you disable the agglomeration, or use small values for agglomeration, then it is 
likely that the resulting demand data will lag when used in Subway Builder.  See the examples for 
reasonable values you can use for most cities.


# Setting up the environment
If running the Python code directly rather than a pre-compiled release, use the provided 
environment.yml file to create a compatible conda environment.  That recipe is confirmed to work as of 
15 Feb 2026.  If you use different package versions, there is no guarantee that the code will work.  
I will not help troubleshoot any issues that arise if you are using different package versions.

First, make sure you have a working conda installation.  If you don't already have conda, 
you can acquire a lightweight version via miniforge:
    https://github.com/conda-forge/miniforge/releases

Once you have conda, build the environment via

    conda env create -f environment.yml
If this fails on your OS/architecture, @slurry in the Discord and provide as much detail as possible.


# Running the code
The code is compatible with Linux and Mac.  Sorry Windows users, blame Microsoft for not following basic 
conventions when forking processes.  The recommended solution for Windows users is to use WSL, then use 
the Linux file in the latest release (or run the raw Python code) within WSL.  If someone is willing to 
refactor the code in a way that runs on Windows and preserves performance on Linux and Mac, I would love 
that - please pull request these changes if you go through the effort to do that.

This script is meant to be run like

    ./create_US_demand_file.py Rochester.json

If you're downloading a pre-compiled release, then it might look something like

    ./create_US_demand_file.bin Rochester.json

You must provide an input JSON file.  Below are all available parameters for this input file.
Some parameters are optional: a few parameters that govern how the code runs, airport required locations and pop sizes, some parameters for universities, and all entertainment-related parameters.
If omitted or not properly specified, they will be ignored.
All other parameters are required.
See the examples/ directory for example JSON input files used for some of the maps shared on the Discord.

# Parameters
## Core City & Map Parameters
<table>
  <tr>
    <th style="width:150px;">Parameter</th>
    <th style="width:100px;">Type</th>
    <th style="width:250px;">Description</th>
    <th>Example</th>
  </tr>

  <tr>
    <td>city</td>
    <td>string</td>
    <td>The city you're modeling.</td>
    <td><code>"Rochester"</code></td>
  </tr>

  <tr>
    <td>states</td>
    <td>string or list</td>
    <td>Two‑letter state code(s) your map covers.</td>
    <td><code>"ny"</code><br><code>["md","dc","va"]</code></td>
  </tr>

  <tr>
    <td>year</td>
    <td>int</td>
    <td>Year of LODES data (2002–2023).<br><b>Note:</b> Some states only have data for part of this range.</td>
    <td><code>2022</code></td>
  </tr>

  <tr>
    <td>bbox</td>
    <td>list of ints</td>
    <td>[min_lon, min_lat, max_lon, max_lat] boundary for the city.</td>
    <td><code>[-77.8216, 43.0089, -77.399, 43.3117]</code></td>
  </tr>

  <tr>
    <td>cbd_bbox</td>
    <td>list of ints</td>
    <td>CBD boundary; reduces clustering downtown.<br>Set to null to disable.</td>
    <td><code>null</code></td>
  </tr>

  <tr>
    <td>HUMAN_READABLE</td>
    <td>(optional)<br>bool</td>
    <td>Indent JSON output for readability.<br>Default: false.</td>
    <td><code>true</code></td>
  </tr>

  <tr>
    <td>MAX_WORKERS</td>
    <td>(optional)<br>int</td>
    <td>Number of workers for parallel processing.<br>Default: None (as many threads as you have).</td>
    <td><code>4</code></td>
  </tr>
</table>

## Clustering & Demand Generation
<table>
  <tr>
    <th style="width:150px;">Parameter</th>
    <th style="width:100px;">Type</th>
    <th style="width:250px;">Description</th>
    <th>Example</th>
  </tr>

  <tr>
    <td>MAXPOPSIZE</td>
    <td>int</td>
    <td>Maximum size of any pop; larger pops are split.<br><b>Note:</b> &le;200 is recommended due to the capacity of trains. If a pop is larger than a train can hold, it cannot use the metro.</td>
    <td><code>200</code></td>
  </tr>

  <tr>
    <td>CALCULATE_ROUTES</td>
    <td>bool</td>
    <td>Whether to calculate commuting routes<br><b>Note:</b> This can take a long time when using <code>osmnx</code>. Set to false while testing to save time until you're ready to make the final version.</td>
    <td><code>true</code></td>
  </tr>

  <tr>
    <td>ROUTING_METHOD</td>
    <td>(optional)<br>string</td>
    <td>Method to use when calculating routes.<br><b>Options:</b> <code>osmnx</code>, <code>osrm</code><br><b>Note:</b> when using OSRM, you must have already set up a local OSRM server on port 5000 for the specified <code>bbox</code>.</td>
    <td>Default: <code>osmnx</code></td>
  </tr>

  <tr>
    <td>SMALL_THRESHOLD</td>
    <td>int</td>
    <td>Max pop size considered for agglomerative clustering.</td>
    <td><code>100</code></td>
  </tr>

  <tr>
    <td>DISTANCE_THRESHOLD_NONCBD</td>
    <td>float</td>
    <td>Clustering distance threshold outside the CBD.</td>
    <td><code>0.1</code></td>
  </tr>

  <tr>
    <td>DISTANCE_THRESHOLD_CBD</td>
    <td>float</td>
    <td>Clustering distance threshold within the CBD.</td>
    <td><code>0.05</code></td>
  </tr>

  <tr>
    <td>MAX_POP_THRESHOLD</td>
    <td>(optional)<br>list of int</td>
    <td>Demand point size thresholds for Colin's clustering approach.<br>Colin uses: <code>[200, 500, 5000, 15000, Infinity]</code><br>Default: <code>[25, 50, 75, 200, 500, 5000, 15000, Infinity]</code></td>
    <td><code>[25, 50, 75, 200, 500, 5000, 15000, Infinity]</code></td>
  </tr>

  <tr>
    <td>BUFFER_METERS</td>
    <td>(optional)<br>list of int or float</td>
    <td>Distance from demand points to merge when using Colin's clustering approach.<br>Must match MAXPOPTHRESHOLD in length.<br>Colin uses: <code>[250, 200, 150, 125, 100]</code><br>Default: <code>[1500, 1000, 500, 250, 200, 150, 125, 100]</code></td>
    <td><code>[1500, 1000, 500, 250, 200, 150, 125, 100]</code></td>
  </tr>

  <tr>
    <td>DEMAND_FACTOR</td>
    <td>(optional)<br>float</td>
    <td>Multiplier for all LODES pop sizes.</td>
    <td><code>2</code></td>
  </tr>

  <tr>
    <td>point_locs_to_move</td>
    <td>list of list of floats</td>
    <td>Coordinates of demand points to move.</td>
    <td><code>[[-77.69260, 43.29925], ...]</code></td>
  </tr>

  <tr>
    <td>moved_point_locs</td>
    <td>list of list of floats</td>
    <td>New coordinates for moved demand points.</td>
    <td><code>[[-77.69253, 43.29669], ...]</code></td>
  </tr>
</table>

## Airport‑related Parameters
<table>
  <tr>
    <th style="width:150px;">Parameter</th>
    <th style="width:100px;">Type</th>
    <th style="width:200px;">Description</th>
    <th>Example</th>
  </tr>

  <tr>
    <td>airport</td>
    <td>list of strings</td>
    <td>IATA codes for local airports; first uniquely identifies the city.</td>
    <td><code>["ROC"]</code></td>
  </tr>

  <tr>
    <td>airport_daily_passengers</td>
    <td>list of ints</td>
    <td>Daily passengers at each airport.</td>
    <td><code>[7000]</code></td>
  </tr>

  <tr>
    <td>airport_loc</td>
    <td>list of list of floats</td>
    <td>Coordinates of the city's airports.</td>
    <td><code>[[-77.67166, 43.12919]]</code></td>
  </tr>

  <tr>
    <td>airport_required_locs</td>
    <td>list of list of list</td>
    <td>Preferred residence locations for airport travelers.</td>
    <td><code>[[[-77.61298, 43.15729], ...]]</code></td>
  </tr>

  <tr>
    <td>air_pop_size_req</td>
    <td>list of ints</td>
    <td>Pop sizes assigned via airport_required_locs.</td>
    <td><code>[200]</code></td>
  </tr>

  <tr>
    <td>air_pop_size_remain</td>
    <td>list of ints</td>
    <td>Remaining airport pop sizes assigned automatically.</td>
    <td><code>[150]</code></td>
  </tr>
</table>

## University‑related Parameters
<table>
  <tr>
    <th style="width:150px;">Parameter</th>
    <th style="width:100px;">Type</th>
    <th style="width:200px;">Description</th>
    <th>Example</th>
  </tr>

  <tr>
    <td>universities</td>
    <td>list of strings</td>
    <td>Identifiers for each university.</td>
    <td><code>["UR","RIT","SJF","NU","RWU"]</code></td>
  </tr>

  <tr>
    <td>univ_loc</td>
    <td>list of list of floats</td>
    <td>Coordinates for each university's demand bubble.</td>
    <td><code>[[-77.62668, 43.12989], ...]</code></td>
  </tr>

  <tr>
    <td>univ_merge_within</td>
    <td>list of ints</td>
    <td>Merge distance (meters) for nearby demand points.</td>
    <td><code>[0, 350, 300, 0, 0]</code></td>
  </tr>

  <tr>
    <td>students</td>
    <td>list of ints</td>
    <td>Number of students at each campus.</td>
    <td><code>[11946, 17166, 4000, 2500, 1500]</code></td>
  </tr>

  <tr>
    <td>perc_oncampus</td>
    <td>list of floats</td>
    <td>Percent of students living on campus.</td>
    <td><code>[0.45, 0.4, 0.33, 0.5, 0.6]</code></td>
  </tr>

  <tr>
    <td>univ_pop_size</td>
    <td>list of ints</td>
    <td>Pop size created for each university.</td>
    <td><code>[75, 75, 75, 75, 75]</code></td>
  </tr>

  <tr>
    <td>univ_perc_travel</td>
    <td>list of list of floats</td>
    <td>Fraction of students [on‑campus, off‑campus] who travel daily.</td>
    <td><code>[0.3, 0.5]</code></td>
  </tr>
</table>

## Entertainment‑related Parameters
<table>
  <tr>
    <th style="width:150px;">Parameter</th>
    <th style="width:100px;">Type</th>
    <th style="width:200px;">Description</th>
    <th style="width:202px;">Example</th>
  </tr>

  <tr>
    <td>entertainment</td>
    <td>list of strings</td>
    <td>Identifiers for entertainment locations.</td>
    <td><code>["AQUA", "CY", "RAV"]</code></td>
  </tr>

  <tr>
    <td>ent_loc</td>
    <td>list of list of floats</td>
    <td>Coordinates for entertainment locations.</td>
    <td><code>[[-76.60832, 39.28540],
  [-76.62232, 39.28318],
  [-76.62269, 39.27800]]</code></td>
  </tr>

  <tr>
    <td>ent_merge_within</td>
    <td>list of ints</td>
    <td>Merge distance (meters) for entertainment demand points.</td>
    <td><code>[0, 0, 0]</code></td>
  </tr>

  <tr>
    <td>ent_req_residences</td>
    <td>list of list of list</td>
    <td>Like airport_required_locs. Required residence locations for entertainment visitors.</td>
    <td><code>[]</code></td>
  </tr>

  <tr>
    <td>ent_size</td>
    <td>list of ints</td>
    <td>Daily visitors to each entertainment location.</td>
    <td><code>[4110, 6247, 4384]</code></td>
  </tr>

  <tr>
    <td>ent_pop_size</td>
    <td>list of ints</td>
    <td>Pop size created for each entertainment location.</td>
    <td><code>[200, 200, 200]</code></td>
  </tr>

</table>

## Hotel‑related Parameters
<table>
  <tr>
    <th style="width:150px;">Parameter</th>
    <th style="width:100px;">Type</th>
    <th style="width:200px;">Description</th>
    <th style="width:202px;">Example</th>
  </tr>

  <tr>
    <td>hotels</td>
    <td>list of strings</td>
    <td>Identifiers for hotel locations.</td>
    <td></td>
  </tr>

  <tr>
    <td>hot_loc</td>
    <td>list of list of floats</td>
    <td>Coordinates for hotel locations.</td>
    <td></td>
  </tr>

  <tr>
    <td>hot_size</td>
    <td>list of ints</td>
    <td>Number of people staying at each hotel.</td>
    <td><code>[4110, 6247, 4384]</code></td>
  </tr>

  <tr>
    <td>hot_pop_size</td>
    <td>list of ints</td>
    <td>Pop size created for each hotel location.</td>
    <td><code>[200, 200, 200]</code></td>
  </tr>

  <tr>
    <td>hot_travel_split</td>
    <td>list of floats</td>
    <td>Fraction of hotel visitors that commute to [entertainment locations, airports] on an average day.</td>
    <td><code>[0.5, 0.5]</code></td>
  </tr>

</table>

## Military base‑related Parameters
<table>
  <tr>
    <th style="width:150px;">Parameter</th>
    <th style="width:100px;">Type</th>
    <th style="width:200px;">Description</th>
    <th style="width:202px;">Example</th>
  </tr>

  <tr>
    <td>bases</td>
    <td>list of strings</td>
    <td>Identifiers for base locations.</td>
    <td><code>["JFTB"]</code></td>
  </tr>

  <tr>
    <td>base_loc</td>
    <td>list of list of floats</td>
    <td>Coordinates for base locations.</td>
    <td><code>[[-76.60832, 39.28540],
  [-76.62232, 39.28318],
  [-76.62269, 39.27800]]</code></td>
  </tr>

  <tr>
    <td>base_merge_within</td>
    <td>list of ints</td>
    <td>Merge distance (meters) for base demand points.</td>
    <td><code>[0, 0, 0]</code></td>
  </tr>

  <tr>
    <td>personnel</td>
    <td>list of ints</td>
    <td>Number of employees that work at each base.</td>
    <td><code>[4110, 6247, 4384]</code></td>
  </tr>

  <tr>
    <td>base_pop_size</td>
    <td>list of ints</td>
    <td>Pop size created for each base location.</td>
    <td><code>[200, 200, 200]</code></td>
  </tr>

  <tr>
    <td>perc_onbase</td>
    <td>list of floats</td>
    <td>Percentage of personnel that live in on-base housing for each base.</td>
    <td><code>[0, 0.3, 0.1, 0.8]</code></td>
  </tr>

  <tr>
    <td>base_perc_travel</td>
    <td>list of floats</td>
    <td>Fraction of personnel that live [on base, off base] that travel on an average day.</td>
    <td><code>[0.4, 0.7]</code></td>
  </tr>

</table>

