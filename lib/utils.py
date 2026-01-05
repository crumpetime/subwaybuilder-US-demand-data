import sys, os
import numpy as np


# Buffer sizing configuration based on point population thresholds - Colin's method
# These are the original, but it doesn't seem to merge enough of the small points
#maxPopThreshold = np.array([200, 500, 5000, 15000, np.inf])
#bufferMeters = np.array([250, 200, 150, 125, 100])
# These are my revised numbers, specifically targeted to ensure (almost) no points <50
maxPopThreshold = np.array([25, 50, 75, 200, 500, 5000, 15000, np.inf])
bufferMeters = np.array([1500, 1000, 500, 250, 200, 150, 125, 100])

def compute_centroid(coords, weights=None):
    if weights is None:
        weights = np.ones(len(coords))
    if not coords:
        return [0.0, 0.0]
    lon = float(weighted_mean([c[0] for c in coords], weights))
    lat = float(weighted_mean([c[1] for c in coords], weights))
    return [lon, lat]

def weighted_mean(values, weights):
    if not values:
        return 0.0
    w = np.array(weights, dtype=float)
    v = np.array(values, dtype=float)
    if w.sum() == 0:
        return float(np.mean(v))
    return float((v * w).sum() / w.sum())

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    From https://stackoverflow.com/a/4913653 w/ slight modifications
    """
    # convert decimal degrees to radians 
    lon1, lat1 = np.radians([lon1, lat1])
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.asin(np.sqrt(a)) 
    r = 6371000 # Radius of earth in meters. Use 3956 for miles. Determines return value units.
    return c * r

def in_cbd(loc, cbd_bbox=None):
    if cbd_bbox is not None:
        if ((loc[0] >= cbd_bbox[0]) and (loc[1] >= cbd_bbox[1]) and \
            (loc[2] <= cbd_bbox[2]) and (loc[3] <= cbd_bbox[3])):
            return True
    return False
