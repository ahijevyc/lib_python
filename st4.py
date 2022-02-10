import cartopy.io.shapereader as shpreader
import datetime
import matplotlib.path
import numpy as np
import os
import pandas as pd
import pdb
from shapely.geometry import Point, multipolygon

def pts_in_shp(lats, lons, shp, debug=False):
    # Map longitude to -180 to +180 range
    lons = np.where(lons > 180, lons-360, lons)
    # If shp is a directory, point to .shp file of same name in it.
    shp = shp.rstrip("/")
    if os.path.isdir(shp):
        shp = shp + "/" + os.path.basename(shp) + ".shp"
    shape = shpreader.Reader(shp)
    ll_array = np.hstack((lons.flatten()[:,np.newaxis],lats.flatten()[:,np.newaxis]))
    mask = np.full(lats.flatten().shape, False)
    # How to make shapefile for EAST_CONUS (CONUS east of 105W)
    # import shapefile
    # import geopandas
    # from shapely.geometry import Polygon
    # shape = geopandas.read_file("./CONUS/CONUS.shp")
    # bbox = Polygon([(-105,65),(-50,65),(-50,10),(-105,10)])
    # shape = shape.intersection(bbox)
    # shape.to_file("EAST_CONUS")
    # It is as simple as that.

    # This seems kind of hacky. Can you recurse through a mixture of Polygons and Multipolygons more elegantly?
    # Tried geopandas read_shape . geometry but it was no more elegant.
    for g in shape.geometries():
        if debug:
            print(__name__, "pts_in_shp area", g.area)
        # How to deal with 3-D polygons (i.e. POLYGON Z)? some shape files are 3D.
        if g.has_z:
            print("Uh oh. shape geometry has z-coordinate in",shp)
            print("I don't know how to process 3-D polygons (i.e. POLYGON Z).")
            sys.exit(1)
        if isinstance(g, multipolygon.MultiPolygon):
            for mp in g.geoms:
                mask = mask | matplotlib.path.Path(mp.exterior.coords).contains_points(ll_array)
        else:
            mask = mask | matplotlib.path.Path(g.exterior.coords).contains_points(ll_array)
        if debug:
            print("pts_in_shp:", mask.sum(), "points")
    shape.close()
    return np.reshape(mask, lats.shape)



# used by read_st4.py and wrf_st4_stats.py
def clean(df, shapefile, interval, debug=False):

    shape = shpreader.Reader(shapefile).geometries()
    # geometries() returns an iterator, which AFAIK, can only be used once
    # unless you convert it to a list.
    shape = list(shape)
    
    # Drop valid times if there is an artifact in the shape


    # define artifacts dictionary where (lon, lat) is the key and
    # a list of times is the value
    if interval == "24h":
        # July 4 12z 24h accumulation has a row of pixels over 500mm and 1000mm near Lake Erie.
        artifacts = {
                (-81,43): [(2015,7, 3,12), (2015,7,4,12)],
                # July 6-8 12z 24h accum has pixels over 800mm on northeast edge, northwest of Maine
                (-72,46): [(2012,7, 7,12), (2012,7, 8,12), (2012,7, 9,12), (2012,7,10,12)], 
                (-72,45): [(2014,3,26,12), (2014,3,27,12), (2015,8,24,12)], 
                # 1 327.6mm pixel in northeast NC
                (-76.210,36.302): [(2013,6,28,12)], 
                # 2 500+mm pixels in west TN
                (-89,36): [(2013,3,19,12)],
                # sharp blob in TN
                (-89,34): [(2011,7,1,12)]
                }

    if interval == "01h":
        artifacts = {
                (-73.76,41.79): [(2010,10,25,21), (2010,10,25,23), (2013,6,28,9)],
                (-86,36.5)    : [(2011,4,24,12), (2011,4,22,18)],
                # Northeast Alabama pixels
                (-86,34.7)    : [(2013,7,7,20)],
                # New England pixels
                (-69.5,45.8)  : [(2014,1,11,22)]
                }

    for p,dts in artifacts.items():
        # if p is in the shape
        if any([g.contains(Point(p)) for g in shape]):
            for dt in dts:
                dt = datetime.datetime(*dt) # unpack tuple or get TypeError: an integer is required
                if dt in df.index:
                    if debug: print("st4.clean() dropping", dt, 'cause', p, 'artifact')
                    df = df.drop(index=dt)
                else:
                    if debug: print("st4.clean() no",dt,"in dataframe")


    return df


