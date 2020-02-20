from shapely.geometry import Point
import datetime
import pandas as pd
import cartopy
import pdb

# used by read_st4.py and wrf_st4_stats.py
def clean(df, shapefile, interval, debug=False):

    shape = cartopy.io.shapereader.Reader(shapefile).geometries()
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
                if df.index.contains(dt):
                    if debug: print("st4.clean() dropping", dt, 'cause', p, 'artifact')
                    df = df.drop(index=dt)
                else:
                    if debug: print("st4.clean() no",dt,"in dataframe")


    return df


