from shapely.geometry import Point
import datetime
import pandas as pd

# used by read_st4.py and wrf_st4_stats.py 
def clean(df, shape, interval):
    if interval == "24h":
        # Drop these valid times
        # July 4 12z 24h accumulation has a row of pixels over 500mm and 1000mm near Lake Erie.
        if any([g.contains(Point(-81,43)) for g in shape]):
            df = df.drop(index=datetime.datetime(2015,7,3,12)) 
            df = df.drop(index=datetime.datetime(2015,7,4,12)) 
        # July 6-8 12z 24h accum has pixels over 800mm on northeast edge, northwest of Maine
        if any([g.contains(Point(-72,46)) for g in shape]):
            df = df.drop(index=datetime.datetime(2012,7,7,12)) 
            df = df.drop(index=datetime.datetime(2012,7,8,12)) 
            df = df.drop(index=datetime.datetime(2012,7,9,12)) 
            df = df.drop(index=datetime.datetime(2012,7,10,12))
        if any([g.contains(Point(-72,45)) for g in shape]):
            df = df.drop(index=datetime.datetime(2014,3,26,12)) 
            df = df.drop(index=datetime.datetime(2014,3,27,12)) 
            df = df.drop(index=datetime.datetime(2015,8,24,12)) 
        # 1 327.6mm pixel in northeast NC
        if any([g.contains(Point(-76.210,36.302)) for g in shape]):
            df = df.drop(index=datetime.datetime(2013,6,28,12)) 
        # 2 500+mm pixels in west TN
        if any([g.contains(Point(-89,36)) for g in shape]):
            df = df.drop(index=datetime.datetime(2013,3,19,12)) 
        # sharp blob in TN
        if any([g.contains(Point(-89,34)) for g in shape]):
            df = df.drop(index=datetime.datetime(2011,7,1,12)) 
    if interval == "01h":
        if any([g.contains(Point(-73.76,41.79)) for g in shape]):
            df = df.drop(index=datetime.datetime(2010,10,25,21)) 
            df = df.drop(index=datetime.datetime(2010,10,25,23)) 
            df = df.drop(index=datetime.datetime(2013,6,28,9)) 
        if any([g.contains(Point(-86,36.5)) for g in shape]):
            df = df.drop(index=datetime.datetime(2011,4,24,12)) 
            df = df.drop(index=datetime.datetime(2011,4,22,18))
        # Northeast Alabama pixels
        if any([g.contains(Point(-86,34.7)) for g in shape]):
            df = df.drop(index=datetime.datetime(2013,7,7,20))
        # New England pixels
        if any([g.contains(Point(-69.5,45.8)) for g in shape]):
            df = df.drop(index=datetime.datetime(2014,1,11,22))
    return df


