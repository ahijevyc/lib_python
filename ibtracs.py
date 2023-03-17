import argparse
import atcf
import cartopy.geodesic
from collections import defaultdict
from datetime import datetime, timedelta
import logging
import matplotlib
matplotlib.use("Agg") # Tried tkAgg for access to plt.show() interactivity but didn't work as user mpasrt with bad x11 connection
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb
import re
import requests
import sys

idir = '/glade/work/ahijevyc/share/ibtracs/'


def get_atcfname_from_stormname_year(stormname, year, version="04r00_20230314"):
    justname = stormname.upper()
    ifile = idir + "IBTrACS_SerialNumber_NameMapping_v"+version+".txt"
    logging.debug(f"ibtracs.get_atcfname_from_stormname_year(): searching for {stormname} {year} in {ifile}")
    atcfname=None
    for line in open(ifile, "r"):
        if line[0:4] == year and re.search(justname, line):
            m = re.search(r" (b[a-z][a-z]\d\d[12]\d\d\d)\[atcf\]", line)
            atcfname = m.group(1)
            return atcfname
    if not atcfname:
        logging.error(f"ibtracs.get_atcfname_from_stormame_year(): no {stormname} {year} in {ifile}")
        sys.exit(1)

def ibtracs_to_atcf(df):

    columns = {
            "BASIN"         : "basin",
            "ISO_TIME"      : "valid_time",
            "NUMBER"        : "cy",
            "NAME"          : "stormname",
            "NATURE"        : "ty",
            "STORM_DIR"     : "heading",
            "STORM_SPEED"   : "speed",
            "SUBBASIN"      : "subregion", # is SUBBASIN = subregion?
            "USA_EYE"       : "eye",
            "USA_GUST"      : "gusts",
            "LAT"           : "lat",
            "LON"           : "lon",
            "USA_POCI"      : "pouter",
            "USA_ROCI"      : "router",
            "WMO_PRES"      : "minp",
            "USA_RMW"       : "rmw",
            "USA_SEAHGT"    : "seas",
            "USA_SEARAD_NE" : "seas1",
            "USA_SEARAD_SE" : "seas2",
            "USA_SEARAD_SW" : "seas3",
            "USA_SEARAD_NW" : "seas4",
            "WMO_WIND"      : "vmax",
            }

    df = df.rename(columns = columns)

    # Fill in nan vmax from WMO_WIND with USA_WIND
    df["vmax"] = df["vmax"].fillna(df["USA_WIND"])

    df["valid_time"]   = pd.to_datetime(df["valid_time"], utc=True)
    df["initial_time"] = df["valid_time"]
    df["cy"]           = df["cy"].astype(str) # cy is not always an integer (e.g. 10E)
    df["technum"]      = np.nan
    df["model"]        = 'BEST'
    df["fhr"]          = 0
    df["windcode"]     = "NEQ"
    df["maxseas"]      = np.nan
    df["initials"]     = ""
    df["depth"]        = "X"
    df["seascode"]     = "NEQ"

    logging.debug("melt 34,50,64-kt windrad columns into multiple rows")
    # Rename USA_R34_NE -> rad1-34,
    #        USA_R34_SE -> rad2-34,
    #          ...
    #        USA_R64_NW -> rad4-64
    columns = {f"USA_R{rad}_{Q}":f"rad{q+1}-{rad}" for rad in [34,50,64] for q,Q in enumerate(["NE","SE","SW","NW"])}
    df["id"] = df.index # Keep track of original index
    df = pd.wide_to_long(df.rename(columns=columns), ["rad1","rad2","rad3","rad4"], i="id", j="rad", sep="-")
    # Bring "rad" MultiIndex level back into columns.
    df = df.reset_index()

    return df
        




def get_df(stormname=None, year=None, version="04r00", basin="ALL"):

    # Get "ALL" file by default. 
    # Specify basin keyword to read a smaller, specialized file.

    dtype = {
            'DS824_STAGE': str,
            'MLC_CLASS': str,
            'USA_RECORD': str, 
            }

    region = basin.upper()
    if region == "AL":
        region="NA" # North Atlantic
    ifile = os.path.join(idir,f'ibtracs.{region}.list.v{version}.csv')
    if not os.path.exists(ifile):
        logging.info(f"download {ifile}")
        url  = "https://www.ncei.noaa.gov/data/"
        url += "international-best-track-archive-for-climate-stewardship-ibtracs/"
        url += "v"+version+"/access/csv/ibtracs."+region+".list.v"+version+".csv"
        logging.debug(f"{ifile} not found. Downloading from {url}")
        myfile = requests.get(url)
        open(ifile, "wb").write(myfile.content)
    # keep_default_na=False . we don't want "NA" or North Atlantic to be treated as NA/NaN.
    # Skip row 1 (units)
    df = pd.read_csv(ifile, header=[0], skiprows=[1], na_values=[' '], keep_default_na=False, dtype=dtype)
    logging.debug(f"ibtracs.get_atcf(): read {len(df)} lines from {ifile}")

    if stormname and year: # optional keyword arguments
        imatch = (df['NAME'] == stormname.upper()) & (df['SEASON'] == year)
        assert imatch.sum(), (f"No {stormname} {year} in ibtracs {ifile}")
        df = df[imatch]

    df = extension(df)
    df = ibtracs_to_atcf(df)

    return df, ifile

def this_storm(df, stormname, year):
    return df.loc[(df['stormname'].str.upper() == stormname.upper()) & (df['SEASON'].astype(str) == str(year))]

def get_stormname_from_atcfname(atcf_filename, version="04r00_20200308"):
    bname = os.path.basename(atcf_filename)
    if bname.startswith("a"): # allow for adeck, but change name to start with "b" so the string may be found in IBTrACS file.
        bname = "b" + bname[1:]
    assert bname.startswith("b")
    assert bname[-4:] == ".dat"
    bname = bname[:-4]
    bname = bname+"\[atcf\]"
    ifile = idir + "IBTrACS_SerialNumber_NameMapping_v"+version+".txt"
    stormname = None
    logging.debug(f"ibtracs.get_stormname_from_atcfname(): searching for {bname} in {ifile}")
    for line in open(ifile, "r"):
        if re.search(bname, line):
            m = re.search(r" ([A-Z_]+)\[", line)
            stormname = m.group(1)
            year = line[0:4]
            logging.debug(f"{year} {bname}")
            assert year == bname[5:9]
            return stormname + " " + year
    assert stormname, f"ibtracs.get_stormname_from_atcfname(): no {bname} in {ifile}"

def extension(df):

    logging.info("use last line of ISAAC 2012 as template for extension")
    template = df[(df.SEASON == 2012) & (df.NAME == "ISAAC")].iloc[-1]
    # tracked manually in 700mb wind in NARR
    times = ["20120901T12", "20120901T15", "20120901T18", "20120901T21", "20120902T00", "20120902T03", "20120902T06", "20120902T09"]
    lats = [ 38.5,  38.7,  38.7,  38.6,  39.1,  38.7,  38.5,  38.9]
    lons = [-93.6, -93.1, -93.0, -92.0, -91.7, -90.9, -90.6, -89.7]
 
    newrows = [template]
    for time, lat, lon in zip(times, lats, lons):
        new = template.copy()
        new[["ISO_TIME","LAT","LON"]] = time, lat, lon
        newrows.append(new)

    # Fix new speed and headings. They were simply copied from last row of ISAAC
    newdf = pd.DataFrame(newrows)
    points = np.column_stack((newdf.LON,newdf.LAT))
    geo = cartopy.geodesic.Geodesic()
    n3 = geo.inverse(points[:-1], points[1:]) #returns np.ndarray shape=(n, 3) - dist in m, and forward azimuths of start and end pts.
    dist = n3[:,0]
    heading = n3[:,1] # forward azimuth of start points
    times = pd.to_datetime(newdf.ISO_TIME)
    dt = np.diff(times) / np.timedelta64(1,'s') # convert to seconds
    speed = dist/dt
    newdf = newdf.iloc[1:] # drop first row of new DataFrame. It was last row of ISAAC.
    # Convert to knots per https://www.ncei.noaa.gov/sites/default/files/2021-07/IBTrACS_v04_column_documentation.pdf
    newdf["STORM_SPEED"] = speed * 1.94384 # replace speeds with proper values. 
    newdf["STORM_DIR"] = heading # same with heading

    # Concatenate new rows to end of DataFrame. 
    # ignore_index=True, or else index from last line of ISAAC will appear multiple times and confuse ibtracs_to_atcf().
    df = pd.concat([df, newdf], axis="index", ignore_index=True)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stormname", help="storm name")
    parser.add_argument("season", help="season (year)")
    parser.add_argument("-b", "--basin", type=str, default="al", help="basin")
    parser.add_argument("-d", "--debug", action="store_true", help="debug mode")
    parser.add_argument("-o", "--outdir", default=".", help="Output path")
    args = parser.parse_args()
    stormname = args.stormname.upper()
    season = args.season
    png = os.path.realpath(os.path.join(args.outdir, stormname+season+".png"))
    dat = os.path.realpath(os.path.join(args.outdir, stormname+season+".dat"))
    if (os.path.exists(png)):
        logging.warning(f"found {png} Exiting")
        sys.exit(1)
    if (os.path.exists(dat)):
        logging.warning(f"found {dat} Exiting")
        sys.exit(1)
    ax = atcf.get_ax()
    track_df, ifile = get_atcf(stormname, season, basin=args.basin)
    atcf.write(dat, track_df)
    logging.info(f"created {dat}")
    start_label=""
    end_label=""
    atcf.plot_track(ax, start_label, track_df, end_label, label_interval_hours=12)
    plt.savefig(png)
    logging.warning(f"created {png}")

if __name__ == "__main__":
    main()
