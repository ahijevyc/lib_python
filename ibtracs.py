import argparse
import atcf
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


def get_atcfname_from_stormname_year(stormname, year, version="04r00_20200308"):
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

    df["valid_time"]   = pd.to_datetime(df["valid_time"])
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

    logging.info("Melt 34,50,64-kt windrad columns into multiple rows")
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
        




def get_df(version="04r00", basin=""):

    # Get "ALL" file by default. 
    # Specify basin keyword to read a smaller, more specialized file.

    dtype = {
            'BASIN':str,
            'NUMBER':str,
            'DS824_STAGE': str,
            'HKO_CAT'  : str,
            'LAT': float, 
            'LON': float, 
            'MLC_CLASS': str,
            'NEWDELHI_GRADE': str,
            'NEUMANN_CLASS': str, 
            'USA_AGENCY': str, 
            'USA_ATCF_ID': str, 
            'USA_GUST': float, 
            'USA_LAT': float, 
            'USA_LON': float, 
            'USA_RECORD': str, 
            'USA_SEAHGT': float,
            'USA_STATUS': str,
            'WMO_AGENCY':str
            }
    na_values = [' ','']

    basin = basin.lower()
    if basin == 'al':
        region = "NA" # North Atlantic
    elif basin == 'ep':
        region = "EP"
    elif basin == 'ni':
        region = "NI"
    elif basin == 'sa':
        region = "SA"
    elif basin == 'si':
        region = "SI"
    elif basin == 'sp':
        region = "SP"
    elif basin == 'wp':
        region = "WP"
    else:
        region = "ALL"
    ifile = os.path.join(idir,f'ibtracs.{region}.list.v{version}.csv')
    if not os.path.exists(ifile):
        url  = "https://www.ncei.noaa.gov/data/"
        url += "international-best-track-archive-for-climate-stewardship-ibtracs/"
        url += "v"+version+"/access/csv/ibtracs."+region+".list.v"+version+".csv"
        logging.debug(f"{ifile} not found. Downloading from {url}")
        myfile = requests.get(url)
        open(ifile, "wb").write(myfile.content)
    # keep_default_na=False . we don't want "NA" or North Atlantic to be treated as NA/NaN.
    # If skipinitialspace=True, add empty string to list of na_values. or you get TypeError. can't convert string to float64.
    logging.debug(f"read {ifile}")
    df = pd.read_csv(ifile, delimiter=',', skipinitialspace=True, header=[0,1], na_values=na_values, keep_default_na=False, low_memory=False, dtype=dtype)
    logging.debug(f"ibtracs.get_atcf(): read {len(df)} lines from {ifile}")
    column_units = {}
    for column, unit in df.columns:
        if unit[0:8] == 'Unnamed:':
            column_units[column] = None
        elif unit == 'nmile':
            column_units[column] = 'nautical_mile'
        elif unit == 'mb':
            column_units[column] = 'hPa'
        else:
            column_units[column] = unit.lower()

    df = df.droplevel(1, axis='columns') # TODO: don't drop level_1 It is units.

    # sanity check - are the wmo and usa lat/lons similar?

    assert (df["LAT"]-df["USA_LAT"]).abs().mean() < 0.1, "wmo and usa latitudes differ a lot"
    assert (np.sin(np.radians(df["LON"]))-np.sin(np.radians(df["USA_LON"]))).abs().mean() < 0.01, "wmo and usa longitudes differ a lot"
    assert (np.cos(np.radians(df["LON"]))-np.cos(np.radians(df["USA_LON"]))).abs().mean() < 0.01, "wmo and usa longitudes differ a lot"

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
    if not stormname:
        logging.error(f"ibtracs.get_stormname_from_atcfname(): no {bname} in {ifile}")
        sys.exit(1)

def extension(stormname, season):
    from metpy.units import units # this is so slow. Only used here.
    # capitalize stormnames in extension dictionary
    inkey = (stormname.upper(), int(season))
    # TODO: Grab last time, lat, lon from ibtracs, not hard coded values.
    # Need last entry from ibtracs to get speed and heading for all new members. speed_heading() needs position before.
    x = {("ISAAC",2012): # first element is last position from ibtracs, then tracked manually in 700mb wind in NARR
            {"valid_time": [pd.to_datetime(x) for x in ["20120901T06", "20120901T12", "20120901T15", "20120901T18", "20120901T21",
                                                  "20120902T00", "20120902T03", "20120902T06", "20120902T09"]],
             "lat"     : [ 38.4,  38.5,  38.7,  38.7,  38.6,  39.1,  38.7,  38.5,  38.9] * units.degree_N,
             "lon"     : [-93.3, -93.6, -93.1, -93.0, -92.0, -91.7, -90.9, -90.6, -89.7] * units.degree_E,
             "storm_size_S" : 1.0, # Does this make sense as a fill-in value? for Vt500km, it does
            }
        }

    if inkey in x:
        x[inkey]["stormname"] = stormname.upper()
        x[inkey]["SEASON"] = season
        speed, heading = atcf.speed_heading(x[inkey]["lon"], x[inkey]["lat"], x[inkey]["valid_time"])
        x[inkey]["speed"], x[inkey]["heading"] = speed, heading
        # throw away first element -- it was only needed to get speed and heading for 1st element of extension
        for c in ["valid_time", "lat", "lon", "speed", "heading"]:
            x[inkey][c] = x[inkey][c][1:]
        x[inkey]["rad"] = 0. # needed for check later where lines with rad > 35 are dropped.
        x = x[inkey]
    else:
        x = None

    return pd.DataFrame(x)

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
