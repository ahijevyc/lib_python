from datetime import datetime, timedelta
import argparse
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import pdb
import re
import atcf
import os
import numpy as np
from metpy import units
import sys
import requests

idir = '/glade/work/ahijevyc/share/ibtracs/'


def get_atcfname_from_stormname_year(stormname, year, version="04r00_20200308", debug=False):
    justname = stormname.upper()
    ifile = idir + "IBTrACS_SerialNumber_NameMapping_v"+version+".txt"
    if debug:
        print("ibtracs.get_atcfname_from_stormname_year(): searching for",stormname,year,"in",ifile)
    atcfname=None
    for line in open(ifile, "r"):
        if line[0:4] == year and re.search(justname, line):
            m = re.search(r" (b[a-z][a-z]\d\d[12]\d\d\d)\[atcf\]", line)
            atcfname = m.group(1)
            return atcfname
    if not atcfname:
        print("ibtracs.get_atcfname_from_stormame_year(): no",stormname,year,"in",ifile)
        sys.exit(1)

def ibtracs_to_atcf(df, debug=False):

    column_match = {
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
    df.rename(columns = column_match, inplace=True)



    df["valid_time"] = pd.to_datetime(df["valid_time"])
    df["initial_time"] = df["valid_time"]
    df["technum"] = np.nan
    df["model"] = 'BEST'
    df["fhr"] = 0
    df["windcode"] = "NEQ"
    df["maxseas"] = np.nan
    df["initials"] = ""
    df["depth"] = "X"
    df["seascode"] = "NEQ"

    #move 34,50,64-kt windrad columns into multiple rows
    id_vars = [x for x in df.columns]
    for r in ['34','50','64']:
        id_vars.remove('USA_R'+r+'_NE')
    # Turns USA_R34_NE, USA_R50_NE, and USA_R64_NE columns into mutiple rows
    # with the column name put in the row.
    # two more columns are added, "rad" and "rad1"
    # The column names go under "rad"
    # and the values of USA_R34_NE, USA_R50_NE, and USA_R64_NE go under "rad1".
    df = df.melt(id_vars=id_vars, var_name="rad", value_name="rad1")
    # change string values "USA_R34_NE", "USA_R50_NE", and "USA_R64_NE" to "34", "50", and "64".
    df.rad = df.rad.str[5:7]
    # make 3 new empty columns to hold SE, SW, and NW quadrants wind radii
    df["rad2"] = np.nan
    df["rad3"] = np.nan
    df["rad4"] = np.nan
    for r in ['34','50','64']:
        irad = df.rad == r
        df.loc[irad,"rad2"] = df[irad]["USA_R"+r+"_SE"]
        df.loc[irad,"rad3"] = df[irad]["USA_R"+r+"_SW"]
        df.loc[irad,"rad4"] = df[irad]["USA_R"+r+"_NW"]
        df.drop(columns=["USA_R"+r+"_SE", "USA_R"+r+"_SW", "USA_R"+r+"_NW"])

    #df.columns = map(str.lower, df.columns) # leads to duplicate lat and lon columns
    return df
        




def get_atcf(stormname, year, version="04r00", basin="", debug=False):

    # Get "ALL" file by default. 
    # Specify basin keyword to read a smaller, more specialized file.

    dtype = {
            'BASIN':str,
            'DS824_STAGE': str,
            'HKO_CAT'  : str,
            'LAT': float, 
            'LON': float, 
            'MLC_CLASS': str,
            'NEWDELHI_GRADE': str,
            'NEUMANN_CLASS': str, 
            'USA_AGENCY': str, 
            'USA_ATCF_ID': str, 
            'USA_GUST': str, 
            'USA_LAT': float, 
            'USA_LON': float, 
            'USA_RECORD': str, 
            'USA_SEAHGT': str,
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
    ifile = idir+'ibtracs.'+region+'.list.v'+version+'.csv'
    if not os.path.exists(ifile):
        url  = "https://www.ncei.noaa.gov/data/"
        url += "international-best-track-archive-for-climate-stewardship-ibtracs/"
        url += "v"+version+"/access/csv/ibtracs."+region+".list.v"+version+".csv"
        if debug:
            print(ifile,"not found. Downloading from", url)
        myfile = requests.get(url)
        open(ifile, "wb").write(myfile.content)
    # keep_default_na=False . we don't want "NA" or North Atlantic to be treated as NA/NaN.
    # If skipinitialspace=True, add empty string to list of na_values. or you get TypeError. can't convert string to float64.
    df = pd.read_csv(ifile, delimiter=',', skipinitialspace=True, header=[0,1], na_values=na_values, keep_default_na=False, dtype=dtype)
    if debug:
        print("ibtracs.get_atcf(): read",len(df),"lines from",ifile)
        pdb.set_trace()
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
    df = df.droplevel(1, axis='columns')

    #df = units.pandas_dataframe_to_unit_arrays(df, column_units=column_units) # creates a "united array" which is a dictionary. what use is that?
    imatch = (df['NAME'].str.upper() == stormname.upper()) & (df['SEASON'].astype(str) == str(year))
    if imatch.sum() == 0:
        print("No",stormname.upper(), year,"in ibtracs.")
        pdb.set_trace()
    df = df[imatch]

    # sanity check - are the wmo and usa lat/lons similar?
    assert (df["LAT"]-df["USA_LAT"]).abs().max() < 0.2
    assert (df["LON"]-df["USA_LON"]).abs().max() < 0.2

    df = ibtracs_to_atcf(df)
    if debug:
        print("ibtracs.get_atcf(): returning",len(df),"lines for",stormname,year)
    return df, ifile

def get_stormname_from_atcfname(atcf_filename, version="04r00_20200308", debug=False):
    bname = os.path.basename(atcf_filename)
    if bname.startswith("a"): # allow for adeck, but change name to start with "b" so the string may be found in IBTrACS file.
        bname = "b" + bname[1:]
    assert bname.startswith("b")
    assert bname[-4:] == ".dat"
    bname = bname[:-4]
    bname = bname+"\[atcf\]"
    ifile = idir + "IBTrACS_SerialNumber_NameMapping_v"+version+".txt"
    stormname = None
    if debug:
        print("ibtracs.get_stormname_from_atcfname(): searching for",bname,"in",ifile)
    for line in open(ifile, "r"):
        if re.search(bname, line):
            m = re.search(r" ([A-Z_]+)\[", line)
            stormname = m.group(1)
            year = line[0:4]
            if debug:
                print(year, bname)
            assert year == bname[5:9]
            return stormname + " " + year
    if not stormname:
        print("ibtracs.get_stormname_from_atcfname(): no",bname,"in",ifile)
        sys.exit(1)

def extension(stormname, season):
    # capitalize stormnames in extension dictionary
    inkey = (stormname.upper(), int(season))
    # TODO: Grab last time, lat, lon from ibtracs, not hard coded values.
    # Need last entry from ibtracs to get speed and heading for all new members. speed_heading() needs position before.
    x = {("ISAAC",2012): # first element is last position from ibtracs, then tracked manually in 700mb wind in NARR
            {"valid_time": [datetime(2012,9,1,6), datetime(2012,9,1,12), datetime(2012,9,1,15), datetime(2012,9,1,18), datetime(2012,9,1,21),
                                                  datetime(2012,9,2, 0), datetime(2012,9,2, 3), datetime(2012,9,2, 6), datetime(2012,9,2, 9)],
             "lat"     : [ 38.4,  38.5,  38.7,  38.7,  38.6,  39.1,  38.7,  38.5,  38.9],
             "lon"     : [-93.3, -93.6, -93.1, -93.0, -92.0, -91.7, -90.9, -90.6, -89.7],
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
    parser.add_argument("-o", "--out", default=".", help="Output path")
    args = parser.parse_args()
    stormname = args.stormname
    season = args.season
    ax = atcf.get_ax()
    track_df, ifile = get_atcf(stormname, season, basin=args.basin)
    start_label=""
    end_label=""
    atcf.plot_track(start_label, track_df, end_label)
    ofile = os.path.join(args.out, stormname+season+".png")
    if (os.path.exists(ofile)):
        print("found",ofile," Exiting")
    else:
        plt.savefig(ofile)
        print("created ",ofile)

if __name__ == "__main__":
    main()
