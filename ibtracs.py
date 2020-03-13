import pandas as pd
import pdb
import re
import os
import numpy as np
from metpy import units
import sys
import requests

idir = '/glade/work/ahijevyc/share/ibtracs/'


def get_atcfname_from_stormname(stormname, version="04r00_20200308", debug=False):
    justname, year = stormname.split()
    justname = justname.upper()
    ifile = idir + "IBTrACS_SerialNumber_NameMapping_v"+version+".txt"
    if debug:
        print("ibtracs.get_atcfname_from_stormname(): searching for",stormname,"in",ifile)
    atcfname=None
    for line in open(ifile, "r"):
        if line[0:4] == year and re.search(justname, line):
            m = re.search(r" (b[a-z][a-z]\d\d[12]\d\d\d)\[atcf\]", line)
            atcfname = m.group(1)
            return atcfname
    if not atcfname:
        print("ibtracs.get_atcfname_from_stormame(): no",stormname,"in",ifile)
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
            "USA_LAT"       : "lat",
            "USA_LON"       : "lon",
            "USA_POCI"      : "pouter",
            "USA_ROCI"      : "router",
            "USA_PRES"      : "minp",
            "USA_RMW"       : "rmw",
            "USA_SEAHGT"    : "seas",
            "USA_SEARAD_NE" : "seas1",
            "USA_SEARAD_SE" : "seas2",
            "USA_SEARAD_SW" : "seas3",
            "USA_SEARAD_NW" : "seas4",
            "USA_WIND"      : "vmax",
            }
    df.rename(columns = column_match, inplace=True)


    df["valid_time"] = pd.to_datetime(df["valid_time"])
    df["initial_time"] = df["valid_time"]
    df["windcode"] = "NEQ"
    df["fhr"] = 0
    df["technum"] = np.nan
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
        




def get_atcf(atcfname, version="04r00", debug=False):
    dtype = {
            'BASIN':str,
            'DS824_STAGE': str,
            'HKO_CAT'  : str,
            'MLC_CLASS': str,
            'NEWDELHI_GRADE': str,
            'NEUMANN_CLASS': str, 
            'USA_AGENCY': str, 
            'USA_ATCF_ID': str, 
            'USA_GUST': str, 
            'USA_RECORD': str, 
            'USA_SEAHGT': str,
            'USA_STATUS': str,
            'WMO_AGENCY':str
            }
    na_values = ' '

    basin = atcfname[0:2]
    if basin.lower() == 'al':
        region = "NA" # North Atlantic
    elif basin.lower() == 'ep':
        region = "EP"
    elif basin.lower() == 'ni':
        region = "NI"
    elif basin.lower() == 'sa':
        region = "SA"
    elif basin.lower() == 'si':
        region = "SI"
    elif basin.lower() == 'sp':
        region = "SP"
    elif basin.lower() == 'wp':
        region = "WP"
    else:
        region = "ALL"
    ifile = idir+'ibtracs.'+region+'.list.v'+version+'.csv'
    if not os.path.exists(ifile):
        url = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v"+version+"/access/csv/ibtracs."+region+".list.v"+version+".csv"
        if debug:
            print(ifile,"not found. Downloading from", url)
        myfile = requests.get(url)
        open(ifile, "wb").write(myfile.content)
    df = pd.read_csv(ifile, delimiter=',', skipinitialspace=True, header=[0,1], na_values=na_values, dtype=dtype)
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
    imatch = df['USA_ATCF_ID'] == atcfname.upper()
    if imatch.sum() == 0:
        print("No",atcfname.upper(),"in ibtracs.")
        pdb.set_trace()
    df = df[imatch]
    df = ibtracs_to_atcf(df)
    if debug:
        print("ibtracs.get_atcf(): returning",len(df),"lines for",atcfname)
    return df

def get_stormname_from_atcfname(atcf_filename, version="04r00_20200308", debug=False):
    bname = os.path.basename(atcf_filename)
    assert bname[0:1] == "b"
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
