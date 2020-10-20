import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
from scipy.stats import circmean # for averaging longitudes
import pandas as pd
import datetime
import cartopy
import pdb
import re
import csv
import warnings # to suppress useless pandas warning when sorting tz-aware index
import os, sys
from netCDF4 import Dataset
from metpy.units import units
import numpy as np

# colors from tropicalatlantic.com
#                               TD         TD            TS           CAT1          CAT2          CAT3         CAT4         CAT5
#     index                      0          1             2             3             4             5            6            7
colors = ['white',(.01,.77,.18),(1.0,1.0,.65),(1.0,.85,.85),(1.0,.67,.67),(1.0,.45,.46),(1,.24,.24),(.85,.16,.17)]
# personal colors
colors = ['white',(.00,.61,.76),(.45,.75,.29),(.97,.94,.53),(.99,.75,.15),(.95,.46,.20),(1,.07,.07),(1.0,.09,.74)]

colors = {"NONTD": colors[0],
        "TD": colors[1],
        "TS": colors[2],
        "CAT1": colors[3],
        "CAT2": colors[4],
        "CAT3": colors[5],
        "CAT4": colors[6],
        "CAT5": colors[7]
        }
idl_water_color = np.array([235.,234.,242.])/255.


def basins():
    return {
        "al": (-99,-22,0,38),
        "Gulf": (-100,-80,21.5,34),
        "Irma": (-89,-70,20,34),
        "Irma1": (-81.1,-78.6,22.5,25.2),
        "ep": (-175,-94,0,34),
        "cp": (150,-135+360,0,34),
        "io": (30,109,0,28),
        "wp": (99,180,0,38),
        "global": (-180,180,-20,70),
        "track" : None # plot domain is simply the storm track
        }
    
    
def get_ax(projection=cartopy.crs.PlateCarree(), bscale='50m'):
    ax = plt.axes(projection=projection) 
    # bscale = countries and ocean the same border scale to match
    # Create a feature countries from Natural Earth
    countries = cartopy.feature.NaturalEarthFeature(category='cultural',
                name='admin_0_countries', scale=bscale,
                facecolor=cartopy.feature.COLORS['land'])

    ocean = cartopy.feature.NaturalEarthFeature(category='physical',
            name='ocean', scale=bscale, edgecolor='face',
            facecolor=idl_water_color) #cartopy.feature.COLORS['water'])

    ax.add_feature(ocean) 
    ax.add_feature(countries, edgecolor='gray', lw=0.375)
    ax.add_feature(cartopy.feature.LAKES, facecolor=idl_water_color)
    ax.add_feature(cartopy.feature.STATES, lw=0.25, edgecolor='gray')
    return ax

def kts2category(kts):
    category = "TD"
    if kts > 34:
        category = "TS"
    if kts > 64:
        category = "CAT1"
    if kts > 83:
        category = "CAT2"
    if kts > 96:
        category = "CAT3"
    if kts > 113:
        category = "CAT4"
    if kts > 137:
        category = "CAT5"

    return category, colors[category]


ifile = '/glade/work/ahijevyc/work/atcf/Irma.ECMWF.dat'
ifile = '/glade/scratch/mpasrt/uni/2018071700/latlon_0.500deg_0.25km/gfdl_tracker/tcgen/fort.64'

def vmax_HollandB_to_minp(vmax_kts, HollandB, environmental_pressure_hPa = 1013, density_of_air = 1.15*units["kg/m^3"], debug=False):
    vmax = vmax_kts * units["knots"].to("m/s")
    environmental_pressure = environmental_pressure_hPa* units["hPa"].to("Pa")
    minp = environmental_pressure - (vmax**2 * density_of_air * math.e / HollandB)
    return np.around(minp / 100.)

def archive_path(atcfname):
    assert atcfname[0:1] in ['a', 'b']
    basin = atcfname[1:3]
    cy = atcfname[3:5]
    year = atcfname[5:9]
    apath = "/glade/work/ahijevyc/atcf/archive/"+year+"/"+atcfname+".dat"  # provide path to atcf archive
    return apath 

def cyclone_phase_space_columns():
    names = []
    names.append('cpsB') # Cyclone Phase Space "Parameter B" for thermal asymmetry. (Values are *10)
    names.append('cpsll') # Cyclone Phase Space lower level (600-900 mb) thermal wind parameter, for diagnosing low-level warm core. (Values are *10)
    names.append('cpsul') # Cyclone Phase Space upper level (300-600 mb) thermal wind parameter, for diagnosing upper-level warm core. (Values are *10)
    return names 


def getcy(cys):
    # return numeric portion of string
    # tc_pairs doesn't match an adeck with cy=13L to a best track with cy=13
    return ''.join([i for i in cys if i.isdigit()])

def speed_heading(lon, lat, time):

    # return speed and heading from previous to present location
    # TODO: from previous halfway point to next halfway point
    speed = np.zeros_like(lon)
    heading = np.zeros_like(lon)

    # Allow lists
    # convert to pandas Series so we can apply the diff() method
    lon = pd.Series(lon)
    lat = pd.Series(lat)
    time = pd.Series(time)

    # 
    for i, (lon1, dlon, lat1, dlat, dt) in enumerate(zip(lon, lon.diff(),lat, lat.diff(), time.diff())):
        if dt < datetime.timedelta(seconds=0): # deal with times at beginning of track
            dt = datetime.timedelta(seconds=0)
        if pd.isnull(dlon) or pd.isnull(dt):
            continue
        d_km, head = dist_bearing(lon1-dlon,lat1-dlat,np.array(lon1),np.array(lat1))
        heading[i] = head
        d_km = d_km * units["km"]
        dt = dt.total_seconds() * units.s
        speed[i] = np.nan
        if dt != 0: # avoid RuntimeWarning: invalid value encountered in double_scalars /divide by zero encountered in double_scalars
            speed[i] = (d_km/dt).to("knots").m # return speed in knots

    return speed, heading


def get_stormname(df):
    # stormname column is blank prior to 2008-ish
    stormname = df.stormname[df.vmax == df.vmax.max()]
    stormname = stormname.iloc[0] # could be more than one match. take first one.

    # Sometimes stormname column at time with highest vmax is empty string after valid stormname. (e.g. Rita 2005)
    if stormname == "":
        stormname = df.stormname.replace("",np.nan).ffill()[df.vmax == df.vmax.max()]
        stormname = stormname.iloc[0] # could be more than one match. take first one.
    if not isinstance(stormname, str) and np.isnan(stormname):
        stormname = ""
    return stormname
    
def mean_track(df):

    # When aggregating numbers, take the mean; when aggregating strings or categories, take the max. Tried mode but it sometimes returned more than one value.
    agg_dict = {
            "technum"  : 'mean', 
            "lat"      : 'mean', 
            "lon"      : 'mean', 
            "vmax"     : 'mean',
            "minp"     : 'mean',
            "ty"         : pd.Series.max,
            "rad1"     : 'mean',
            "rad2"     : 'mean',
            "rad3"     : 'mean',
            "rad4"     : 'mean',
            "pouter"   : 'mean',
            "router"   : 'mean',
            "rmw"      : 'mean',
            "gusts"    : 'mean',
            "eye"      : 'mean',
            "subregion"  : pd.Series.max,
            "maxseas"  : 'mean',
            "initials"   : pd.Series.max,
            "heading"  : 'mean',
            "speed"    : 'mean',
            'stormname'  : pd.Series.max,
            'depth'      : pd.Series.max,
            "seas"     : 'mean',
            "seas1"    : 'mean',
            "seas2"    : 'mean',
            "seas3"    : 'mean',
            "seas4"    : 'mean',
            "userdefine1": pd.Series.max,
            "userdata1"  : pd.Series.max,
            }

    # Optional columns. If they are not in df, you get KeyError in aggregate.
    for ud in ["userdefine2", "userdata2", "userdefine3", "userdata3", "userdefine4", "userdata4"]:
        if ud in df:
            agg_dict[ud] = pd.Series.max
    for ud in ["fhr"]: # made fhr optional - it may be part of index
        if ud in df:
            agg_dict[ud] = 'mean'

    dfg = df.groupby(['basin','cy','initial_time','valid_time','windcode','rad','seascode'])
    df = dfg.agg(agg_dict)
    df.reset_index(inplace=True)
    df["model"] = "MEAN"
    return df

def contrasting_color(color):
    # color could be a size-3 tuple, or a string, like "white"
    assert mcolors.is_color_like(color)
    # to_rgba() returns a tuple, which has no attribute 'mean'
    if np.array(mcolors.to_rgba(color)).mean() >= 0.45:
        return 'black'
    return 'white'

def plot_track(start_label,group,end_label, scale=1, debug=False, label_interval_hours=1, **kwargs):
    if debug:
        print("plot_track: "+start_label)
        print(group)
    group = group.sort_values("valid_time")

    if plt.gca().projection.proj4_params["lon_0"] == 180:
        print("plot_track(): changing longitudes from -180,180 to 0,360")
        group.loc[group.lon<0,"lon"] += 360

    for i in range(0, len(group)):
        row = group.iloc[i]
        if i == 0:
            # first half-segment
            row_1 = group.iloc[i]
            if len(group) == 1: # track is 1-point-long
                row1 = group.iloc[i]
            else:
                row1 = group.iloc[i+1]
            plt.text(row.lon,row.lat, start_label, color='black', clip_on=True, 
                    ha='center', va='baseline', fontsize=7*scale, transform=cartopy.crs.PlateCarree())
        elif i == len(group)-1:
            # last half-segment
            row_1 = group.iloc[i-1]
            row1 = group.iloc[i]
            plt.text(row.lon, row.lat, end_label, color='black', clip_on=True,
                    ha='center', va='baseline', fontsize=7*scale, transform=cartopy.crs.PlateCarree())
        else:
            # middle segments
            row_1 = group.iloc[i-1]
            row1 = group.iloc[i+1]
        lat0 = (row_1.lat + row.lat) / 2.
        lon0 = (row_1.lon + row.lon) / 2. # TODO use circmean?
        lw = 2.5 if row.vmax > 34 else 1
        lat1 = (row.lat + row1.lat) / 2.
        lon1 = (row.lon + row1.lon) / 2.
        TCcategory, color = kts2category(row.vmax)
        if TCcategory == "TD" and row1.basin == 'TG':
            # IF this is TC genesis track and not a storm with TC vitals
            # use "OTHER (NON TD)" instead of "CLASSIFIED TD" color
            color = "white"
        if debug: print(i, "plot segment", [lon0,lon1],[lat0,lat1])
        segment = plt.plot([lon0,row.lon,lon1],[lat0,row.lat,lat1], # Include middle point. Full segment is a bent line.
                c=color, lw=lw*scale, transform=cartopy.crs.PlateCarree())
        if row.valid_time.hour % label_interval_hours == 0: # label if hour is multiple of label_interval_hours
            format = "%-d"
            if row.valid_time.hour != 0:
                format = "%-Hz"
            plt.text(row.lon, row.lat, row.valid_time.strftime(format), color=contrasting_color(color), clip_on=True,
                ha='center',va='center_baseline',fontsize=7*scale, transform=cartopy.crs.PlateCarree())
        markersize = scale*8 if row.valid_time.hour == 0 else scale*4
        plt.plot(row.lon, row.lat, 'o', markersize=markersize, markerfacecolor=color, color=color, transform=cartopy.crs.PlateCarree())
    label = "track label description"
    already_annotated = label in [x.get_label() for x in plt.gca().texts]
    if not already_annotated:
        plt.annotate(s = "day of month at hour 0", xy=(3,3), xycoords='axes pixels', fontsize=6, label=label)


def interpolate(df, interval, debug=False):

    if interval not in ['1H','3H','6H','12H','24H']:
        print("atcf.interpolate(): unexpected time interval",interval)
        print("Expected one of ['1H','3H','6H','12H','24H']")
        sys.exit(1)

    # Copied from ~ahijevyc/bin/interpolate_atcf.py on Mar 2, 2020. This method should supercede that script.

    if debug:
        print("atcf.interpolate():",interval)

    nbad = df.valid_time.isna().sum()
    if nbad > 0:
        if debug:
            print("Dropping",nbad,"wind radii line(s) with NaN valid_time")
        df.dropna(how='all', subset=['valid_time'], inplace=True)

    # Can we distinguish 34, 50 and 64-kt lines but treat 0kt and 34kt as the same?
    # Create dummy column "myrad" to do this. When grouping by myrad it keeps 0 and 34-kt lines together.
    df["myrad"] = df["rad"]
    df.loc[df["rad"].astype(int) == 0,"myrad"] = "34"


    df2 = pd.DataFrame()

    # Separate best track from model tracks.
    besttrack = df[df.model == 'BEST']
    othrtrack = df[df.model != 'BEST']

    # Interpolate in time

    if not besttrack.empty:
        # Used to group by rad before interpolating, but when available rads change with time,
        # treating them separately is incorrect
        # In case of best track, don't separate by initial_time. Put multiple initial_times into one group. They all have same fhr=0.
        for index, group in besttrack.groupby(['basin', 'cy', 'model']):
            if debug: print("atcf.interpolate(): setting index of",index,"to valid_time")
            x = group.set_index('valid_time')
            if debug: print("atcf.interpolate(): interpolating",index,"in time")
            with warnings.catch_warnings():
                # Pandas 0.24.1 emits useless warning when sorting tz-aware index
                warnings.simplefilter("ignore")
                x = x.resample(interval).interpolate(method='time') # TODO: understand FutureWarning about timezone-aware DatetimeArray
            if debug: print("atcf.interpolate(): padding",index,"and resetting index")
            x = x.fillna(method='pad').reset_index()
            # NaT in column not handled by df.interpolate().
            x["initial_time"] = x["valid_time"]
            # redo heading because of circular values get messed up in interpolation
            if debug:
                print("atcf.interpolate(): redo best track speed and heading")
            speed, heading = speed_heading(x.lon, x.lat, x.valid_time)
            x.loc[1:,"heading"] = heading[1:] # skip first element; it is always missing coming out of speed_heading()
            df2 = df2.append(x)

    # handle multiple models, init_times, etc.
    if not othrtrack.empty:
        for index, group in othrtrack.groupby(['basin', 'cy', 'initial_time', 'model', 'myrad']):
            # Used to say drop=False in set_index() method. but I want the interpolate() method to affect the valid_time
            # and for those interpolated times to be a column again by applying reset_index() method.
            if debug:
                pdb.set_trace()
            # TODO: expand all 0, 34, 50 and 64 knot lines so missing ones are properly treated as zero wind radii when interpolating.
            #x = return_expandedwindradii(group)

            x = group.set_index('valid_time').resample(interval).interpolate(method='time').reset_index()
            # TODO: do I need to fix initial_time column like I did for besttrack above? Yes, to avoid write afcf ValueError: Unknown format code 's' for object of type 'float'
            # initial_time is in the index of this group dataframe.
            # perhaps look at github.com/pandas-dev/issues/11701 for future hints.
            # redo heading because of circular values get messed up in interpolation
            if debug: print("atcf.interpolate(): padding",index,"and resetting index")
            x = x.fillna(method='pad').reset_index()
            if debug:
                print("atcf.interpolate(): redo model track circular heading")
            speed, heading = speed_heading(x.lon, x.lat, x.valid_time)
            x.loc[1:,"heading"] = heading[1:] # skip first element; it is always missing coming out of speed_heading()
            x = x.dropna(how='all', subset=['initial_time']) # I think this cleans up undefined extrapolated times
            df2 = df2.append(x)

    df2.drop(columns="myrad", inplace=True)
    df2.sort_values(by=['basin','cy','initial_time','model','valid_time','rad'], inplace=True)
    return df2

def stringlatlon2float(df, debug=False):
    # Extract last character of lat and lon columns
    # Multiply integer by -1 if "S" or "W"
    # Divide by 10
    S = df.lat.str[-1] == 'S'
    lat = df.lat.str[:-1].astype(float) / 10.
    lat[S] = lat[S] * -1
    df.lat = lat
    W = df.lon.str[-1] == 'W'
    lon = df.lon.str[:-1].astype(float) / 10.
    lon[W] = lon[W] * -1
    df.lon = lon

    if debug:
        print("finished converting string lat/lons to float")
        pdb.set_trace()
    return df



def return_expandedwindradii(df):
    # Set append=True to avoid losing columns when you make them an index
    df = df.set_index(['basin','cy','initial_time','model','fhr','rad'])
    df.index.set_levels(['0','34','50','64'], inplace=True, level='rad') # make sure full set of thresholds is defined. original dataframe may not have them all.
    mi = pd.MultiIndex.from_product(df.index.levels,names=df.index.names) # create MultiIndex with no missing rads.
    # TODO: Correctly use fill method option when applying reindex method.
    #       For example, with method='pad', it wrongly propagates forward previous time's values 
    #       for 0-knot line (because 0-knot line didn't exist). Tried method='nearest' but
    #       NotImplementedError: method='nearest' not implemented yet for MultiIndex; see GitHub issue 9365.
    df = df.reindex(mi) # now there is a 0,34,50, and 64-knot line for each entry.
    df = df.reset_index()  # and all the indexes are moved back to columns
    df.loc[:,["rad1","rad2","rad3","rad4"]] = df[["rad1","rad2","rad3","rad4"]].fillna(value=0) # Make missing wind radii zero.
    # Tried leaving as MultiIndex DataFrame but it led to all sorts of problems.
    return df

converters = {
        "basin" : lambda x: x.upper(), # official is capitalized
        # The problem with CY is ATCF only reserves 2 characters for it.
        "cy" : getcy, # cy is not always an integer (e.g. 10E) 
        "initial_time" : lambda x: pd.to_datetime(x.strip(),format='%Y%m%d%H'),
        #"vmax": float,
        #"minp": float,
        "ty": str, # why was this not in here for so long?
        "rad" : lambda x: x.strip(), # not a continuous value; it is a category. Important for interpolating in time.
        "windcode" : lambda x: x[-3:],
        "subregion": lambda x: x[-2:],
         # subregion ends up being 3 characters when written with .to_string
         # strange subregion only needs one character, but official a-decks leave 3. 
        "initials" : lambda x: x[-3:],
        'stormname': lambda x: x[-9:],
        'depth'    : lambda x: x[-1:],
        "seascode" : lambda x: x[-3:],
        "userdefine1": str,
        "userdata1"  : str,
        "userdefine2": str,
        "userdata2"  : str,
        "userdefine3": str,
        "userdata3"  : str,
        "userdefine4": str,
        "userdata4"  : str,
        }

dtype = {
        'technum'  : float, #Tried int, but int can't be nan. 
        "pouter"   : float,
        "rad1"     : float,
        "rad2"     : float,
        "rad3"     : float,
        "rad4"     : float,
        "router"   : float,
        'rmw'      : float,
        'gusts'    : float,
        'eye'      : float,
        'maxseas'  : float,
        "seas1"    : float,
        "seas2"    : float,
        "seas3"    : float,
        "seas4"    : float,
        'heading'  : float,
        'speed'    : float,
        "seas"     : float,
        } 

# Tried using converter for these columns, but couldn't convert 4-space string to float.
# If you add a key na_values, also add it to dtype dict, and remove it from converters.
na_values = {
        "technum"  : 3*' ',
        "rad1"     : 4*' ',
        "rad2"     : 4*' ',
        "rad3"     : 4*' ',
        "rad4"     : 4*' ',
        "pouter"   : 4*' ',
        "router"   : 4*' ',
        "rmw"      : 4*' ',
        "gusts"    : 4*' ',
        "eye"      : 4*' ',
        "maxseas"  : 4*' ',
        "heading"  : 4*' ',
        "speed"    : 4*' ',
        "seas"     : 3*' ', # one less than other columns
        "seas1"    : 4*' ',
        "seas2"    : 4*' ',
        "seas3"    : 4*' ',
        "seas4"    : 4*' ',
        "seas4"    : 4*' ',
        "HollandB" : 'Infinity',
        "B1"       : 'Infinity',
        "B2"       : 'Infinity',
        "B3"       : 'Infinity',
        "B4"       : 'Infinity',
        }

# Standard ATCF columns (doesn't include track id, like in fort.66).
# https://www.nrlmry.navy.mil/atcf_web/docs/database/new/abrdeck.html
# Updated format https://www.nrlmry.navy.mil/atcf_web/docs/database/new/abdeck.txt 
atcfcolumns=["basin","cy","initial_time","technum","model","fhr","lat","lon","vmax","minp","ty",
    "rad", "windcode", "rad1", "rad2", "rad3", "rad4", "pouter", "router", "rmw", "gusts", "eye",
    "subregion", "maxseas", "initials", "heading", "speed", "stormname", "depth", "seas", "seascode",
    "seas1", "seas2", "seas3", "seas4", "userdefine1", "userdata1", "userdefine2", "userdata2",
    "userdefine3", "userdata3", "userdefine4", "userdata4"]


# Return list of columns that are all empty.
def empty_cols(df, cols=["userdefine","userdata"]):
    empty_n = []
    for number in ["1","2","3","4"]:
        # append number string to column name
        # if all columns in cols array are empty for this numbered column, add this number to list of empties.
        these_cols = [col+number for col in cols]
        if df[these_cols].eq('').all(axis=None): # Tried all(df) but always evaluated to True.
            empty_n.append(number)
    return empty_n 


def add_missing_dummy_columns(df, columns, debug=False):
    for col in columns:
        if col not in df.columns:
            if debug:
                print(col, 'not in DataFrame. Fill with appropriate value.')
            # if column doesn't exist make it zeroes
            if col in ['rad1', 'rad2', 'rad3', 'rad4','pouter', 'router', 'seas', 'seas1','seas2','seas3','seas4']:
                df[col] = 0.

            # if rad column doesn't exist make it string zero.
            if col in ['rad']:
                df[col] = '0'

            # Initialize other default values.
            if col in ['windcode', 'seascode']:
                df[col] = '   '

            # Numbers are NaN
            if col in ['rmw','gusts','eye','maxseas','heading','speed']:
                df[col] = np.NaN

            # Strings are empty
            if col in ['subregion','stormname','userdefine1','userdata1','userdefine2','userdata2','userdefine3','userdata3','userdefine4','userdata4']:
                df[col] = ''

            if col in ['ty']:
                df[col] = 'XX'

            if col in ['initials', 'depth']:
                df[col] = 'X'
    return df        

def read_aswip(ifile = ifile, debug=False):
    # Read data into Pandas Dataframe
    if debug:
        print('Reading', ifile)

    # https://adcirc.org/home/documentation/users-manual-v50/input-file-descriptions/single-file-meteorological-forcing-input-fort-22/
    #           1     2         3            4        5      6     7     8     9     10     11
    names=["basin","cy","initial_time","technum","model","fhr","lat","lon","vmax","minp","ty",
    #     12      13         14      15       16     17       18        19       20     21       22
        "rad", "windcode", "rad1", "rad2", "rad3", "rad4", "pouter", "router", "rmw", "gusts", "eye",
    #       23          24          25         26         27        28      
        "subregion", "maxseas", "initials", "heading", "speed", "stormname",
    #           29                30         31      32      33      34      35       36       37       38       39
        "time_record_number", "nisotachs", "use1", "use2", "use3", "use4", "rmax1", "rmax2", "rmax3", "rmax4", "HollandB",
    #    40    41    42    43     44       45       46       47  
        "B1", "B2", "B3", "B4", "vmax1", "vmax2", "vmax3", "vmax4"]

    df = pd.read_csv(ifile,index_col=None,header=None, delimiter=",", names=names, converters=converters, 
        na_values=na_values, dtype=dtype, skipinitialspace=True, engine='c') # engine='c' is faster than engine="python"

    # convert string lat lon column to float. So we can write atcf file. valid_time not needed by write() method..
    df = stringlatlon2float(df)

    # Add missing ATCF columns with dummy data.
    df = add_missing_dummy_columns(df, atcfcolumns)

    # Derive valid time.   valid_time = initial_time + fhr (used in read_aswip too)
    df['valid_time'] = df.initial_time + pd.to_timedelta(df.fhr, unit='h')

    return df

def read(ifile = ifile, debug=False, fullcircle=False, expandwindradii=False):
    # Read data into Pandas Dataframe
    if debug:
        print('Reading', ifile, 'fullcircle=', fullcircle, 'expandwindradii=', expandwindradii)



    names = list(atcfcolumns) # make a copy of list, not a copy of the reference to the list.

    reader = csv.reader(open(ifile),delimiter=',')
    testline = next(reader)
    num_cols = len(testline)
    if debug:
        print("test line num_cols:", num_cols)
        print(testline)
    del reader
    with open(ifile) as f:
        max_num_cols = max(len(line.split(',')) for line in f)
        if debug:
            print("max number of columns", max_num_cols)

    # Output from GFDL vortex tracker, fort.64 and fort.66
    # are mostly ATCF format but have subset of columns
    if num_cols == 43:
        print('assume GFDL tracker fort.64-style output with 43 columns in', ifile)
        TPstr = "THERMO PARAMS"
        if testline[35].strip() != TPstr:
            print("expected 36th column to be", TPstr)
            print("got", testline[35].strip())
            sys.exit(4)
        for ii in range(20,35):
            names[ii] = "space filler" + str(ii-19) # duplicate names not allowed
        names = names[0:35]
        names.append(TPstr)
        names.extend(cyclone_phase_space_columns())
        names.append('warmcore')
        names.append("warmcore_strength")
        names.append("string1")
        names.append("string2")

    # fort.66 has track id in the 3rd column.
    if num_cols == 31:
        print('Assuming GFDL track fort.66-style with 31 columns in', ifile)
        # There is a cyclogenesis ID column for fort.66
        if debug:
            print('inserted ID for cyclogenesis in column 2 (zero-based)')
        names.insert(2, 'id') # ID for the cyclogenesis
        print('Using 1st 21 elements of names list')
        names = names[0:21]
        if debug:
            print('redefining columns 22-31')
        names.extend(cyclone_phase_space_columns())
        names.append('warmcore')
        names.append('heading')
        names.append('speedms')
        names.append('vort850mb')
        names.append('maxvort850mb')
        names.append('vort700mb')
        names.append('maxvort700mb')

    # TODO read IDL output
    if num_cols == 44 and 'min_warmcore_fract d' in testline[35]:
        print("Looks like IDL output")
        names = [n.replace('userdata1', 'min_warmcore_fract') for n in names]
        names.append('dT500')
        names.append('dT200')
        names.append('ddZ850200')
        names.append('rainc')
        names.append('rainnc')
        names.append('id')

    if num_cols == 11:
        print("Assuming", ifile,"is simple adeck with 11 columns")
        if ifile[-4:] != '.dat':
            print("even though file doesn't end in .dat", ifile)
        names = names[0:11]

    if len(names) > max_num_cols:
        names = names[0:max_num_cols]
    usecols = list(range(len(names)))

    # If you get a beyond index range (or something like that) error, see if userdata1 column is intermittent and has commas in it. 
    # If so, clean it up (i.e. truncate it)

    #atrack = ['basin','cy','initial_time','technum','model']
    #if 'id' in names:
    #    atrack.append('id')

    if debug:
        print("before pd.read_csv")
        for name,v in zip(names,testline):
            print(name+": "+v)
        print("converters=",converters)
        print("dype=", dtype)

    df = pd.read_csv(ifile,index_col=None,header=None, delimiter=",", usecols=usecols, names=names, 
            converters=converters, na_values=na_values, dtype=dtype, skipinitialspace=True, engine='c') # engine='c' is faster than engine="python"
    # fort.64 has asterisks sometimes. Problem with hwrf_tracker. 
    badlines = df['lon'].str.contains("\*")
    if any(badlines):
        df = df[~badlines]


    df = stringlatlon2float(df, debug=debug)

    if max_num_cols != num_cols and df.model.nunique() > 1:
        print("atcf.read(): test line has", num_cols, "columns, but another line has ", max_num_cols, "columns.")
        print("atcf.read(): may not handle different numbers of columns.")
        print("It's hard to deal with userdefined columns and data in a file with multiple types of models.")
        print("Exiting.")
        sys.exit(2)

    # Derive valid time.   valid_time = initial_time + fhr (used in read_aswip too)
    df['valid_time'] = df.initial_time + pd.to_timedelta(df.fhr, unit='h')
    # add minutes for BEST tracks. 2-digit minutes are in the TECHNUM column for BEST tracks. TECHNUM means something else for non-BEST tracks and shouldn't be added like a timedelta.
    besttracks = df[df.model == 'BEST']
    besttracks['valid_time']   +=  pd.to_timedelta(besttracks.technum.fillna(0), unit='minute')
    besttracks['initial_time'] +=  pd.to_timedelta(besttracks.technum.fillna(0), unit='minute')
    df[df.model == 'BEST'] = besttracks




    # Prior to 1999, rad column is blank. Fill with string zeros. 
    # Downstream programs assume rads are convertable to floats. Empty strings are not convertable to floats.
    if "rad" not in df.columns or all(df.rad == ''):
        if debug:
            print("atcf.read(): Empty rad column. This happens in pre-2000 files. Changing to string '0' so we can convert to float downstream")
        df["rad"] = '0'

    # sanity check for rad values
    if not all(df.rad.isin(['0','34','50','64'])):
        print("atcf.read(): unexpected rad value(s) in atcf file",ifile)
        print(df.rad.value_counts())
        # adecks before 2002 had 35-knot lines, not 34. That's okay. 
        if df.valid_time.min().year > 2001:
            print("atcf.read(): this should not happen with post-2001 files. Exiting.")
            sys.exit(1)

    # Add missing ATCF columns with dummy data.
    df = add_missing_dummy_columns(df, atcfcolumns)

    if fullcircle:
        if debug:
            print("full circle wind radii")
        # Full circle wind radii instead of quadrants
        df['windcode'] = 'AAA'
        df['rad1'] = df[['rad1','rad2','rad3','rad4']].max(axis=1)
        df['rad2'] = 0
        df['rad3'] = 0
        df['rad4'] = 0


    if expandwindradii:
        df = return_expandedwindradii(df)

    fill_speed_heading = all(((df.speed == 0) | pd.isnull(df.speed)) & ((df.heading == 0) | pd.isnull(df.heading))) # assume bad if everything is zero
    if fill_speed_heading:
        if debug:
            print("Deriving speed and heading")
        speed, heading = speed_heading(df.lon, df.lat, df.valid_time)
        df.loc[:,"speed"] = speed
        df.loc[:,"heading"] = heading

    return df

def f2s(x): # float to string
    # Convert absolute value of float to integer number of tenths for ATCF lat/lon
    # called by lat2s and lon2s
    x *= 10
    x = np.around(x)
    x = np.abs(x)
    return str(int(x))

def lat2s(lat):
    NS = 'N' if lat >= 0 else 'S'
    lat = f2s(lat) + NS
    return lat

def lon2s(lon):
    EW = 'E' if lon >= 0 else 'W'
    lon = f2s(lon) + EW
    return lon

# function to compute great circle distance between point lat1 and lon1 and arrays of points 
# INPUT:
#   lon1 - longitude of origin
#   lat1 - latitude of origin
#   lons - longitudes of points to get distance to
#   lats - latitudes of points to get distance to
# Returns 2 things:
#   1) distance in km
#   2) initial bearing from 1st pt (lon1, lat1) to an array of other points (lons, lats).
def dist_bearing(lon1,lat1,lons,lats,debug=False):
    assert lat1 < 90, "lat1 > 90"
    assert lat1 > -90, "lat1 < -90"
    assert lats.max() < 90, "lats element > 90"
    assert lats.min() > -90, "lats element < -90"
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lons = np.radians(lons)
    lats = np.radians(lats)
    # great circle distance. 
    arg = np.sin(lat1)*np.sin(lats)+np.cos(lat1)*np.cos(lats)*np.cos(lon1-lons)
    #arg = np.where(np.fabs(arg) < 1., arg, 0.999999) # sometimes arg = 1.000000000000002
    if (np.fabs(arg) > 1).any():
        if debug:
            print("atcf.dist_bearing(): minarg=",arg.min(),"maxarg=",arg.max())

    arg = np.where(arg <=  1., arg,  1.) # sometimes arg = 1.000000000000002 
    arg = np.where(arg >= -1., arg, -1.) 

    dlon = lons-lon1
    bearing = np.arctan2(np.sin(dlon)*np.cos(lats), np.cos(lat1)*np.sin(lats) - np.sin(lat1)*np.cos(lats)*np.cos(dlon)) 

    # convert from radians to degrees
    bearing = np.degrees(bearing)

    # -180 - 180 -> 0 - 360
    bearing = (bearing + 360) % 360
    
    # Ellipsoid [CLARKE 1866]  Semi-Major Axis (Equatorial Radius)
    a = 6378.2064

    if (np.fabs(arg) > 1).any():
        print("atcf.dist_bearing(): arg=",arg)
        pdb.set_trace()
        
    return np.arccos(arg)* a, bearing 


ms2kts = 1 * units["m/s"].to("knots").magnitude # 1.94384
km2nm  = 1 * units["km"].to("nautical_mile").magnitude # 0.539957

quads = {'NE':0, 'SE':90, 'SW':180, 'NW':270}
thresh_kts = np.array([34, 50, 64])

def get_azimuthal_mean(x, distance_km, binsize_km = 25.):
    radius = np.arange(0, max(distance_km), binsize_km)
    x_vs_radius = [] # maybe should be numpy array but can't remember syntax
    for r in radius:
        i = (r <= distance_km) & (distance_km < r+binsize_km)
        npts = np.sum(i)
        if npts == 0:
            print("get_azimuthal_mean: no pts b/t", r, "and", r+binsize_km)
            sys.exit(1)
        if npts == 1:
            print("get_azimuthal_mean: only " + "%d" % npts + " grid cell b/t", r, "and", r+binsize_km)
        x_vs_radius.append(np.mean(x[i]))
    return x_vs_radius, radius

def get_ext_of_wind(speed_kts, distance_km, bearing, raw_vmax_kts, quads=quads, thresh_kts=thresh_kts, 
        rad_search_radius_nm=300., lonCell=None, latCell=None, debug=False, wind_radii_method='max'):
    
    # speed_kts is converted to masked array. Masked where distance >= 300 nm
    wind_radii_nm = {"wind_radii_method":wind_radii_method}
    # Put in dictionary "wind_radii_nm" where
    # wind_radii_nm = {
    #           34: {'NE':rad1, 'SE':rad2, 'SW':rad3, 'NW':rad4},
    #           50: {'NE':rad1, 'SE':rad2, 'SW':rad3, 'NW':rad4},
    #           64: {'NE':rad1, 'SE':rad2, 'SW':rad3, 'NW':rad4} 
    #          }

    rad_search_radius_km = rad_search_radius_nm / km2nm

    wind_radii_nm['raw_vmax_kts'] = raw_vmax_kts
    wind_radii_nm['thresh_kts'] = thresh_kts
    wind_radii_nm['quads'] = quads

    # Originally had distance_km < 800, but Chris D. suggested 300nm in Sep 2018 email
    # This was to deal with Irma and the unrelated 34 knot onshore flow in Georgia
    # Looking at HURDAT2 R34 sizes (since 2004), ex-tropical storm Karen 2015 had 710nm.
    # Removing EX-tropical storms, the max was 480 nm in Hurricane Sandy 2012
    # see /glade/work/ahijevyc/atcf/R34noEX.png and R34withEX.png
    speed_kts = np.ma.array(speed_kts, mask = distance_km >= rad_search_radius_km)

    for wind_thresh_kts in thresh_kts[thresh_kts < raw_vmax_kts]:
        ithresh = speed_kts >= wind_thresh_kts # Boolean array same shape
        imax = np.argmax(distance_km * ithresh) # arg must be same shape as subscript target
        iedge = np.unravel_index(imax, distance_km.shape)
        # warn if max_dist_of_wind_threshold is on edge of 2-d domain (like nested WRF grid)
        if distance_km.ndim == 2:
            if debug:
                print("imax:", imax)
                print("iedge:", iedge)
            if iedge[0] == distance_km.shape[0]-1 or iedge[1] == distance_km.shape[1]-1 or any(iedge) == 0:
                print("get_ext_of_wind(): R"+str(wind_thresh_kts)+" at edge of domain",iedge,"shape:",distance_km.shape)
        wind_radii_nm[wind_thresh_kts] = {}
        if debug:
            print('get_ext_of_wind(): method ' + wind_radii_method)
            print('get_ext_of_wind(): kts quad azimuth npts    dist   bearing     lat     lon')
        for quad,az in quads.items():
            # Compute azimuthal mean
            if wind_radii_method == "azimuthal_mean":
                # I thought I wouldn't need (distance_km < rad_search_radius_km) because speed_kts was masked beyond
                # the search radius. But it makes a difference.
                iquad = (az <= bearing) & (bearing < az+90) & (distance_km < rad_search_radius_km)
                speed_kts_vs_radius_km, radius_km = get_azimuthal_mean(speed_kts[iquad], distance_km[iquad], binsize_km = 25.)
                wind_radii_nm[wind_thresh_kts][quad] = 0.
                if any(speed_kts_vs_radius_km >= wind_thresh_kts):
                    max_dist_of_wind_threshold_nm = np.max(radius_km[speed_kts_vs_radius_km >= wind_thresh_kts]) * km2nm
                    wind_radii_nm[wind_thresh_kts][quad] = max_dist_of_wind_threshold_nm
                    if debug:
                        print('get_ext_of_wind():', "%3d "%wind_thresh_kts, quad, ' %3d-%3d'%(az,az+90), '%4d'%np.sum(iquad), 
                              '%6.2fnm'%max_dist_of_wind_threshold_nm, end="")
                        print(radius_km)
                        print(speed_kts_vs_radius_km)
                        print()
            else:
                # I thought I wouldn't need (distance_km < rad_search_radius_km) because speed_kts was masked beyond
                # the search radius. But it makes a difference.
                iquad = (az <= bearing) & (bearing < az+90) & (speed_kts >= wind_thresh_kts) & (distance_km < rad_search_radius_km)
                wind_radii_nm[wind_thresh_kts][quad] = 0.
                if np.sum(iquad) > 0:
                    x_km = distance_km[iquad]
                    if wind_radii_method[-10:] == "percentile":
                        # assume wind_radii_method is a number followed by the string "percentile".
                        distance_percentile = float(wind_radii_method[:-10])
                        # index of array entry nearest to percentile value
                        idist_of_wind_threshold=abs(x_km-np.percentile(x_km,distance_percentile,interpolation='nearest')).argmin()
                        wind_radii_nm[wind_thresh_kts][quad] = np.percentile(x_km, distance_percentile) * km2nm
                    elif wind_radii_method == "max":
                        idist_of_wind_threshold = np.argmax(x_km)
                        wind_radii_nm[wind_thresh_kts][quad] = np.max(x_km) * km2nm
                    else:
                        print("unexpected wind_radii_method:" + wind_radii_method)
                        sys.exit(1)
                    if debug:
                        print('get_ext_of_wind():', "%3d "%wind_thresh_kts, quad, ' %3d-%3d'%(az,az+90), '%4d'%np.sum(iquad), 
                              '%6.2fnm'%wind_radii_nm[wind_thresh_kts][quad], '%4.0fdeg'%bearing[iquad][idist_of_wind_threshold], end="")
                        if lonCell is not None:
                            print('%8.2fN'%latCell[iquad][idist_of_wind_threshold], '%7.2fE'%lonCell[iquad][idist_of_wind_threshold], end="")
                        print()
    return wind_radii_nm


def derived_winds(u10, v10, mslp, lonCell, latCell, row, vmax_search_radius=250., mslp_search_radius=100., 
        wind_radii_method="max", pouter_search_radius=800, debug=False):

    # Given a row (with row.lon and row.lat)...

    # Derive cell distances and bearings
    distance_km, bearing = dist_bearing(row.lon, row.lat, lonCell, latCell)

    # Derive 10m wind speed and Vt from u10 and v10
    speed_kts = np.sqrt(u10**2 + v10**2) * ms2kts

    # Tangential (cyclonic) wind speed
    # v dx - u dy 
    dx = lonCell - row.lon
    # work on the dateline?
    dx[dx>=180] = dx[dx>=180]-360.
    dy = latCell - row.lat
    Vt = v10 * dx - u10 * dy
    if row.lat < 0:
        Vt = -Vt

    # Restrict Vmax search to a certain radius (vmax_search_radius)
    vmaxrad = distance_km < vmax_search_radius
    ispeed_max = np.argmax(speed_kts[vmaxrad])
    raw_vmax_kts =  speed_kts[vmaxrad].max()

    # If vmax > 17, check if tangential component of max wind is negative (anti-cyclonic)
    if row.vmax > 17 and Vt[vmaxrad][ispeed_max] < 0:
        print("center", row.valid_time, row.lat, row.lon)
        print("max wind is anti-cyclonic! (unknown units)", Vt[vmaxrad][ispeed_max])
        print("max wind lat/lon", latCell[vmaxrad][ispeed_max], lonCell[vmaxrad][ispeed_max])
        print("max wind U/V",         u10[vmaxrad][ispeed_max],     v10[vmaxrad][ispeed_max])
        if debug: pdb.set_trace()

    # Check if average tangential wind within search radius is negative (anti-cyclonic)
    average_tangential_wind = np.average(Vt[vmaxrad])
    if average_tangential_wind < 0:
        print("center", row.valid_time, row.lat, row.lon)
        print("avg wind is anti-cyclonic!", average_tangential_wind)
        if debug: pdb.set_trace()

    # Get radius of max wind
    raw_RMW_nm = distance_km[vmaxrad][ispeed_max] * km2nm
    if debug:
        print('max wind lat', latCell[vmaxrad][ispeed_max], 'lon', lonCell[vmaxrad][ispeed_max])

    # Restrict min mslp search
    mslprad = distance_km < mslp_search_radius
    raw_minp = mslp[mslprad].min() / 100.

    # Get max extent of wind at thresh_kts thresholds.
    wind_radii_nm = get_ext_of_wind(speed_kts, distance_km, bearing, raw_vmax_kts, latCell=latCell, lonCell=lonCell, wind_radii_method=wind_radii_method, debug=debug)

    # Restrict pressure of last closed isobar and radius of last closed isobar search to a certain radius
    pouter_rad = distance_km < pouter_search_radius
    imslp_max = np.argmax(mslp[pouter_rad])
    raw_pouter_mb =  mslp[pouter_rad][imslp_max] / 100.
    assert raw_pouter_mb == mslp[pouter_rad].max() / 100.
    raw_router_nm =  distance_km[pouter_rad][imslp_max] * km2nm

    return raw_vmax_kts, raw_RMW_nm, raw_minp, wind_radii_nm, raw_pouter_mb, raw_router_nm


def add_wind_rad_lines(row, wind_radii_nm, fullcircle=False, debug=False):
    raw_vmax_kts = wind_radii_nm['raw_vmax_kts']
    thresh_kts = wind_radii_nm['thresh_kts']
    # if not empty...must be NEQ
    if row.windcode.strip() and row.windcode != 'NEQ':
        print('bad windcode', row.windcode, 'in', row)
        print('expected NEQ')
        sys.exit(1)
    lines = pd.DataFrame()
    for thresh in thresh_kts[thresh_kts < raw_vmax_kts]:
        if any(wind_radii_nm[thresh].values()):
            newrow = row.copy()
            # not sure how to do this with "rad" as part of the DataFrame index or Series name
            #if row.fhr > 110: #  Why? Doesn't 2017090512 initialization go past 110?
            #    pdb.set_trace()
            newrow['rad'] = str(thresh) # needs to be string in dataframe. It is written as a string in write() method. it is a category not a float.
            if fullcircle:
                # Append row with full circle 34, 50, or 64 knot radius
                # MET-TC will not derive this on its own - see email from John Halley-Gotway Oct 11, 2018
                # Probably shouldn't have AAA and NEQ in same file. 
                newrow[['windcode','rad1']] = ['AAA',np.nanmax(list(wind_radii_nm[thresh].values()))] 
                newrow[['rad2','rad3','rad4']] = np.nan
            else:
                newrow['windcode'] = 'NEQ' 
                newrow[['rad1','rad2','rad3','rad4']] = [
                        wind_radii_nm[thresh]['NE'],
                        wind_radii_nm[thresh]['SE'],
                        wind_radii_nm[thresh]['SW'],
                        wind_radii_nm[thresh]['NW']
                        ]
            # Append row with 34, 50, or 64 knot radii
            lines = lines.append(newrow)
    
    return lines


def update_df(df, initial_time, model, fhr, raw_vmax_kts, raw_RMW_nm, raw_minp, wind_radii_nm, raw_pouter_mb=None,
        raw_router_nm=None, gridfile=None, debug=False):

    # Called by origgrid and origmesh
   
    i = (df["initial_time"] == initial_time) & (df["model"] == model) & (df["fhr"] == fhr)
    # Make sure is True in only one place
    assert i.sum() == 1

    if debug:
        print("atcf.update_df(): before update")
        print(df.loc[i, ['valid_time','lon','lat', 'vmax', 'minp', 'rmw', 'pouter','router']]) 
    df.loc[i, "vmax"] = raw_vmax_kts
    df.loc[i, "minp"] = raw_minp
    df.loc[i, "rmw"]  = raw_RMW_nm
    if raw_pouter_mb:
        df.loc[i, "pouter"] = raw_pouter_mb
    if raw_router_nm:
        df.loc[i, "router"] = raw_router_nm
    # stop for debugging if "origmesh" string found in one of the user defined columns
    for n in ["1", "2", "3", "4"]:
        for col in ["userdefine"+n, "userdata"+n]:
            if col in df.columns and 'origmesh' in df.loc[i, col]:
                print("wait. this row already has original mesh values")
                pdb.set_trace()
    # Add note of original mesh = True in user data (not defined) column
    df.loc[i, "userdefine1"] = 'origmesh wind_radii_method'
    df.loc[i,"userdata1"]   = wind_radii_nm["wind_radii_method"]
    if debug:
        print("appended original mesh wind radii method to atcf row")
    if gridfile is not None:
        # Append origmesh file to userdata1 column (after a comma)
        df.loc[i,"userdefine2"] = 'originalmeshfile'
        if not os.path.isfile(os.path.realpath(gridfile)):
            print("atcf.update_df(): no gridfile",gridfile)
            sys.exit(1)
        df.loc[i,"userdata2"]   = os.path.realpath(gridfile)
        if debug:
            print("appended original mesh filename to atcf row")
    if debug:
        print('after')
        print(df.loc[i, ['vmax', 'minp', 'rmw', 'pouter', 'router']]) 

    # Append 34/50/64 knot lines to DataFrame
    # add_wind_rad_lines expects a series, so apply squeeze()
    newlines = add_wind_rad_lines(df.loc[i,:].squeeze(), wind_radii_nm, debug=debug)
    # If there are new lines, drop the old one and append new ones.
    if not newlines.empty:
        if debug:
            print("dropping old row")
        df = df[~i]
        if debug:
            print("appending")
            print(newlines)
        df = df.append(newlines, sort=False) 
    # Sort DataFrame by index (deal with appended wind radii lines)
    # sort by rad too. I tried to avoid this, but when rad=0, it would be left behind other fhrs that had rad>0.
    df = df.sort_index().sort_values(['initial_time','fhr','rad'])

    return df





def write(ofile, df, fullcircle=False, append=False, debug=False):
    if df.empty:
        print("afcf.write(): DataFrame is empty.")

    # TODO: deal with fullcircle.

    if debug:
        print("writing to", ofile)
        print(df.head(0))
        pdb.set_trace()

    atcf_lines = ""
    for index, row in df.iterrows():
        atcf_lines += "{:2s}, ".format(row.basin) 
        atcf_lines += "{:2s}, ".format(row.cy.zfill(2))
        atcf_lines += "{:8s}, ".format(row.initial_time.strftime('%Y%m%d%H'))
        atcf_lines += "  , " if np.isnan(row.technum) else "{:02.0f}, ".format(row.technum) # avoid turning 'nan' into 3 spaces below
        atcf_lines += "{}, ".format(row.model)
        atcf_lines += "{:3.0f}, ".format(row.fhr)
        atcf_lines += "{:>4s}, ".format(lat2s(row.lat))
        atcf_lines += "{:>5s}, ".format(lon2s(row.lon))
        atcf_lines += "{:3.0f}, ".format(row.vmax)
        atcf_lines += "{:4.0f}, ".format(row.minp)
        atcf_lines += "{:2s}, ".format(row.ty)
        atcf_lines += "{:>3s}, ".format(row.rad)
        atcf_lines += "{:>3s}, ".format(row.windcode)
        atcf_lines += "{:4.0f}, ".format(row.rad1)
        atcf_lines += "{:4.0f}, ".format(row.rad2)
        atcf_lines += "{:4.0f}, ".format(row.rad3)
        atcf_lines += "{:4.0f}, ".format(row.rad4)
        atcf_lines += "{:4.0f}, ".format(row.pouter)
        atcf_lines += "{:4.0f}, ".format(row.router)
        atcf_lines += "{:3.0f}, ".format(row.rmw)
        atcf_lines += "{:3.0f}, ".format(row.gusts)
        atcf_lines += "{:3.0f}, ".format(row.eye)
        atcf_lines += "{:>3s}, ".format(row.subregion) # supposedly 1 character, but always 3 in official b-decks
        atcf_lines += "{:3.0f}, ".format(row.maxseas)
        atcf_lines += "{:>3s}, ".format(row.initials)
        atcf_lines += "{:3.0f}, ".format(row.heading)
        atcf_lines += "{:3.0f}, ".format(row.speed)
        atcf_lines += "{:>10s}, ".format(row.stormname)
        atcf_lines += "{:>1s}, ".format(row.depth)
        atcf_lines += "{:2.0f}, ".format(row.seas)
        atcf_lines += "{:>3s}, ".format(row.seascode)
        atcf_lines += "{:4.0f}, ".format(row.seas1)
        atcf_lines += "{:4.0f}, ".format(row.seas2)
        atcf_lines += "{:4.0f}, ".format(row.seas3)
        atcf_lines += "{:4.0f}, ".format(row.seas4)
        atcf_lines += "{:s}, ".format(row.userdefine1) # Described as 1-20 chars in atcf doc. 
        atcf_lines += "{:s}, ".format(row.userdata1) # described as 1-100 chars in atcf doc
        # Propagate optional columns
        for col in ["userdefine2", "userdata2", "userdefine3", "userdata3", "userdefine4", "userdata4"]:
            if col in row:
                if not isinstance(row[col], str):
                    pdb.set_trace()
                atcf_lines += "{:s}, ".format(row[col]) 
        atcf_lines += "\n"

    atcf_lines = atcf_lines.replace("nan","   ") # TODO: Check above for potential nans and substitute appropriate number of spaces instead of forcing 3 spaces here.

    if debug:
        pdb.set_trace()

    mode = "w"
    if append:
        mode = "a"
    f = open(ofile, mode)
    f.write(atcf_lines)

    f.close()
    if append:
        print("appended to", ofile)
    else:
        print("wrote", ofile)

def origgridWRF(df, griddir, grid="d03", wind_radii_method = "max", debug=False):
    # Get vmax, minp, radius of max wind, max radii of wind thresholds from WRF by Alex Kowaleski
    
    wregex = r'WF((\d\d)|(CO))'
    WRFmember = df.model.str.extract(wregex, flags=re.IGNORECASE)
    # column 0 will have match or null
    if pd.isnull(WRFmember.iloc[:,0]).any():
        if debug:
            print('Assuming WRF ensemble member, but not all model strings match '+wregex)
            print(df)
        pdb.set_trace()
    ens = WRFmember.iloc[0,0]
    #ens = int(ens) # strip leading zero
    for index, row in df.iterrows():
        if 'origmeshTrue' in row.userdata1:
            print("wait. row",index,"already has original mesh values. Skipping.")
            continue
        gridfile = griddir + "EPS_"+str(ens)+"/E"+str(ens)+"_"+row.initial_time.strftime('%m%d%H') + \
            "_"+grid+"_"+ row.valid_time.strftime('%Y-%m-%d_%H:%M:%S') +"_ll.nc"
        if debug:
            print('opening ' + gridfile)
        nc = Dataset(gridfile, "r")
        lon = nc.variables['lon'][:]
        lat = nc.variables['lat'][:]
        lonCell,latCell = np.meshgrid(lon, lat)
        iTime = 0
        u10  = nc.variables['u10'][iTime,:,:]
        v10  = nc.variables['v10'][iTime,:,:]
        mslpvar = nc.variables['slp']
        if mslpvar.units != 'hPa':
            print("atcf.origgridWRF: unexpected units for mslp: "+mslpvar.units)
            sys.exit(1)
        mslp = mslpvar[iTime,:,:] * 100.
        if debug:
            print('closing ' + gridfile)
        nc.close()

        if debug:
            print("Extract vmax, RMW, minp, and radii of wind thresholds from row", row.name)
        raw_vmax_kts, raw_RMW_nm, raw_minp, wind_radii_nm, raw_pouter_mb, raw_router_nm = derived_winds(u10, v10, mslp, lonCell, latCell, row, wind_radii_method=wind_radii_method, debug=debug)
        # TODO: does this call to update_df() have the right arguments? I think I changed it in the origgrid for ECMWF reading. 
        df = update_df(df, row, raw_vmax_kts, raw_RMW_nm, raw_minp, wind_radii_nm, raw_pouter_mb=raw_pouter_mb,
                raw_routernm=raw_router_nm, gridfile=gridfile, debug=debug)

    # Sort DataFrame by index (deal with appended wind radii lines)
    # sort by rad too
    df = df.sort_index().sort_values(['initial_time','fhr','rad'])
    return df

def get_var_with_str(nc, s):
    # find and return variable that contains string s (case-insensitive)
    # error if more than one match.
    matching_vars = [v for v in nc.variables if s.lower() in v.lower()] # .lower() makes it case-insensitive
    if len(matching_vars) != 1:
        print("number of matching variables not 1")
        print(nc.variables, s, matching_vars)
        sys.exit(1)
    return matching_vars[0]


def origgrid(df, griddir, ensemble_prefix="ens_", wind_radii_method="max", debug=False):
    # Get vmax, minp, radius of max wind, max radii of wind thresholds from ECMWF grid, not from tracker.
    # Assumes
    #   ECMWF data came from TIGGE and were converted from GRIB to netCDF with ncl_convert2nc.
    #   4-character model string in ATCF file is "EExx" (where xx is the 2-digit ensemble member).
    #   ECMWF ensemble member in directory named "ens_xx" (where xx is the 2-digit ensemble member). 
    #   File path is "ens_xx/${gs}yyyymmddhh.xx.nc", where ${gs} is the grid spacing (0p15, 0p25, or 0p5).
    # ensemble_prefix may be a single string or a list of strings

    if isinstance(ensemble_prefix, str):
        ensemble_prefixes = [ensemble_prefix]
    elif isinstance(ensemble_prefix, (list, tuple)):
        ensemble_prefixes = ensemble_prefix

    # Group by initial_time and model because each group corresponds to a different input file.
    # If you process each row independently, you have to open and close the same input file for each foreacst hour.
    for (initial_time, model), group in df.groupby(['initial_time', 'model']):
        m = re.search(r'EE(\d\d)', model)
        if not m:
            if debug:
                print('Assuming ECMWF ensemble member, but did not find EE\d\d in model string')
                print('no original grid for',model,'- skipping')
            continue
        ens = int(m.group(1)) # strip leading zero

        # used to skip EE00 because I didn't know how to handle control run. Now it is handled.
        #if ens < 1:
        #    continue

        # Allow some naming conventions
        # ens_n/yyyymmddhh.n.nc
        # ens_n/0p15yyyymmddhh_sfc.nc
        # ens_n/0p25yyyymmddhh_sfc.nc
        # ens_n/0p5yyyymmddhh_sfc.nc
        yyyymmddhh = initial_time.strftime('%Y%m%d%H')
        yyyymmdd_hhmm   = initial_time.strftime('%Y%m%d_%H%M')
        potential_gridfiles = []
        for ensemble_prefix in ensemble_prefixes:
            # If first filename doesn't exist, try the next one, and so on...
            # List in order of most preferred to least preferred.
            potential_gridfiles.extend([
                                   ensemble_prefix+str(ens)+"/SFC_"+yyyymmdd_hhmm+".nc", # Linus-style
                                   ensemble_prefix+str(ens)+"/"+ "0p125"+yyyymmddhh+"."+str(ens)+".nc",
                                   ensemble_prefix+str(ens)+"/"+ "0p15"+yyyymmddhh+"."+str(ens)+".nc",
                                   ensemble_prefix+str(ens)+"/"+ "0p25"+yyyymmddhh+"."+str(ens)+".nc",
                                   ensemble_prefix+str(ens)+"/"+ "0p5"+yyyymmddhh+"."+str(ens)+".nc",
                                   ensemble_prefix+str(ens)+"/"+ yyyymmddhh+"."+str(ens)+".nc"
                                   ])
        for gridfile in potential_gridfiles:
            if os.path.isfile(griddir + gridfile):
                gridfile = griddir + gridfile # gridfile now has full path. Important for saving in atcf file. Or else os.path.realpath might use current directory.
                break
            else:
                if debug:
                    print("no", griddir + gridfile)

        print('opening', gridfile)
        nc = Dataset(gridfile, "r")
        lon = nc.variables[get_var_with_str(nc, 'lon_')][:]
        lat = nc.variables[get_var_with_str(nc, 'lat_')][:]
        lonCell,latCell = np.meshgrid(lon, lat)
        u10s  = nc.variables[get_var_with_str(nc, '10u')][:]
        v10s  = nc.variables[get_var_with_str(nc, '10v')][:]
        mslps = nc.variables[get_var_with_str(nc, 'msl')][:]
        model_forecast_times = nc.variables['forecast_time0'][:]
        nc.close()


        for fhr, row in group.groupby(["fhr"]):
            if not any(model_forecast_times == fhr):
                print('forecast hour', fhr, 'not in model file')
                continue
            itime = np.argmax(model_forecast_times == fhr)
            u10  =  u10s[itime,:,:]
            v10  =  v10s[itime,:,:]
            mslp = mslps[itime,:,:]

            # Extract vmax, RMW, minp, and radii of wind thresholds
            #row = row.squeeze() # Convert 1-row DataFrame to Series. Do this when calling function, so row isn't really changed after function is completed.
            raw_vmax_kts, raw_RMW_nm, raw_minp, wind_radii_nm, raw_pouter_mb, raw_router_nm = derived_winds(u10, v10, mslp, lonCell, latCell, row.squeeze(), wind_radii_method=wind_radii_method, debug=debug)

            df = update_df(df, initial_time, model, fhr, raw_vmax_kts, raw_RMW_nm, raw_minp, wind_radii_nm, raw_pouter_mb=raw_pouter_mb,
                    raw_router_nm=raw_router_nm, gridfile=gridfile, debug=debug)

    return df

if __name__ == "__main__":
    read(ifile=sys.argv[1], debug=True)
