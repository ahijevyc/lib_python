import cartopy
import csv
import math # for math.e
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import datetime
import metpy.calc
from metpy.units import units
import narr
from netCDF4 import Dataset
import numpy as np
import os, sys
import pandas as pd
import pdb
import re
from scipy.stats import circmean # for averaging longitudes
import warnings # to suppress useless pandas warning when sorting tz-aware index
import xarray

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


# Ellipsoid [CLARKE 1866]  Semi-Major Axis (Equatorial Radius)
Rearth = 6378.2064 * units.km


quads = {'NE':0, 'SE':90, 'SW':180, 'NW':270}
wind_threshes = np.array([34, 50, 64]) * units("knots")

def getcy(cys): # needed to define converters below.
    # return numeric portion of string
    # tc_pairs doesn't match an adeck with cy=13L to a best track with cy=13
    return ''.join([i for i in cys if i.isdigit()])

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


def basins():
    return {
        "al": (-99,-22,0,38),
        "Gulf": (-100,-80,21.5,34),
        "Michael2018": (-93,-80,21.5,34),
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



def V500c(Vmax, latitude_degrees):
    # Climatological tangential wind 500 km from the center 
    # Zehr and Knaff, equations 4, 5, 6
    # https://journals.ametsoc.org/waf/article/22/1/71/38805/Reexamination-of-Tropical-Cyclone-Wind-Pressure
    Vmax_knots = Vmax.to("knots").m 
    shape_factor_x = 0.1147 + 0.0055 * Vmax_knots - 0.001 * (latitude_degrees - 25.)
    Rmax = 66.785 - 0.09102 * Vmax_knots + 1.0619 * (latitude_degrees - 25.)
    climatological_tangential_wind_500_km_from_the_center = Vmax_knots * (Rmax/500.)**shape_factor_x

    if np.ndim(Vmax_knots) == 0: # deal with scalar
        if Vmax_knots < 15.:
            return Vmax_knots
    else: # deal with iterables
        if any(Vmax_knots < 15.):
            print("atcf.V500c(): leaving Vmax < 15 unchanged")
            climatological_tangential_wind_500_km_from_the_center[Vmax_knots < 15] = Vmax_knots[Vmax_knots < 15]
    return climatological_tangential_wind_500_km_from_the_center * units("knots")

def Knaff_Zehr_Pmin(Vsrm1_knots, storm_size_S, latitude_degrees, environmental_pressure_hPa, debug=False):
    # Equation 1 in Courtney and Knaff http://rammb.cira.colostate.edu/resources/docs/Courtney&Knaff_2009.pdf
    Pc = 23.286 - 0.483 * Vsrm1_knots - ( Vsrm1_knots/24.254 )**2. - 12.587*storm_size_S - 0.483*latitude_degrees + environmental_pressure_hPa
    return Pc

def Vsrm(Vmax_knots, storm_motion_in_knots_C):
    # Replace NaN with zero.
    storm_motion_in_knots_C = storm_motion_in_knots_C.fillna(0)
    return Vmax_knots - 1.5 * storm_motion_in_knots_C**0.63

def speed_heading(lon, lat, time):
    # return speed and heading from previous point to next point
    speed = np.zeros_like(lon) * units["m/s"]
    bearing = np.zeros_like(lon) * units.deg

    nt = len(time)
    for t in range(nt):
        if t == 0:
            first_point = 0
            second_point = 1 
        elif t == nt-1:
            first_point = -2
            second_point = -1
        else:
            first_point = t-1
            second_point = t+1
        d, b = dist_bearing(lon[first_point], lat[first_point], lon[second_point], lat[second_point])
        bearing[t] = b
        dt = time[second_point] - time[first_point]
        dt = dt.total_seconds() * units.s
        speed[t] = d/dt 

    
    return speed, bearing


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

def plot_track(ax, start_label,group,end_label, scale=1, debug=False, label_interval_hours=1, **kwargs):
    if debug:
        print("plot_track: "+start_label)
        print(group)
    group = group.sort_values("valid_time")

    if ax.projection.proj4_params["lon_0"] == 180:
        print("plot_track(): changing longitudes from -180,180 to 0,360")
        group.loc[group.lon<0,"lon"] += 360 * units.deg

    for i in range(0, len(group)):
        row = group.iloc[i]
        TCcategory, color = kts2category(row.vmax)
        if end_label == "BEST":
            color = 'black'
        if i == 0:
            # first half-segment
            row_1 = group.iloc[i]
            if len(group) == 1: # track is 1-point-long
                row1 = group.iloc[i]
            else:
                row1 = group.iloc[i+1]
            ax.text(row.lon,row.lat, start_label, color='black', clip_box=ax.bbox, clip_on=True, 
                    ha='center', va='baseline', fontsize=7*scale, transform=cartopy.crs.PlateCarree())
        elif i == len(group)-1:
            # last half-segment
            row_1 = group.iloc[i-1]
            row1 = group.iloc[i]
            # hide for now
            #ax.text(row.lon, row.lat, end_label, color='black', clip_box=ax.bbox, clip_on=True,
            #        ha='center', va='baseline', fontsize=7*scale, transform=cartopy.crs.PlateCarree())
        else:
            # middle segments
            row_1 = group.iloc[i-1]
            row1 = group.iloc[i+1]
        if row.valid_time == datetime.datetime(2018,10,10,18):
            ax.text(row.lon, row.lat, end_label, color="black", clip_box=ax.bbox, clip_on=True, 
                ha='center',va='center_baseline',fontsize=10*scale, transform=cartopy.crs.PlateCarree())
        lat0 = (row_1.lat + row.lat) / 2.
        lon0 = (row_1.lon + row.lon) / 2. # TODO use circmean?
        lw = 1.5
        lat1 = (row.lat + row1.lat) / 2.
        lon1 = (row.lon + row1.lon) / 2.
        if TCcategory == "TD" and row1.basin == 'TG':
            # IF this is TC genesis track and not a storm with TC vitals
            # use "OTHER (NON TD)" instead of "CLASSIFIED TD" color
            color = "white"
        if debug: print(i, "plot segment", [lon0,lon1],[lat0,lat1])
        segment = ax.plot([lon0,row.lon,lon1],[lat0,row.lat,lat1], # Include middle point. Full segment is a bent line.
                c=color, lw=lw*scale, transform=cartopy.crs.PlateCarree())
        if row.valid_time.hour % label_interval_hours == 0: # label if hour is multiple of label_interval_hours
            lformat = "%-d"
            if row.valid_time.hour != 0:
                lformat = "%-Hz"
            lformat = "" # hide for now
            ax.text(row.lon, row.lat, row.valid_time.strftime(lformat), color=contrasting_color(color), clip_box=ax.bbox, clip_on=True, 
                ha='center',va='center_baseline',fontsize=7*scale, transform=cartopy.crs.PlateCarree())
        markersize = scale*8 if row.valid_time.hour == 0 else 0.
        ax.plot(row.lon, row.lat, 'o', markersize=markersize, markerfacecolor=color, markeredgewidth=0.0, color=color, transform=cartopy.crs.PlateCarree())
    label = "track label description"
    already_annotated = label in [x.get_label() for x in ax.texts]
    if lformat and not already_annotated:
        ax.annotate(s = "day of month at hour 0", xy=(3,3), xycoords='axes pixels', fontsize=6, label=label)

def TClegend(fig, left=94, up=32):
    # legend was screen-grabbed from tropicalatlantic.com
    #legend = plt.imread('/glade/work/ahijevyc/share/tropicalatlantic.legend.png')
    legend = plt.imread('/glade/work/ahijevyc/share/TClegend.png')
    # overlay left pixels in and up pixels up from bottom left corner
    l = fig.figimage(legend, left, up)
    # Bring legend to front
    l.set_zorder(3)
    return l

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
            speed, heading = speed_heading(x.lon*units["degrees_E"], x.lat*units["degrees_N"], x.valid_time)
            x.loc[1:,"heading"] = heading[1:] # skip first element; it is always missing coming out of speed_heading()
            df2 = df2.append(x, sort=True)

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
            speed, heading = speed_heading(x.lon*units["degrees_E"], x.lat*units["degrees_N"], x.valid_time)
            x.loc[1:,"heading"] = heading[1:] # skip first element; it is always missing coming out of speed_heading()
            x = x.dropna(how='all', subset=['initial_time']) # I think this cleans up undefined extrapolated times
            df2 = df2.append(x, sort=True)

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


def expand_wind_radii(df, debug=False):
    df = df.set_index("rad").reindex(['0','34','50','64']).reset_index() # make sure full set of thresholds is defined. original dataframe may not have them all.
    columns = list(df.columns)
    for f in ["rad", "seas"]:
        for r in ["1", "2", "3", "4"]:
            columns.remove(f+r)
    df[columns] = df[columns].fillna(axis='index', method='backfill')        
    df[columns] = df[columns].fillna(axis='index', method='ffill')        
    return df

def return_expandedwindradii(df, debug=False):
    tracks = df.groupby(['basin','cy','initial_time','model','fhr'])
    df = tracks.apply(expand_wind_radii, debug=debug)
    df = df.reset_index(drop=True) # Don't need level_0 and level_1 columns in DataFrame
    df[["rad1","rad2","rad3","rad4"]] = df[["rad1","rad2","rad3","rad4"]].fillna(value=0) # Make missing wind radii zero.
    df[["seas1","seas2","seas3","seas4"]] = df[["seas1","seas2","seas3","seas4"]].fillna(value=0) # Make missing seas radii zero.
    return df

# Return list of user-defined columns that are all empty.
def empty_usercols(df, cols=["userdefine","userdata"]):
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
    df.loc[:,'valid_time'] = df.initial_time + pd.to_timedelta(df.fhr, unit='h')
    # add minutes for BEST tracks. 2-digit minutes are in the TECHNUM column for BEST tracks. TECHNUM means something else for non-BEST tracks and shouldn't be added like a timedelta.
    besttracks = df[df.model == 'BEST']
    besttracks.loc[:,'valid_time']   +=  pd.to_timedelta(besttracks.technum.fillna(0), unit='minute')
    besttracks.loc[:,'initial_time'] +=  pd.to_timedelta(besttracks.technum.fillna(0), unit='minute')
    df.loc[df.model == 'BEST'] = besttracks




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
        speed, heading = speed_heading(df.lon*units["degrees_E"], df.lat*units["degrees_N"], df.valid_time)
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
#   lon1 - longitude of origin with units
#   lat1 - latitude of origin with units
#   lons - longitudes of points to get distance to. Could be DataArray or numpy array with units
#   lats - latitudes of points to get distance to
# Returns 2 things:
#   1) distance (pint quantity with units km)
#   2) initial bearing from 1st pt (lon1, lat1) to an array of other points (lons, lats). (also pint quantity)
def dist_bearing(lon1,lat1,lons,lats,Rearth=Rearth, debug=False):
    assert lat1 <= 90*units.deg, f"lat1 {lat1} > 90deg"
    assert lat1 >= -90*units.deg,f"lat1 {lat1} < -90deg"
    # TODO: allow scalar lons, lats
    #assert lats.max() < 90, "lats element > 90"
    #assert lats.min() > -90, "lats element < -90"
    if hasattr(lons, 'metpy'):
        lons = lons.metpy.quantify()
        lats = lats.metpy.quantify()
    # great circle distance. 
    arg = np.sin(lat1)*np.sin(lats)+np.cos(lat1)*np.cos(lats)*np.cos(lon1-lons)
    #arg = np.where(np.fabs(arg) < 1., arg, 0.999999) # sometimes arg = 1.000000000000002
    if (np.fabs(arg) > 1).any():
        if debug:
            print("atcf.dist_bearing(): minarg=",arg.min(),"maxarg=",arg.max())

    if hasattr(arg, "where"):
        # Use xarray.DataArray.where to preserve DataArray coordinates
        arg = arg.where(arg <=  1., other= 1.) # sometimes arg = 1.000000000000002 
        arg = arg.where(arg >= -1., other=-1.) 
    else:
        arg = np.where(arg <=  1., arg,  1.) # sometimes arg = 1.000000000000002 
        arg = np.where(arg >= -1., arg, -1.) 

    dlon = lons-lon1
    bearing = np.arctan2(np.sin(dlon)*np.cos(lats), np.cos(lat1)*np.sin(lats) - np.sin(lat1)*np.cos(lats)*np.cos(dlon)) 

    # -180 - 180 -> 0 - 360
    bearing = (bearing + 360*units.deg) % (360*units.deg) # parentheses around 360*units.deg are important
    
    if (np.fabs(arg) > 1).any():
        print("atcf.dist_bearing(): arg=",arg)
        pdb.set_trace()
    
    distance_from_center = np.arccos(arg)*Rearth
    # Treating DataArrays and pint arrays separately sure is getting kludgy.
    if hasattr(lons, 'metpy'):
        bearing = bearing.metpy.convert_units("degrees")
        distance_from_center = distance_from_center.metpy.convert_units("km")
    else:
        bearing = bearing.to("degrees")
        distance_from_center = distance_from_center.to("km")

    return distance_from_center, bearing


def get_azimuthal_mean(x, distance, binsize = 25.*units.km):
    # Same units for distance and binsize
    distance = distance.metpy.convert_units(binsize.units)
    # Don't use xarray or pint quanity, because these aren't implemented in np.histogram.
    bins = np.arange(0, distance.data.max().m, binsize.m)
    n, bin_edges = np.histogram(distance, bins=bins)
    assert (bin_edges == bins).all()
    if (n == 0).any():
        print("get_azimuthal_mean: no pts b/t", bins[n == 0], "and", bins[n == 0] + binsize)
        sys.exit(1)
    if (n == 1).any():
        print("get_azimuthal_mean: only 1 pt b/t", bins[n == 1], "and", bins[n == 1] + binsize)
    h, _ = np.histogram(distance, bins=bins, weights=x) * x.metpy.units
    x_vs_radius = h/n
    return x_vs_radius, bins * binsize.units


def get_ext_of_wind(wind_speed, distance, bearing, raw_vmax_kts, quads=quads, wind_threshes=wind_threshes, 
        rad_search_radius=300.*units("nautical_mile"), lonCell=None, latCell=None, debug=False, wind_radii_method='max'):
    
    wind_radii_nm = {"wind_radii_method":wind_radii_method}
    # Put in dictionary "wind_radii_nm" where
    # wind_radii_nm = {
    #           34: {'NE':rad1, 'SE':rad2, 'SW':rad3, 'NW':rad4},
    #           50: {'NE':rad1, 'SE':rad2, 'SW':rad3, 'NW':rad4},
    #           64: {'NE':rad1, 'SE':rad2, 'SW':rad3, 'NW':rad4} 
    #          }


    wind_radii_nm['raw_vmax_kts'] = raw_vmax_kts
    wind_radii_nm['thresh_kts'] = wind_threshes
    wind_radii_nm['quads'] = quads

    # Originally had distance < 800km, but Chris D. suggested 300nm in Sep 2018 email
    # This was to deal with Irma and the unrelated 34 knot onshore flow in Georgia
    # Looking at HURDAT2 R34 sizes (since 2004), ex-tropical storm Karen 2015 had 710nm.
    # Removing EX-tropical storms, the max was 480 nm in Hurricane Sandy 2012
    # see /glade/work/ahijevyc/atcf/R34noEX.png and R34withEX.png
    # wind_speed is masked beyond rad_search_radius
    wind_speed = wind_speed.where(distance < rad_search_radius)

    for wind_thresh in wind_threshes:
        if wind_thresh > raw_vmax_kts:
            continue
        imax = distance.where(wind_speed >= wind_thresh).argmax(distance.dims)
        # warn if max_dist_of_wind_threshold is on edge of 2-d domain (like nested WRF grid)
        if distance.ndim == 2:
            if debug:
                print("get_ext_of_wind(): imax", imax)
            for dim, i in imax.items():
                if i == 0 or i == distance[dim].size-1:
                    print(f"get_ext_of_wind(): R{wind_thresh} at edge of domain. {imax} shape: {distance.shape}")
        wind_radii_nm[wind_thresh] = {}
        if debug:
            print('get_ext_of_wind(): method ' + wind_radii_method)
            print('get_ext_of_wind(): kts quad azimuth npts   dist   bearing     lat     lon')
        for quad,az in quads.items():
            # Compute azimuthal mean
            if wind_radii_method == "azimuthal_mean":
                iquad = (az <= bearing) & (bearing < az+90) 
                wind_speed_vs_radius_km, radius_km = get_azimuthal_mean(wind_speed[iquad], distance[iquad], binsize = 25.*units.km)
                wind_radii_nm[wind_thresh][quad] = 0.*units.km
                if any(wind_speed_vs_radius_km >= wind_thresh):
                    max_dist_of_wind_threshold_nm = np.max(radius_km[wind_speed_vs_radius_km >= wind_thresh]).metpy.convert_units("nautical_mile")
                    wind_radii_nm[wind_thresh][quad] = max_dist_of_wind_threshold_nm.data
                    if debug:
                        print('get_ext_of_wind():', "%3d "%wind_thresh, quad, ' %3d-%3d'%(az,az+90), '%4d'%np.sum(iquad), 
                              '%6.1fnm'%max_dist_of_wind_threshold_nm, end="")
                        print(radius_km)
                        print(wind_speed_vs_radius_km)
                        print()
            else:
                iquad = (az <= bearing) & (bearing < az+90) & (wind_speed >= wind_thresh)
                wind_radii_nm[wind_thresh][quad] = 0.*units.km
                if np.sum(iquad) > 0:
                    x_km = distance.where(iquad)
                    if wind_radii_method[-10:] == "percentile":
                        # assume wind_radii_method is a number followed by the string "percentile".
                        distance_percentile = float(wind_radii_method[:-10])
                        # index of array entry nearest to percentile value
                        idist_of_wind_threshold=abs(x_km-np.percentile(x_km,distance_percentile,interpolation='nearest')).argmin()
                        wind_radii_nm[wind_thresh][quad] = np.percentile(x_km, distance_percentile).metpy.convert_units("nautical_mile")
                    elif wind_radii_method == "max":
                        idist_of_wind_threshold = x_km.argmax(x_km.dims)
                        wind_radii_nm[wind_thresh][quad] = x_km.max().metpy.convert_units("nautical_mile").data
                    else:
                        print("unexpected wind_radii_method:" + wind_radii_method)
                        sys.exit(1)
                    if debug:
                        print(f'get_ext_of_wind(): {wind_thresh} {quad} {az}-{az+90} {iquad.sum()}'
                              '%6.2fnm'%wind_radii_nm[wind_thresh][quad], bearing.isel(idist_of_wind_threshold), end="")
                        if lonCell is not None:
                            print('%8.2fN'%latCell.isel(idist_of_wind_threshold), '%7.2fE'%lonCell.isel(idist_of_wind_threshold), end="")
                        print()
    return wind_radii_nm

def get_normalize_range_by(df, index, normalize_by, debug=False):
    #print(df.loc[index,:])
    # Grab the value to normalize by from df DataFrame row, index.
    if normalize_by == 'r34':
        rad = df.loc[index, 'rad']
        if rad == '34':
            wind_radii = df.loc[index, ['rad1','rad2','rad3','rad4']]
            value = wind_radii.max()
            print(wind_radii)
            if np.isnan(value):
                value = 50.
                print("get_normalize_range_by(): r34 is zero. Normalize by {:.0f} nautical miles".format(value))
        else:
            print("get_normalize_range_by(): Unexpected 'rad' value", rad)
            sys.exit(1)
    elif normalize_by == 'Vt500km':
        valid_time = df.loc[index, "valid_time"]
        # Had targetdir set to "." but it grabbed and converted NARR grb in the current directory
        data = narr.vectordata("wind10m", valid_time, targetdir=workdir, debug=debug)
        lon, lat = data.metpy.longitude, data.metpy.latitude
        u, v = data
        derived_vitals_dict = atcf.derived_winds(u, v, xarray.full_like(u, 1013.)*units('hPa'), lon, lat, df.loc[index, :], debug=debug)
        storm_size_S = derived_vitals_dict["storm_size_S"]
        if np.isnan(storm_size_S):
            print("storm_size_S is nan. This may be for Isaac 2012, which has artificial lat/lon extension, but no vmax")
            storm_size_S = df.loc[index, "storm_size_S"]
        if storm_size_S < 0.25:
            print("storm_size_S is too small",storm_size_S)
            print("setting to 0.25")
            storm_size_S = 0.25
        print(f'normalizing range by Knaff_Zehr S. {df.loc[index, "lat"]:.2f}N Vmax {df.loc[index, "vmax"]:.2f}')
        if debug:
            print("Vmax from NARR (not used)", derived_vitals_dict["raw_vmax_kts"], "kts")
        print(f'Vt_500km_kts={derived_vitals_dict["Vt_500km_kts"]:.2f}  S={storm_size_S:.2f}')
        # Originally took inverse of storm_size_S, but that is wrong. If you have a storm 10% larger than normal, 
        # you want to pull everything 10% closer to the origin, so it matches up with other storms that are normal sized.
        # The radial distance is divided by this value.
        assert storm_size_S != 0, "can't be zero"+str(df.loc[index,:])
        if np.isnan(storm_size_S):
            print ("value to normalize range by can't be nan")
            pdb.set_trace()
        return storm_size_S
    else:
        value = df.loc[index, normalize_by]
        if np.isnan(value):
            if normalize_by == 'rmw':
                value = 25. # 25 nautical miles is default rmw in aswip.
            else:
                print("get_normalize_range_by(): Null value for", normalize_by)
                print("not sure how to define")
                sys.exit(1)


    value = value * units["nautical_mile"].to("km")


    assert value != 0, "can't be zero"+str(df.loc[index,:])
    assert not np.isnan(value), "can't be nan"+str(df.loc[index,:])

    return value
