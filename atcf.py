import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import csv
import logging
import math # for math.e
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker 
import datetime
import metpy.calc
import metpy.constants
from metpy.units import units
import numpy as np
import os, sys
import pandas as pd
import pdb
import re
from scipy.stats import circmean # for averaging longitudes
#from stormevents.nhc import nhc_storms, VortexTrack
import xarray

basin_bounds = {
        "al": (-99,-22,0,38),
        "Gulf": (-100,-80,21.5,34),
        "Michael2018": (-93,-80,21.5,34),
        "Irma": (-89,-70,20,34),
        "Irma1": (-81.1,-78.6,22.5,25.2),
        "ep": (-175,-94,0,34),
        "cp": (150,-135+360,0,34),
        "io": (30,109,0,28),
        "wp": (99,180,0,38),
        "global": (-179.999,180,-20,70), # changed -180 to 179.999 to avoid UserWarning: Attempting to set identical left == right == 179.99999999999932 results in singular transformations
        "track" : None # plot domain is simply the storm track
        }


degE = units.parse_expression("degree_E")
degN = units.parse_expression("degree_N")
deg  = units.parse_expression("degree")
kt   = units.parse_expression("knot")
hPa  = units.parse_expression("hPa")
nmi  = units.parse_expression("nautical_mile")

# Ellipsoid [CLARKE 1866]  Semi-Major Axis (Equatorial Radius)
#Rearth = 6378.2064 * units.km
Rearth = metpy.constants.earth_avg_radius


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


def contrasting_color(color):
    # color could be a size-3 tuple, or a string, like "white"
    assert mcolors.is_color_like(color)
    # to_rgba() returns a tuple, which has no attribute 'mean'
    if np.array(mcolors.to_rgba(color)).mean() >= 0.45:
        return 'black'
    return 'white'


quads = {
            'NEQ' : [0, 90, 180, 270]*deg,
            'SEQ' : [90, 180, 270, 0]*deg,
            'SWQ' : [180, 270, 0, 90]*deg,
            'NWQ' : [270, 0, 90, 180]*deg
        }

wind_threshes = np.array([34, 50, 64]) * kt


def icy(cys): # needed to define converters below.
    # return numeric portion of string, keeping as string.
    # tc_pairs doesn't match an adeck with cy=13L to a best track with cy=13
    return ''.join([i for i in cys if i.isdigit()])

converters = {
        "basin" : lambda x: x.upper(), # official is capitalized
        # The problem with CY is ATCF only reserves 2 characters for it.
        "cy" : icy, # cy is not always an integer (e.g. 10E) 
        "initial_time" : lambda x: pd.to_datetime(x.strip(),format='%Y%m%d%H'),
        #"vmax": float,
        #"minp": float,
        "ty": str, # why was this not in here for so long?
        "rad" : lambda x: x.strip(), # not a number. it is a category. Important for interpolating in time.
        "windcode" : lambda x: x[-3:],
        "subregion": lambda x: x[-2:],
         # subregion ends up being 3 characters when written with .to_string
         # strange subregion only needs one character, but official a-decks leave 3. 
        "initials" : lambda x: x[-3:],
        'stormname': lambda x: x[-9:],
        'depth'    : lambda x: x[-1:],
        "seascode" : lambda x: x[-3:],
        "userdefine1": str,
        "userdefine2": str,
        "userdefine3": str,
        "userdefine4": str,
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
# If you add a key-value pair to na_values, also add it to dtype dict, and remove it from converters.
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


# for metpy.units.pandas_dataframe_to_unit_arrays(df, column_units=columns_units)
column_units = {x:'dimensionless' for x in atcfcolumns} 
column_units.update(
        dict(
            fhr          = units.hour,
            lat          = degN,
            lon          = degE,
            vmax         = kt,
            minp         = hPa,
            rad          = kt,
            rad1         = nmi,
            rad2         = nmi,
            rad3         = nmi,
            rad4         = nmi,
            pouter       = hPa,
            router       = nmi,
            rmw          = nmi,
            gusts        = kt,
            maxseas      = units.feet,
            dir          = deg,
            speed        = kt,
            seas         = units.feet,
            seas1        = nmi,
            seas2        = nmi,
            seas3        = nmi,
            seas4        = nmi,
            penv         = hPa,
            penv_mb      = hPa,
            Vt_500km     = kt,
            Vt_500km_kts = kt,
            )
        )

class Atcf(pd.DataFrame):
    def __init__(self, x):
        super().__init__(x)
    # No need for ntrack attribute. Use df.groupby(unique_track).ngroups

    def write(self, ofile, fullcircle=False, append=False):
        if self.empty:
            logging.warning("afcf.write(): DataFrame is empty.")

        models = self["model"].unique()

        if fullcircle:
            # deal with fullcircle.
            self = self.groupby(["basin","cy","initial_time","fhr","rad"]).apply(fullcircle_windradii)

        logging.debug(f"writing {models} to {ofile}")
        logging.debug(self.head(1))

        assert 0 not in self.columns, 'TODO: stop putting 0 column'

        # Extra columns not in atcfcolumns.
        extras = [x for x in self.columns if x not in atcfcolumns]
        extras.remove("valid_time")
       
        
        atcf_lines = ""
        for index, row in self.iterrows():
            line = ""
            line += "{:2s}, ".format(row.basin) 
            line += "{:2s}, ".format(row.cy.zfill(2))
            line += "{:10s}, ".format(row.initial_time.strftime('%Y%m%d%H'))
            line += "  , " if np.isnan(row.technum) else "{:02.0f}, ".format(row.technum) # avoid turning 'nan' into 3 spaces below
            line += "{:4s}, ".format(row.model)
            line += "{:3.0f}, ".format(row.fhr)
            line += "{:>4s}, ".format(lat2s(row.lat))
            line += "{:>5s}, ".format(lon2s(row.lon))
            line += "{:3.0f}, ".format(row.vmax)
            line += "{:4.0f}, ".format(row.minp)
            line += "{:2s}, ".format(row.ty)
            line += "{:>3s}, ".format(row.rad)
            line += "{:>3s}, ".format(row.windcode)
            line += "{:4.0f}, ".format(row.rad1)
            line += "{:4.0f}, ".format(row.rad2)
            line += "{:4.0f}, ".format(row.rad3)
            line += "{:4.0f}, ".format(row.rad4)
            line += "{:4.0f}, ".format(row.pouter)
            line += "{:4.0f}, ".format(row.router)
            line += "{:3.0f}, ".format(row.rmw)
            line += "{:3.0f}, ".format(row.gusts)
            line += "{:3.0f}, ".format(row.eye)
            line += "{:>3s}, ".format(row.subregion) # supposedly 1 character, but always 3 in official b-decks
            line += "{:3.0f}, ".format(row.maxseas)
            line += "{:>3s}, ".format(row.initials)
            line += "{:3.0f}, ".format(row.heading)
            line += "{:3.0f}, ".format(row.speed)
            line += "{:>10s}, ".format(row.stormname)
            line += "{:>1s}, ".format(row.depth)
            line += "{:2.0f}, ".format(row.seas)
            line += "{:>3s}, ".format(row.seascode)
            line += "{:4.0f}, ".format(row.seas1)
            line += "{:4.0f}, ".format(row.seas2)
            line += "{:4.0f}, ".format(row.seas3)
            line += "{:4.0f}, ".format(row.seas4)
            # Propagate optional userdefine columns
            for userdefine in ["userdefine"+n for n in ['1','2','3','4','5']]:
                if userdefine in row and row[userdefine].strip(): # exists and non-empty
                    userdata = userdefine.replace("define","data")
                    line += "{:s}, ".format(row[userdefine]) # Described as 1-20 chars in atcf doc. 
                    line += "{:s}, ".format(row[userdata]) # described as 1-100 chars in atcf doc
            # Propagate extra columns
            for extra in extras:
                if extra in row:
                    line += f"{extra}, "
                    line += f"{row[extra]}, "

            atcf_lines += f"{line}\n"

        # Tried assert "nan," not in atcf_lines, f'found nan, in atcf_lines {atcf_lines}'
        # but maxseas may be np.nan.
        atcf_lines = atcf_lines.replace(" nan,","    ,") # possible columns with np.nan all happen to be 3 character columns

        mode = "w"
        if append:
            mode = "a"
        if ofile == "<stdout>":
            sys.stdout.write(atcf_lines)
        else:
            f = open(ofile, mode)
            f.write(atcf_lines)
            f.close()

            if append:
                logging.info(f"appended {models} to {ofile}")
            else:
                logging.info(f"wrote {models} as {ofile}")


# Best tracks have different initial_times and fhr is always 0. 
# To group best tracks with the same unique_list as for models, we will make change all best track 
# initial_times to the first initial_time and make the best track fhr hold the time offset.
unique_track = ["basin", "cy", "initial_time", "model"]


def new_besttrack_times(best_track):
    if "technum" in best_track and best_track.technum.any():
        # add minutes for BEST tracks. 2-digit minutes are in the TECHNUM column for BEST tracks. 
        # TECHNUM means something else for non-BEST tracks and shouldn't be added like a timedelta.
        extra_minutes = pd.to_timedelta(best_track.technum.fillna(0), unit='minute')
        best_track['initial_time'] +=  extra_minutes
        best_track['valid_time']   +=  extra_minutes
        # Set technum to zero so this operation is not applied again.
        best_track["technum"] = 0

    # return unchanged best_track if this operation has been done already
    if "initial_time" in best_track and best_track.initial_time.nunique() == 1:
        expected_valid_time = best_track.initial_time + pd.to_timedelta(best_track.fhr,unit='H')
        assert (best_track.valid_time == expected_valid_time).all(), 'atcf.besttrack_fhr(): initial_time, fhr, and valid_time inconsistent'
        return best_track

    # Make initial_time the same across an entire best track and save the time difference in fhr.
    # Make initial_time all the same as first and store difference in fhr.
    # Helps when you group by unique_track later. 
    first_time = best_track.valid_time.min()
    best_track["fhr"] = best_track.valid_time - first_time
    best_track["fhr"] /= np.timedelta64(1,'s')*3600
    best_track["initial_time"] = first_time
    return best_track

# TODO: is this pointless (putting write in class Atcf)?
def write(ofile, df, **kwargs):
    Atcf(df).write(ofile, **kwargs)


def get_ax(projection=cartopy.crs.PlateCarree(), bscale='50m'):
    fig = plt.figure()
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
    gl = ax.gridlines(crs=projection, draw_labels=True, color="white", linestyle='--', alpha=0.75)
    gl.xformatter = LONGITUDE_FORMATTER
    # draw longitude line every n degrees.  Default xlocator is 30 deg, which is too far apart. 
    gl.xlocator = matplotlib.ticker.MultipleLocator(base=10.)
    gl.top_labels = False # Don't overwrite title
    gl.yformatter = LATITUDE_FORMATTER
        
    return ax

def vmax2category(vmax):
    if vmax > 137*kt:
        category = "CAT5"
    elif vmax > 113*kt:
        category = "CAT4"
    elif vmax > 96*kt:
        category = "CAT3"
    elif vmax > 83*kt:
        category = "CAT2"
    elif vmax > 64*kt:
        category = "CAT1"
    elif vmax > 34*kt:
        category = "TS"
    else:
        category = "TD"

    return category, colors[category]


def vmax_HollandB_to_minp(vmax_kts, HollandB, environmental_pressure_hPa = 1013, density_of_air = 1.15*units.parse_expression("kg/m^3")):
    vmax = vmax_kts * kt.to("m/s")
    environmental_pressure = environmental_pressure_hPa* hPa.to("Pa")
    minp = environmental_pressure - (vmax**2 * density_of_air * math.e / HollandB)
    return np.around(minp / 100.)


def iswarmcore(track, min_warmcore_percent=25):
    if 'warmcore' not in track.columns:
        logging.error("No warm core column.")
        sys.exit(1)
    s = warmcore = track.warmcore.str.strip()
    warmcore = s == 'Y'
    known = s != 'U' # not unknown
    # If warmcore column exists, make sure at least one time is warmcore or unknown.
    if any(track.warmcore.str.strip() == 'U'):
        logging.warning("warm core unknown")
        return True
    else:
        warmcore_percent = 100*warmcore.sum()/known.sum()
        return warmcore_percent >= min_warmcore_percent


def iswind_radii_method(s):
    # make sure option is 'max', 'azimuthal_mean', or 'xpercentile' where x is a number
    if s == 'max' or s == 'azimuthal_mean':
        return s
    if s[-10:] != 'percentile':
        raise argparse.ArgumentTypeError("String must end with 'percentile'")
    if not float(s[:-10]):
        raise argparse.ArgumentTypeError("First part of string must be a number")
    return s


def archive_path(atcfname): # used by NARR_composite.py
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
            logging.info("atcf.V500c(): leaving Vmax < 15 unchanged")
            climatological_tangential_wind_500_km_from_the_center[Vmax_knots < 15] = Vmax_knots[Vmax_knots < 15]
    return climatological_tangential_wind_500_km_from_the_center * kt

def Knaff_Zehr_Pmin(Vsrm1, storm_size_S, latitude_degrees, environmental_pressure):
    # Equation 1 in Courtney and Knaff http://rammb.cira.colostate.edu/resources/docs/Courtney&Knaff_2009.pdf
    #environmental_pressure_hPa = environmental_pressure.to("hPa").m
    #Vsrm1_knots = Vsrm1.to("knots").m
    #Pc = 23.286 - 0.483 * Vsrm1_knots - ( Vsrm1_knots/24.254 )**2. - 12.587*storm_size_S - 0.483*latitude_degrees + environmental_pressure_hPa
    # unitize coefficients so Vsrm1, latitude, environmental_pressure can be in any units
    Pc = 23.286*hPa - 0.483*units.parse_expression("hPa/knot") * Vsrm1 - ( Vsrm1/(24.254*units.parse_expression("knot * hPa**-0.5")) )**2. - 12.587*hPa*storm_size_S - 0.483*units.parse_expression("hPa/degrees_N")*latitude_degrees + environmental_pressure
    return Pc

def Vsrm(Vmax, storm_motion_C):
    # TODO: Replace NaN with zero?
    # Vmax and storm_motion_C are unitized speeds. Doesn't matter what units.
    return Vmax - 1.5 * kt**(1-0.63) * storm_motion_C**0.63 

def fill_speed_heading(track):
    assert track.groupby(unique_track).ngroups == 1, 'atcf.fill_speed_heading() given multiple tracks'
    oneradtrack = track.groupby("valid_time").first().reset_index() # reduce to one wind radius for each valid_time
    if len(oneradtrack) == 1:
        return track # no way to get distance and heading from one point.
    # get speed and heading of track for one wind radius.
    lon = oneradtrack.lon.values * degE
    lat = oneradtrack.lat.values * degN
    time = oneradtrack["valid_time"].values # .values so speed_heading can use numpy array indices, not iloc
    # return speed and heading from previous point to next point
    speed, bearing = speed_heading(lon, lat, time)
    oneradtrack.loc[:,"speed"] = speed.to("knots").m
    oneradtrack.loc[:,"heading"] = bearing
    # Copy oneradtrack speed and heading to original track (possibly with multiple wind radii lines)
    track = track.set_index("fhr")
    track.loc[:,["speed","heading"]] = oneradtrack.set_index("fhr")[["speed","heading"]]
    track = track.reset_index() # put fhr back in column
    return track

def speed_heading(lon, lat, time):
    assert lon.units == degE, "atcf.speed_heading(): lon not deg E"
    assert lat.units == degN, "atcf.speed_heading(): lat not deg N"
    # return speed and heading from previous point to next point
    speed = np.zeros_like(lon) * units.parse_expression("m/s")
    bearing = np.zeros_like(lon) * deg

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
        if isinstance(dt, np.timedelta64):
            dt /= np.timedelta64(1,"s") # np.timedelta64
        else:
            dt = dt.total_seconds() # datetime.timedelta
        dt *= units.s
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
    # Aggregate with pd.Series.max
    agg_dict.update({x:pd.Series.max for x in df.columns if x.startswith("user")})
    # made fhr optional - it may be part of index
    if "fhr" in df:
        agg_dict["fhr"] = 'mean'

    dfg = df.groupby(['basin','cy','initial_time','valid_time','windcode','rad','seascode'])
    df = dfg.agg(agg_dict)
    df.reset_index(inplace=True)
    df["model"] = "MEAN"
    return df

def plot_track(ax, start_label,group,end_label, scale=1, label_interval_hours=1, onecolor=None):
    logging.debug(f"plot_track: {start_label} {group}")
    group = group.sort_values("valid_time")

    lformat = None
    for i in range(0, len(group)):
        row = group.iloc[i]
        TCcategory, color = vmax2category(row.vmax*kt)
        if i == 0:
            # first half-segment
            row_1 = group.iloc[i]
            if len(group) == 1: # track is 1-point-long
                row1 = group.iloc[i]
            else:
                row1 = group.iloc[i+1]
            ax.text(row.lon,row.lat, start_label, clip_box=ax.bbox, clip_on=True, 
                    ha='center', va='center', fontsize=7*scale, transform=cartopy.crs.PlateCarree())
        elif i == len(group)-1:
            # last half-segment
            row_1 = group.iloc[i-1]
            row1 = group.iloc[i]
            ax.text(row.lon, row.lat, end_label, clip_box=ax.bbox, clip_on=True,
                    ha='center', va='center', fontsize=7*scale, transform=cartopy.crs.PlateCarree())
        else:
            # middle segments
            row_1 = group.iloc[i-1]
            row1 = group.iloc[i+1]
        lat0 = (row_1.lat + row.lat) / 2.
        lon0 = circmean([row_1.lon, row.lon], low=-180, high=180)
        lw = 1.5
        lat1 = (row.lat + row1.lat) / 2.
        lon1 = circmean([row.lon, row1.lon], low=-180, high=180)
        if TCcategory == "TD" and row1.basin == 'TG':
            # IF this is TC genesis track and not a storm with TC vitals
            # use "OTHER (NON TD)" instead of "CLASSIFIED TD" color
            color = "white"
        if onecolor:
            color = onecolor
        # Include middle point in line segments. Full segment is a bent line.
        lons = np.array([lon0,row.lon,lon1])
        lats = [lat0,row.lat,lat1]
        dlon = max(lons) - min(lons)
        span_dateline = dlon > 180.
        if span_dateline:
            logging.warning(f"{row.cy} spans dateline")
            lons[lons < 0] += 360
        logging.debug(f"{i} plot segment {lons} {lats}")
        
        segment = ax.plot(lons, lats, c=color, lw=lw*scale, transform=cartopy.crs.PlateCarree())
        if label_interval_hours and row.valid_time.hour % label_interval_hours == 0: # label if hour is multiple of label_interval_hours
            lformat = "%-d"
            if row.valid_time.hour != 0:
                lformat = "%-Hz"
            ax.text(row.lon, row.lat, row.valid_time.strftime(lformat), color=contrasting_color(color), clip_box=ax.bbox, clip_on=True, 
                ha='center',va='center_baseline',fontsize=6*scale, transform=cartopy.crs.PlateCarree())
        markersize = scale*8 if row.valid_time.hour == 0 else 0.
        ax.plot(row.lon, row.lat, 'o', markersize=markersize, markerfacecolor=color, markeredgewidth=0.0, color=color, transform=cartopy.crs.PlateCarree())
    label = "track label description"
    already_annotated = label in [x.get_label() for x in ax.texts]
    if lformat and not already_annotated:
        ax.annotate(text="day of month at hour 0", xy=(5,6), xycoords='axes pixels', fontsize=6, label=label, bbox={'facecolor':'white',
            'linewidth':0, 'alpha':.9, 'pad':0.1, 'boxstyle':'round4'})

def TClegend(ax):
    # legend was screen-grabbed from tropicalatlantic.com
    img = plt.imread('/glade/work/ahijevyc/share/TClegend.png')
    xmin, dx = 0, 1
    ymin = ax.get_position().ymin/2 # halfway between bottom of axes and bottom of figure
    hgt, wid, _ = img.shape
    dy = dx*hgt/wid
    legax = ax.figure.add_axes([xmin, ymin, dx, dy])
    legax.yaxis.set_visible(False)
    legax.xaxis.set_visible(False)
    legax.imshow(img)
    return legax

def interpolate_by_rad(rad, interval):
    # Interpolate in time 
    # This should be from one wind speed threshold 
    logging.debug(f"atcf.interpolate(): padding {rad.name}")
    rad = rad.set_index('valid_time').resample(interval).interpolate(method='pad')
    # redo model track circular heading
    speed, heading = speed_heading(rad.lon.values*degE, rad.lat.values*degN, rad.index)
    rad["heading"] = heading 
    rad = rad.reset_index() # return valid_time from index to column
    return rad 

def interpolate(df, interval):

    if interval not in ['1H','3H','6H','12H','24H']:
        logging.error(f"atcf.interpolate(): unexpected time interval {interval}")
        logging.error("Expected one of ['1H','3H','6H','12H','24H']")
        sys.exit(1)

    # Copied from ~ahijevyc/bin/interpolate_atcf.py on Mar 2, 2020. This method should supercede that script.

    logging.debug(f"atcf.interpolate() {interval}")

    nbad = df.valid_time.isna().sum()
    if nbad > 0:
        logging.info(f"Dropping {nbad} wind radii line(s) with NaN valid_time")
        df.dropna(how='all', subset=['valid_time'], inplace=True)

    # distinguish 34, 50 and 64-kt lines but treat 0kt and 34kt as the same
    df.loc[df["rad"] == "0","rad"] = "34"

    # Include 34, 50 and 64 knot wind radii for each time, so missing radii are explicitly zero km when interpolating.
    df = return_expandedwindradii(df)

    # Interpolate in time
    # handle multiple models, init_times, etc.
    byrad = ['basin', 'cy', 'initial_time', 'model', 'rad']
    df = df.groupby(byrad, as_index=False, group_keys=False).apply(interpolate_by_rad, interval)
    df = df.sort_values(by=['basin','cy','initial_time','model','valid_time','rad']) 
    return df


def stringlatlon2float(lon, lat):
    # Extract last character of lat and lon columns
    # Multiply integer by -1 if "S" or "W"
    # Divide by 10
    S = lat.str[-1] == 'S'
    lat = lat.str[:-1].astype(float) / 10.
    lat[S] = lat[S] * -1
    W = lon.str[-1] == 'W'
    lon = lon.str[:-1].astype(float) / 10.
    lon[W] = lon[W] * -1

    logging.debug("finished converting string lat/lons to float")
    return lon, lat

# rad1-4 and seas1-4 columns
rscols = []
for f in ["rad", "seas"]:
    for r in ["1", "2", "3", "4"]:
        rscols.append(f+r)
        
def expand_wind_radii(df, rads=['34','50','64']):
    df = df.set_index("rad").reindex(rads) # make sure full set of thresholds is defined. original dataframe may not have them all.
    df[rscols] = df[rscols].fillna(value=0) # fill missing rad1-4 and seas1-4 with zeros
    df = df.fillna(method='ffill')
    return df

def return_expandedwindradii(df, rads=["34","50","64"]):
    tracktimes = df.groupby(['basin','cy','initial_time','model','fhr'], sort=False) 
    df = tracktimes.apply(expand_wind_radii, rads=rads)
    df = df.droplevel(['basin','cy','initial_time','model','fhr']).reset_index()
    return df

def add_missing_dummy_columns(df, columns):
    for col in columns:
        if col in df.columns:
            continue
        logging.debug(f"{col} not in DataFrame. Fill with appropriate value.")

        # if column doesn't exist make it zeroes
        if col in ['rad1', 'rad2', 'rad3', 'rad4','pouter', 'router', 'seas', 'seas1','seas2','seas3','seas4']:
            df[col] = 0.

        # if rad column doesn't exist make it string zero.
        elif col in ['rad']:
            df[col] = '0'

        # Initialize other default values.
        elif col in ['windcode', 'seascode']:
            df[col] = '   '

        # Numbers are NaN - change to 3*' ' in write
        elif col in ['rmw','gusts','eye','maxseas','heading','speed']:
            df[col] = np.NaN

        # Strings are empty
        elif col in ['subregion','stormname'] or col.startswith('user'):
            df[col] = ''

        elif col in ['ty']:
            df[col] = 'XX'

        elif col in ['initials', 'depth']:
            df[col] = 'X'
        else:
            logging.error(f"atcf.add_missing_dummy_columns(): unexpected col {col}")
            sys.exit(1)

    return df        

ifile = '/glade/work/ahijevyc/work/atcf/Irma.ECMWF.dat'
ifile = '/glade/scratch/mpasrt/uni/2018071700/latlon_0.500deg_0.25km/gfdl_tracker/tcgen/fort.64'

def read_aswip(ifile = ifile):
    # Read data into Pandas Dataframe
    logging.debug(f'Reading {ifile}')

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
    df["lon"], df["lat"] = stringlatlon2float(df.lon, df.lat)

    # Add missing ATCF columns with dummy data.
    df = add_missing_dummy_columns(df, atcfcolumns)

    # Derive valid time.   valid_time = initial_time + fhr (used in read_aswip too)
    df['valid_time'] = df.initial_time + pd.to_timedelta(df.fhr, unit='h')

    return df

def read(ifile = ifile, fullcircle=False):
    # Read data into Pandas Dataframe
    logging.debug(f'Reading {ifile} fullcircle={fullcircle}')

    names = list(atcfcolumns) # make a copy of list, not a copy of the reference to the list.

    reader = csv.reader(open(ifile),delimiter=',')
    testline = next(reader)
    num_cols = len(testline)
    logging.debug(f"test line num_cols {num_cols}")
    logging.debug(testline)
    del reader
    with open(ifile) as f:
        max_num_cols = max(len(line.split(',')) for line in f)
        logging.debug(f"max number of columns {max_num_cols}")

    # Output from GFDL vortex tracker, fort.64 and fort.66
    # are mostly ATCF format but have subset of columns
    if num_cols == 43:
        logging.info(f'assume GFDL tracker fort.64-style output with 43 columns in {ifile}')
        TPstr = "THERMO PARAMS"
        if testline[35].strip() != TPstr:
            logging.error(f"expected 36th column to be {TPstr}. got {testline[35].strip()}")
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
        logging.info(f'Assuming GFDL track fort.66-style with 31 columns in {ifile}')
        # There is a cyclogenesis ID column for fort.66
        logging.debug('inserted ID for cyclogenesis in column 2 (zero-based)')
        names.insert(2, 'id') # ID for the cyclogenesis
        logging.info('Using 1st 21 elements of names list')
        names = names[0:21]
        logging.debug('redefining columns 22-31')
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
        logging.info("Looks like IDL output")
        names = [n.replace('userdata1', 'min_warmcore_fract') for n in names]
        names.append('dT500')
        names.append('dT200')
        names.append('ddZ850200')
        names.append('rainc')
        names.append('rainnc')
        names.append('id')

    if num_cols == 11:
        logging.info(f"Assuming {ifile} is simple adeck with 11 columns")
        if ifile[-4:] != '.dat':
            logging.info(f"even though file doesn't end in .dat {ifile}")
        names = names[0:11]

    if len(names) > max_num_cols:
        names = names[0:max_num_cols]


    # Tack on user define/data column pairs until all available columns are given names.
    usercolumnindex = 5
    while len(names) < max_num_cols-1:
        names.append(f"userdefine{usercolumnindex}")
        names.append(f"userdata{usercolumnindex}")
        usercolumnindex += 1


    usecols = list(range(len(names)))

    # If you get a beyond index range (or something like that) error, see if userdata1 column is intermittent and has commas in it. 
    # If so, clean it up (i.e. truncate it)

    logging.debug("before pd.read_csv")
    for name,v in zip(names,testline):
        logging.debug(f"{name}:{v}")
    logging.debug(f"converters={converters}")
    logging.debug(f"dype={dtype}")
    logging.debug(f"column_units={column_units}")


    df = pd.read_csv(ifile,index_col=None,header=None, delimiter=",", usecols=usecols, names=names, 
            converters=converters, na_values=na_values, dtype=dtype, skipinitialspace=True, engine='c') # engine='c' is faster than engine="python"

    # fort.64 has asterisks sometimes. Problem with hwrf_tracker. 
    badlines = df['lon'].str.contains("\*")
    if any(badlines):
        print.warning(f"tossing {len(badlines)} lines with asterisk in lon")
        df = df[~badlines]


    df["lon"], df["lat"] = stringlatlon2float(df.lon, df.lat)

    if max_num_cols != num_cols and df.model.nunique() > 1:
        logging.warning(f"test line has {num_cols} columns, but another line has {max_num_cols} columns. Unexpected results may occur.")
        logging.warning("It's hard to deal with userdefined columns and data in a file with multiple types of models.")
        

    # Derive valid time.   valid_time = initial_time + fhr (used in read_aswip too)
    df.loc[:,'valid_time'] = df.initial_time + pd.to_timedelta(df.fhr, unit='h')

    
    # adjust best track times: 1) add technum (2-digit minutes) and set technum to zero
    # 2) make initial_times all the same as the first and put time offset in fhr.
    # This makes it possible to groupby best tracks with same unique_track list as model tracks.
    isbesttrack = df.model == 'BEST'
    df.loc[isbesttrack] = df.loc[isbesttrack].groupby(["basin","cy","model"]).apply(new_besttrack_times)


    # Prior to 1999, rad column is blank. Fill with string zeros. 
    # Downstream programs assume rads are convertable to floats. Empty strings are not convertable to floats.
    if "rad" not in df.columns or all(df.rad == ''):
        logging.warning("atcf.read(): Empty rad column. This happens in pre-2000 files. Changing to string '0' so we can convert to float downstream")
        df["rad"] = '0'

    # sanity check for rad values
    if not all(df.rad.isin(['0','34','50','64'])):
        logging.warning(f"atcf.read(): unexpected rad value(s) in atcf file {ifile}")
        logging.warning(df.rad.value_counts())
        # adecks before 2002 had 35-knot lines, not 34. That's okay. 
        if df.valid_time.min().year > 2001:
            logging.error("atcf.read(): this should not happen with post-2001 files. Exiting.")
            sys.exit(1)

    # Add missing ATCF columns with dummy data.
    df = add_missing_dummy_columns(df, atcfcolumns)

    if fullcircle:
        # Full circle wind radii instead of quadrants
        df = df.groupby(["basin","cy","initial_time","fhr","rad"]).apply(fullcircle_windradii)

    missing_speed_heading = all(((df.speed == 0) | pd.isnull(df.speed)) & ((df.heading == 0) | pd.isnull(df.heading))) # assume bad if everything is zero
    if missing_speed_heading:
        logging.debug("Deriving speed and heading")
        df = df.groupby(unique_track).apply(fill_speed_heading)
        df = df.droplevel(unique_track) # indices are nice but mess up plot_atcf.py later. 

    # Put userdefine/userdata column pairs into single columns. 
    # Allow unaligned userdefine columns from multiple atcf files.
    # ["1", "2", "3", ... ] 
    for x in [str(x) for x in range(1,usercolumnindex)]: 
        # for each unique userdefine value in column 
        for v in df["userdefine"+x].unique():
            if v == "": continue # Ignore empty strings
            # identify rows with this userdefine value
            rows = df["userdefine"+x] == v
            # identify userdata associated with this userdefine value
            userdata = df.loc[rows, "userdata"+x]
            # set values of a column (named after the value in userdefine) to userdata value.
            df.loc[rows, v] = userdata
        # Drop this pair of userdefine/userdata columns. 
        df = df.drop(columns=[u+x for u in ["userdefine","userdata"]])

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
def dist_bearing(lon1,lat1,lons,lats,Rearth=Rearth):
    assert lat1 <=  90*deg, f"lat1 {lat1} >  90deg"
    assert lat1 >= -90*deg, f"lat1 {lat1} < -90deg"
    # TODO: allow scalar lons, lats
    assert lats.max() <  90, "lats element > 90"
    assert lats.min() > -90, "lats element < -90"
    if hasattr(lons, 'metpy'):
        lons = lons.metpy.quantify()
        lats = lats.metpy.quantify()
    # great circle distance. 
    arg = np.sin(lat1)*np.sin(lats)+np.cos(lat1)*np.cos(lats)*np.cos(lon1-lons)
    #arg = np.where(np.fabs(arg) < 1., arg, 0.999999) # sometimes arg = 1.000000000000002
    if (np.fabs(arg) > 1).any():
        logging.debug(f"atcf.dist_bearing(): minarg={arg.min()} maxarg={arg.max()}")

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
    bearing = (bearing + 360*deg) % (360*deg) # parentheses around 360*deg are important
    
    if (np.fabs(arg) > 1).any():
        logging.error(f"atcf.dist_bearing(): arg={arg}")
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
        logging.error(f"get_azimuthal_mean: no pts b/t {bins[n == 0]} and {bins[n == 0] + binsize}")
        sys.exit(1)
    if (n == 1).any():
        logging.info(f"get_azimuthal_mean: only 1 pt b/t {bins[n == 1]} and {bins[n == 1] + binsize}")
    h, _ = np.histogram(distance, bins=bins, weights=x) # Depreciation warning about ragged arrays if you add units here 
    x_vs_radius = h/n
    bin_centers = bins[:-1] * binsize.units + binsize/2
    bin_centers *= binsize.units
    x_vs_radius *= x.metpy.units
    da = xarray.DataArray(data=x_vs_radius, coords={"radius":bin_centers})
    return da


def get_ext_of_wind(wind_speed, distance, bearing, raw_vmax, windcode='NEQ', wind_threshes=wind_threshes, 
        rad_search_radius=300.*units.parse_expression("nautical_mile"), lonCell=None, latCell=None, wind_radii_method='max'):
    
    wind_radii = {"wind_radii_method":wind_radii_method}
    # Returns dictionary "wind_radii" where
    # wind_radii = {
    #  wind_radii_method : wind_radii_method,
    #           windcode : windcode,
    #           raw_vmax : raw_vmax,
    #                rads: {
    #                  34kt: [rad1,rad2,rad3,rad4],
    #                  50kt: [rad1,rad2,rad3,rad4],
    #                  64kt: [rad1,rad2,rad3,rad4]
    #                  }
    #          }

    wind_radii['raw_vmax'] = raw_vmax
    wind_radii['windcode'] = windcode
    wind_radii['rads'] = {}
    # Originally had distance < 800km, but Chris D. suggested 300nm in Sep 2018 email
    # This was to deal with Irma and the unrelated 34 knot onshore flow in Georgia
    # Looking at HURDAT2 R34 sizes (since 2004), ex-tropical storm Karen 2015 had 710nm.
    # Removing EX-tropical storms, the max was 480 nm in Hurricane Sandy 2012
    # see /glade/work/ahijevyc/atcf/R34noEX.png and R34withEX.png
    # wind_speed is masked beyond rad_search_radius
    wind_speed = wind_speed.where(distance < rad_search_radius)
    if raw_vmax < wind_threshes[0]:
        wind_radii['rads'][wind_threshes[0]] = [0,0,0,0]*units.km # write out zero rads for first wind threshold and return 
        return wind_radii

    logging.debug(f'  get_ext_of_wind(): method {wind_radii_method} windcode {windcode}')
    logging.debug('  get_ext_of_wind(): wind_thresh     azimuth       npts    dist   bearing      lat        lon')

    for wind_thresh in wind_threshes:
        if (wind_speed >= wind_thresh).sum() == 0:
            return wind_radii
        if distance.ndim == 2:
            imax = distance.where(wind_speed >= wind_thresh).argmax(distance.dims)
            # warn if max_dist_of_wind_threshold is on edge of 2-d domain (like nested WRF grid)
            logging.debug(f"  get_ext_of_wind(): imax {imax}")
            for dim, i in imax.items():
                if i == 0 or i == distance[dim].size-1:
                    print(f"  get_ext_of_wind(): R{wind_thresh} at edge of domain. {imax} shape: {distance.shape}")
        wind_radii['rads'][wind_thresh] = []
        for az in quads[windcode]:
            daz = 90*deg
            # Compute azimuthal mean
            if wind_radii_method == "azimuthal_mean":
                iquad = (az <= bearing) & (bearing < az+daz) 
                wind_speed_vs_radius_km = get_azimuthal_mean(wind_speed[iquad], distance[iquad], binsize = 25.*units.km)
                if any(wind_speed_vs_radius_km >= wind_thresh):
                    max_dist_of_wind_threshold_nm = np.max(radius_km[wind_speed_vs_radius_km >= wind_thresh])
                    wind_radii['rads'][wind_thresh].append(max_dist_of_wind_threshold_nm.data)
                else:
                    wind_radii['rads'][wind_thresh].append(0.*units.km)
            else:
                # percentile method or max method
                iquad = (az <= bearing) & (bearing < az+daz) & (wind_speed >= wind_thresh)
                if iquad.sum():
                    x_km = distance.where(iquad)
                    if wind_radii_method[-10:] == "percentile":
                        # assume wind_radii_method is a number followed by the string "percentile".
                        distance_percentile = float(wind_radii_method[:-10])
                        # index of array entry nearest to percentile value
                        idist_of_wind_threshold=abs(x_km-np.percentile(x_km,distance_percentile,interpolation='nearest')).argmin()
                        wind_radii['rads'][wind_thresh].append(np.percentile(x_km, distance_percentile))
                    elif wind_radii_method == "max":
                        idist_of_wind_threshold = x_km.argmax(x_km.dims)
                        wind_radii['rads'][wind_thresh].append(x_km.max().data)
                    else:
                        logging.error(f"unexpected wind_radii_method: {wind_radii_method}")
                        sys.exit(1)
                else:
                    wind_radii['rads'][wind_thresh].append(0.*units.km)

            # leading zeros to keep same number of columns
            debugmsg = f'  get_ext_of_wind():   {wind_thresh}   {az:~03.0f}-{az+daz:~03.0f}  {iquad.sum().data:5d}  {wind_radii["rads"][wind_thresh][-1]:~03.0f}'
            if iquad.sum():
                debugmsg += f'  {bearing.isel(idist_of_wind_threshold).data:~03.0f}'
                debugmsg += f'  {latCell.isel(idist_of_wind_threshold).data:~03.1f}'
                debugmsg += f'  {lonCell.isel(idist_of_wind_threshold).data:~04.1f}'
            logging.debug(debugmsg)
                      
                      
        
    return wind_radii

def debugplot(row, lonCell, latCell, *zs, where=None):
    # *zs can be any number of arguments
    from scipy.interpolate import griddata
    npts = 100
    nz = len(zs)
    ncols = int(np.ceil(np.sqrt(nz)))
    nrows = int(np.ceil(nz/ncols))
    fig, axes = plt.subplots(nrows=nrows,ncols=ncols)
    xlim = row.lon + np.array([-2.25,2.25]) / np.cos(np.radians(row.lat))
    ylim = row.lat + [-2.25,2.25]
    xi = np.linspace(*xlim,npts)
    yi = np.linspace(*ylim,npts)
    if where is not None:
        x = lonCell.where(where,drop=True).to_numpy()
        y = latCell.where(where,drop=True).metpy.dequantify()
    else:
        x,y = lonCell,latCell.metpy.dequantify()
    x[x>=180] -= 360
    for ax,z in zip(axes.flatten(), [*zs]):
        if hasattr(z, "name"):
            label = z.name
        if hasattr(z, "long_name"):
            label = z.long_name
        z = z.where(where,drop=True).metpy.dequantify()
        zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')
        #ax.contour(xi, yi, zi)
        cntr1 = ax.contourf(xi,yi,zi, cmap="RdBu_r")
        ax.set_title(label)
        fig.colorbar(cntr1, ax=ax)
        #ax.plot(x,y, 'ko')
        ax.set(xlim=xlim, ylim=ylim)

    return fig

def derived_winds(u10, v10, mslp, lonCell, latCell, row, vmax_search_radius=250.*units.km, mslp_search_radius=100.*units.km, 
        wind_radii_method="max", pouter_search_radius=800*units.km, debug=False):
    # Given a row (with row.lon and row.lat)...
    # Derive cell distances and bearings
    distance, bearing = dist_bearing(row.lon*degE, row.lat*degN, lonCell, latCell)

    # Derive 10m wind speed and tangential wind speed Vt from u10 and v10
    speed = metpy.calc.wind_speed(u10,v10)

    # Tangential (cyclonic) wind speed
    # v * sin(bearing) - u * cos(bearing)
    Vt = ( v10 * np.sin(np.radians(bearing)) - u10 * np.cos(np.radians(bearing)) )
    if row.lat < 0:
        Vt = -Vt

    # Restrict Vmax search to vmax_search_radius
    vmaxrad = distance < vmax_search_radius
    ispeed_max = speed.where(vmaxrad).argmax(dim=speed.dims)
    raw_vmax =  speed.isel(ispeed_max).data

    # If vmax > 34, check if tangential component of max wind is negative (anti-cyclonic)
    if row.vmax > 34 and Vt.isel(ispeed_max) < 0:
        print(" TC center", row.valid_time, row.lat, row.lon)
        print(f"  max wind is anti-cyclonic! {Vt.isel(ispeed_max).data:~3.2f}")
        print(f"  max wind lat/lon           {latCell.isel(ispeed_max).data:~3.1f} {lonCell.isel(ispeed_max).data:~4.1f}")
        print(f"  max wind U/V               {u10.isel(ispeed_max).data:~3.2f} {v10.isel(ispeed_max).data:~3.2f}")
        if debug:
            fig = debugplot(row,lonCell,latCell,u10,v10,speed,mslp,where=vmaxrad)
            ofile = f"negVt.{lon2s(row.lon)}.{lat2s(row.lat)}.{row.valid_time.strftime('%Y%m%d%H')}.png"
            plt.savefig(ofile)
            print(f"created {os.path.realpath(ofile)}")

    # Check if average tangential wind within search radius is negative (anti-cyclonic)
    average_tangential_wind = Vt.where(vmaxrad).mean()
    if average_tangential_wind < 0:
        print(" TC center", row.valid_time, row.lat, row.lon, f" avg wind within vmax search radius is anti-cyclonic! {average_tangential_wind.data:~.2f}")
        if debug:
            fig = debugplot(row,lonCell,latCell,u10,v10,speed,mslp,where=vmaxrad)
            ofile = f"negaverageVt.{lon2s(row.lon)}.{lat2s(row.lat)}.{row.valid_time.strftime('%Y%m%d%H')}.png"
            plt.savefig(ofile)
            print(f"created {os.path.realpath(ofile)}")

    # Get radius of max wind
    raw_rmw = distance.isel(ispeed_max).data
    logging.debug(f" TC center {row.valid_time} {row.lat} {row.lon},  max wind {raw_vmax:~3.1f} @  {latCell.isel(ispeed_max).data:~3.1f}  {lonCell.isel(ispeed_max).data:~4.1f}  {raw_rmw:~3.0f}")

    # Restrict min mslp search
    mslprad = distance < mslp_search_radius
    raw_minp = mslp.where(mslprad).min().data

    # Get max extent of wind at thresh_kts thresholds.
    wind_radii = get_ext_of_wind(speed, distance, bearing, raw_vmax, latCell=latCell, lonCell=lonCell, wind_radii_method=wind_radii_method)

    # Restrict pressure of last closed isobar and radius of last closed isobar search to a certain radius.
    pouter_rad = distance < pouter_search_radius
    imslp_max = mslp.where(pouter_rad).argmax(dim=mslp.dims)
    raw_pouter =  mslp.isel(imslp_max).data
    raw_router =  distance.isel(imslp_max).data
    # Option 2
    # Zehr and Knaff (2007) https://journals.ametsoc.org/waf/article/22/1/71/38805/Reexamination-of-Tropical-Cyclone-Wind-Pressure
    # define pouter_rad as the annular average sea level pressure 800-1000 km.
    # I guess raw_router_nm (ROCI) is fixed at 900 km in this case.

    Vt_azimuthal_mean = get_azimuthal_mean(Vt, distance, binsize = 200*units.km)
    Vt_500km = Vt_azimuthal_mean.sel(radius = 500*units.km).data

    if Vt_500km < 0:
        logging.info(" atcf.derived_winds(): mean tangential wind 400-600km out is negative! TODO: figure out how to define storm_size_S.") 
    storm_size_S = Vt_500km / V500c(row.vmax*kt, row.lat*degN) # Should I use the input row.vmax or the raw_vmax from the raw model?
    # I originally put raw_vmax, but I changed to row.vmax. It depends which one you think is more accurate. 
    # For NARR at least, I think the input row.vmax is more accurate because NARR is coarse and biased low.
    penv_azimuthal_mean = get_azimuthal_mean(mslp, distance, binsize = 200*units.km)
    penv = penv_azimuthal_mean.sel(radius = 900*units.km).data

    derived_winds_dict = { # these values were xarrays, but I applied .data() to them already.
            "raw_vmax"     : raw_vmax,
            "raw_rmw"      : raw_rmw,
            "raw_minp"     : raw_minp,
            "wind_radii"   : wind_radii, 
            "raw_pouter"   : raw_pouter,
            "raw_router"   : raw_router,
            "Vt_500km"     : Vt_500km, # used for Zehr and Knaff Pmin derivation.
            "storm_size_S" : storm_size_S.to_base_units(), # used for Zehr and Knaff Pmin derivation.
            "penv"         : penv # used for Zehr and Knaff Pmin derivation.
            }
    return derived_winds_dict

def fullcircle_windradii(row):
    # full circle 34, 50, or 64 knot radius (max of rad1-rad4)
    # MET-TC will not derive this on its own - see email from John Halley-Gotway Oct 11, 2018
    # Probably shouldn't have AAA and NEQ in same file. 
    row.windcode = 'AAA'
    rads = row[['rad1','rad2','rad3','rad4']]
    row.rad1 = np.nanmax(rads) # rad1 is the maximum of rad1-rad4
    row[['rad2','rad3','rad4']] = np.nan # the rest are nans
    return row


def add_wind_rad_lines(row, wind_radii):
    lines = []
    for thresh in wind_radii['rads']:
        # row with 34, 50, or 64 knot radii
        newrow = row.copy()
        # 'rad' must be string. It is written as a string in write() method. 
        # It is a category not a float.
        newrow['rad'] = str(thresh.to("knot").m)
        newrow['windcode'] = wind_radii['windcode']
        newrow[['rad1','rad2','rad3','rad4']] = [np.round(x.to("nautical_mile").m) for x in  wind_radii['rads'][thresh]]
        lines.append(newrow)
    return pd.DataFrame(lines)

def origgridWRF(df, griddir, grid="d03", wind_radii_method = "max"):
    # Get vmax, minp, radius of max wind, max radii of wind thresholds from WRF by Alex Kowaleski
    
    # assert this is a single track
    assert df.groupby(['basin','cy','initial_time','model']).ngroups == 1, 'mpas.origmesh got more than 1 track'
    basin, cy, initial_time, model = df.iloc[0][['basin', 'cy', 'initial_time', 'model']]

    wregex = r'WF((\d\d)|(CO))'
    WRFmember = df.model.str.extract(wregex, flags=re.IGNORECASE)
    # column 0 will have match or null
    if pd.isnull(WRFmember.iloc[:,0]).any():
        logging.warning(f'Assuming WRF ensemble member, but not all model strings match {wregex}')
        pdb.set_trace()
    ens = WRFmember.iloc[0,0]
    df = df.groupby('fhr').apply(WRFraw_vitals,griddir, ens,wind_radii_method=wind_radii_method)
    df = df.droplevel('fhr')
    return df

def WRFraw_vitals(fhr, griddir, ens, wind_radii_method='max'):
    if 'origmesh' in time.userdata1:
        logging.info(f"wait. fhr {fhr.name} already has original mesh values. Skipping.")
        return fhr
    row = fhr.head(1).squeeze()
    gridfile = os.path.join(griddir, "/EPS_"+str(ens)+"/E"+str(ens)+"_"+row.initial_time.strftime('%m%d%H') + \
        "_"+grid+"_"+ row.valid_time.strftime('%Y-%m-%d_%H:%M:%S') +"_ll.nc")
    logging.debug(f'opening {gridfile}')
    ds = xarray.open_dataset(gridfile)
    ds = ds.isel(time=0)
    u10  = ds['u10']
    v10  = ds['v10']
    mslp = ds['slp']
    lonCell,latCell = np.meshgrid(ds.lon_0, ds.lat_0)
    lonCell = xarray.DataArray(lonCell*units(ds.lon_0.units), coords=u10.coords) # units and coordinates needed for derived_winds()
    latCell = xarray.DataArray(latCell*units(ds.lat_0.units), coords=u10.coords) # or else derived_winds() chokes on distance.isel()

    logging.debug(f"Extract vmax, RMW, minp, and radii of wind thresholds from row {row.name}")
    derived_winds_dict = derived_winds(u10, v10, mslp, lonCell, latCell, row, wind_radii_method=wind_radii_method)
    return row

def origgrid(df, griddir, ensemble_prefix="ens_", wind_radii_method="max"):
    # Get vmax, minp, radius of max wind, max radii of wind thresholds from ECMWF grid, not from tracker.
    # Assumes
    #   ECMWF data came from TIGGE and were converted from GRIB to netCDF with ncl_convert2nc.
    #   4-character model string in ATCF file is "EExx" (where xx is the 2-digit ensemble member).
    #   ECMWF ensemble member in directory named "<ensemble_prefix>xx" (where xx is the 2-digit ensemble member). 
    #   File path is "<ensemble_prefix>xx/${gs}yyyymmddhh.xx.nc", where ${gs} is the grid spacing (0p125, 0p15, 0p25, or 0p5).
    # ensemble_prefix may be a single string or a list of strings

    # assert this is a single track
    assert df.groupby(['basin','cy','initial_time','model']).ngroups == 1, 'mpas.origmesh got more than 1 track'
    basin, cy, initial_time, model = df.iloc[0][['basin', 'cy', 'initial_time', 'model']]


    if isinstance(ensemble_prefix, str):
        ensemble_prefixes = [ensemble_prefix]
    elif isinstance(ensemble_prefix, (list, tuple)):
        ensemble_prefixes = ensemble_prefix
    
    m = re.search(r'EE(\d\d)', model)
    if not m:
        logging.debug('Assuming ECMWF ensemble member, but did not find EE\d\d in model string')
        logging.debug(f'no original grid for {model} - skipping')
        return df
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
        gridfile = os.path.join(griddir,gridfile) # gridfile now has full path. Important for saving in atcf file. Or else os.path.realpath might use current directory.
        if os.path.isfile(gridfile):
            break
        else:
            logging.debug(f"no {gridfile}")

    df["originalmeshfile"] = gridfile
    logging.info(f'opening {gridfile}')
    ds = xarray.open_dataset(gridfile).metpy.quantify()
    df = df.groupby('fhr').apply(ECMWFraw_vitals,ds,wind_radii_method=wind_radii_method)
    df = df.droplevel('fhr')
    return df

def ECMWFraw_vitals(row, ds, wind_radii_method=None):
    row = row.head(1).squeeze() # make multiple wind rad lines one series. 
    forecast_time0 = pd.to_timedelta(row.fhr, unit='H')
    if forecast_time0 not in ds.forecast_time0:
        print(f"atcf.ECMWFraw_vitals(): fhr {row.fhr} not in original mesh. Dropping time.")
        return None # Don't return squeezed DataFrame (which is a Series now)
    ds = ds.sel(forecast_time0 = forecast_time0)
    u10  = ds["10u_P1_L103_GLL0"]
    v10  = ds["10v_P1_L103_GLL0"]
    mslp = ds["msl_P1_L101_GLL0"]
    lonCell,latCell = np.meshgrid(ds.lon_0, ds.lat_0)
    lonCell = xarray.DataArray(lonCell*units(ds.lon_0.units), coords=u10.coords) # units and coordinates needed for derived_winds()
    latCell = xarray.DataArray(latCell*units(ds.lat_0.units), coords=u10.coords) # or else derived_winds() chokes on distance.isel()

    # Extract vmax, RMW, minp, and radii of wind thresholds
    derived_winds_dict = derived_winds(u10, v10, mslp, lonCell, latCell, row, wind_radii_method=wind_radii_method)

    row = unitless_row(derived_winds_dict, row)

    row = add_wind_rad_lines(row, derived_winds_dict["wind_radii"])

    return row

# unitless_row() used in ECMWFraw_vitals() and mpas.raw_vitals()
def unitless_row(derived_winds_dict, row):
    for c in ["vmax", "minp", "rmw", "pouter", "router"]:
        row[c] = derived_winds_dict["raw_"+c].to(column_units[c]).m
    row["wind_radii_method"] = derived_winds_dict["wind_radii"]["wind_radii_method"]
    for c in ["penv", "Vt_500km"]: # Used in Knaff and Zehr wind pressure relationship.
        row[c] = derived_winds_dict[c].to(column_units[c]).m 
    return row
    

def ll_arc_distance(lat0=0*degN, lon0=0*degE, heading=0*deg, distance=0.*units.km, Rearth=Rearth):

    # Don't want to deal with Pandas indices. it will try to match them up, and you get NaNs and a bigger Series if they don't. 
    lon0 = np.radians(lon0)
    lat0 = np.radians(lat0)
    heading_rad = np.radians(heading)
    d_rad = distance/Rearth

    newlat = np.arcsin( np.sin(lat0)*np.cos(d_rad) +
            np.cos(lat0)*np.sin(d_rad)*np.cos(heading_rad))

    newlon = lon0 + np.arctan2(np.sin(heading_rad)*np.sin(d_rad)*np.cos(lat0),
                     np.cos(d_rad)-np.sin(lat0)*np.sin(newlat))


    return np.degrees(newlon), np.degrees(newlat)

def cross_track(track, cross):
    track = track.sort_values("valid_time")
    ptrack = track.copy()

    for i, row in track.iterrows():
        future_times = track.valid_time > row.valid_time
        if future_times.sum() == 0:
            break
        # Find the first future time. 
        next_time = track.valid_time == track.loc[future_times,"valid_time"].min()
        # Get lon/lat and elapsed time at next location.
        # With .mean(), reduce (possibly) multiple rows (for different wind radii) to one.
        lon1 = track.loc[next_time, "lon"].mean() * degE
        lat1 = track.loc[next_time, "lat"].mean() * degN
        # days since start of track
        dt = track.loc[next_time, "valid_time"].mean() - track.valid_time.min()
        dt_days = dt.total_seconds()/24/3600 * units("day")
        cross_track_error = cross * dt_days
        perpendicular_heading = ptrack.loc[next_time,"heading"].mean() *units("deg") + 90*units("deg")
       
        # head off perpendicular to track, starting from end of control segment 
        newlon, newlat = ll_arc_distance(lon0=lon1, lat0=lat1, distance=cross_track_error, heading=perpendicular_heading)
        # Assign one value to (possibly) multiple rows (for different wind radii).
        ptrack.loc[next_time, "lon"] = newlon.to(degE).m
        ptrack.loc[next_time, "lat"] = newlat.to(degN).m

    return ptrack



def veer_track(track, veer):
    track = track.sort_values("valid_time")
    ptrack = track.copy()

    for i, row in track.iterrows():
        lon0 = row.lon * degE
        lat0 = row.lat * degN
        valid_time = row.valid_time
        future_times = track.valid_time > valid_time
        if future_times.sum() == 0:
            logging.debug(f"no times later than {valid_time}. stop veering.")
            break
        # Find the first future time. 
        next_time = track.valid_time == track.loc[future_times,"valid_time"].iloc[0]
        # define control segment from time to next time
        # With .mean(), reduce (possibly) multiple rows (for different wind radii) to one.
        lon1 = track.loc[next_time, "lon"].mean() * degE
        lat1 = track.loc[next_time, "lat"].mean() * degN
        # days since start of track
        dt = track.loc[next_time, "valid_time"].mean() - track.valid_time.min()
        distance, heading = dist_bearing(lon0,lat0,lon1,lat1)
        dt_days = dt.total_seconds()/24/3600 * units("day") 
        new_heading = heading + veer * dt_days
       
        # start of perturbed segment is end of previous perturbed segment
        lon0 = ptrack.loc[ptrack.valid_time == valid_time, "lon"].mean() * degE
        lat0 = ptrack.loc[ptrack.valid_time == valid_time, "lat"].mean() * degN
        # head off in new direction for distance of control segment
        newlon, newlat = ll_arc_distance(lon0=lon0, lat0=lat0, distance=distance, heading=new_heading)
        # Assign one value to (possibly) multiple rows (for different wind radii).
        ptrack.loc[next_time, "lon"] = newlon.to("degrees_E").m
        ptrack.loc[next_time, "lat"] = newlat.to("degrees_N").m

    return ptrack



if __name__ == "__main__":
    df = read(ifile=sys.argv[1])
    #df = df.loc[df.valid_time >= pd.to_datetime("20121027")]  # delay the perturbation to SANDY
    #df = df.loc[df.valid_time <= pd.to_datetime("201210311200")]  # truncate end
    unique_track = ['basin', 'cy', 'initial_time', 'model']
    perts = np.arange(-2,3,1) * units.parse_expression("deg/day") # directional error
    perts = np.arange(-50,51,25) * units.parse_expression("km/day") # cross-track error
    ofile = f"perts{perts[0].m}-{perts[-1].m}"
    if os.path.exists(ofile+".dat"):
        logging.info(f"{ofile}.dat exists already. Will append")
    ax = get_ax()
    for track_id, track_df in df.groupby(unique_track):
        track_df = interpolate(track_df, '3H')
        logging.info(track_id)
        for PF, pert in enumerate(perts): 
            ptrack = cross_track(track_df, pert)
            ptrack["model"] = "PF{:02.0f}".format(PF)
            plot_track(ax, "", ptrack, "{:~3.1f}".format(pert), label_interval_hours=24, scale=0.8)
            write(ofile+".dat", ptrack, append=True)
    l = TClegend(ax.figure)
    plt.savefig(ofile + ".png", dpi=200)
    print(ofile+".dat")
    print(ofile+".png")

