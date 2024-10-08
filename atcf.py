""" tools to work with atcf """
from collections import OrderedDict
import csv
import logging
import os
import pdb
import re
import sys
import warnings

import cartopy
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import metpy.calc
import metpy.constants
from metpy.units import units
from metpy.constants import earth_avg_radius
import numpy as np
import pandas as pd
import pint
import xarray
from scipy.stats import circmean  # for averaging longitudes

import spc

basin_bounds = {
    "al": (-99, -22, 0, 38),
    "Gulf": (-100, -80, 21.5, 34),
    "Michael2018": (-93, -80, 21.5, 34),
    "Irma": (-89, -70, 20, 34),
    "Irma1": (-81.1, -78.6, 22.5, 25.2),
    "ep": (-175, -94, 0, 34),
    "cp": (150, -135+360, 0, 34),
    "io": (30, 109, 0, 28),
    "wp": (99, 180, 0, 38),
    # changed -180 to 179.999 to avoid UserWarning: Attempting to set identical left == right == 179.99999999999932 results in singular transformations
    "global": (-179.999, 180, -20, 70),
}


degE = units.parse_expression("degree_E")
degN = units.parse_expression("degree_N")
deg = units.parse_expression("degree")
kt = units.parse_expression("knot")
hPa = units.parse_expression("hPa")
nmi = units.parse_expression("nautical_mile")

# colors from tropicalatlantic.com
#           TD         TD            TS           CAT1          CAT2          CAT3         CAT4         CAT5
#   index    0          1             2             3             4             5            6            7
colors = ['white', (.01, .77, .18), (1.0, 1.0, .65), (1.0, .85, .85),
          (1.0, .67, .67), (1.0, .45, .46), (1, .24, .24), (0.85, .16, .17)]
# personal colors
colors = ['white', (.00, .61, .76), (.45, .75, .29), (.97, .94, .53),
          (.99, .75, .15), (.95, .46, .20), (1, .07, .07), (1.00, .09, .74)]
# more purple, yellow
colors = [(1, 1, 1), (.10, .66, .85), (.50, .80, .34), (.97, .94, .25),
          (.99, .75, .05), (.95, .46, .20), (1, .07, .07), (0.77, .09, .77)]

colors = {
    "TD": colors[1],
    "TS": colors[2],
    "CAT1": colors[3],
    "CAT2": colors[4],
    "CAT3": colors[5],
    "CAT4": colors[6],
    "CAT5": colors[7]
}

cmap = mcolors.ListedColormap(colors.values())

idl_water_color = np.array([235., 234., 242.])/255.


def contrasting_color(color):
    """ color could be a size-3 tuple, or a string, like "white" """
    assert mcolors.is_color_like(color)
    # to_rgba() returns a tuple, which has no attribute 'mean'
    if np.array(mcolors.to_rgba(color)).mean() >= 0.45:
        return 'black'
    return 'white'


quads = {
    'NEQ': [0, 90, 180, 270]*deg,
    'SEQ': [90, 180, 270, 0]*deg,
    'SWQ': [180, 270, 0, 90]*deg,
    'NWQ': [270, 0, 90, 180]*deg
}

wind_threshes = np.array([34, 50, 64]) * kt


def icy(cys):  # needed to define converters below.
    # return numeric portion of string, keeping as string.
    # tc_pairs doesn't match an adeck with cy=13L to a best track with cy=13
    return ''.join([i for i in cys if i.isdigit()])


converters = {
    "basin": lambda x: x.upper(),  # official is capitalized
    # The problem with CY is ATCF only reserves 2 characters for it.
    "cy": icy,  # cy is not always an integer (e.g. 10E)
    "initial_time": lambda x: pd.to_datetime(x.strip(), format='%Y%m%d%H'),
    # "vmax": float,
    # "minp": float,
    "ty": str,  # why was this not in here for so long?
    # not a number. it is a category. Important for interpolating in time.
    "rad": lambda x: x.strip(),
    "windcode": lambda x: x[-3:],
    "subregion": lambda x: x[-2:],
    # subregion ends up being 3 characters when written with .to_string
    # strange subregion only needs one character, but official a-decks leave 3.
    "initials": lambda x: x[-3:],
    'stormname': lambda x: x[-9:],
    'depth': lambda x: x[-1:],
    "seascode": lambda x: x[-3:],
    "userdefine1": str,
    "userdefine2": str,
    "userdefine3": str,
    "userdefine4": str,
}

dtype = {
    'technum': float,  # Tried int, but int can't be nan.
    "pouter": float,
    "rad1": float,
    "rad2": float,
    "rad3": float,
    "rad4": float,
    "router": float,
    'rmw': float,
    'gusts': float,
    'eye': float,
    'maxseas': float,
    "seas1": float,
    "seas2": float,
    "seas3": float,
    "seas4": float,
    'heading': float,
    'speed': float,
    "seas": float,
}

# Tried using converter for these columns, but couldn't convert 4-space string to float.
# If you add a key-value pair to na_values, also add it to dtype dict, and remove it from converters.
na_values = {
    "technum": 3*' ',
    "rad1": 4*' ',
    "rad2": 4*' ',
    "rad3": 4*' ',
    "rad4": 4*' ',
    "pouter": 4*' ',
    "router": 4*' ',
    "rmw": 4*' ',
    "gusts": 4*' ',
    "eye": 4*' ',
    "maxseas": 4*' ',
    "heading": 4*' ',
    "speed": 4*' ',
    "seas": 3*' ',  # one less than other columns
    "seas1": 4*' ',
    "seas2": 4*' ',
    "seas3": 4*' ',
    "seas4": 4*' ',
    "HollandB": 'Infinity',
    "B1": 'Infinity',
    "B2": 'Infinity',
    "B3": 'Infinity',
    "B4": 'Infinity',
}

# Standard ATCF columns (doesn't include track id, like in fort.66).
# https://www.nrlmry.navy.mil/atcf_web/docs/database/new/abrdeck.html
# Updated format https://www.nrlmry.navy.mil/atcf_web/docs/database/new/abdeck.txt
atcfcolumns = ["basin", "cy", "initial_time", "technum", "model", "fhr", "lat", "lon", "vmax", "minp", "ty",
               "rad", "windcode", "rad1", "rad2", "rad3", "rad4", "pouter", "router", "rmw", "gusts", "eye",
               "subregion", "maxseas", "initials", "heading", "speed", "stormname", "depth", "seas", "seascode",
               "seas1", "seas2", "seas3", "seas4", "userdefine1", "userdata1", "userdefine2", "userdata2",
               "userdefine3", "userdata3", "userdefine4", "userdata4"]


# for metpy.units.pandas_dataframe_to_unit_arrays(df, column_units=columns_units)
column_units = {x: 'dimensionless' for x in atcfcolumns}
column_units.update(
    dict(
        fhr=units.hour,
        lat=degN,
        lon=degE,
        vmax=kt,
        minp=hPa,
        rad=kt,
        rad1=nmi,
        rad2=nmi,
        rad3=nmi,
        rad4=nmi,
        pouter=hPa,
        router=nmi,
        rmw=nmi,
        gusts=kt,
        maxseas=units.feet,
        dir=deg,
        speed=kt,
        seas=units.feet,
        seas1=nmi,
        seas2=nmi,
        seas3=nmi,
        seas4=nmi,
        penv=hPa,
        penv_mb=hPa,
        Vt_500km=kt,
        Vt_500km_kts=kt,
    )
)

# Best tracks have different initial_times and fhr is always 0.
# To group best tracks with the same unique_list as for models, we will make change all best track
# initial_times to the first initial_time and make the best track fhr hold the time offset.
unique_track = ["basin", "cy", "initial_time", "model"]


def drop_wind_radii_all_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Drop lines with all zero wind radii if they are r50 or r64"""

    no_r50 = (df.rad == "50") & (df.rad1 == 0) & (df.rad2 == 0) & (df.rad3 == 0) & (df.rad4 == 0)
    if no_r50.sum():
        logging.warning(f"dropping {no_r50.sum()} r50 lines with all radii=0")
        df = df[~no_r50]
    no_r64 = (df.rad == "64") & (df.rad1 == 0) & (df.rad2 == 0) & (df.rad3 == 0) & (df.rad4 == 0)
    if no_r64.sum():
        logging.warning(f"dropping {no_r64.sum()} r64 lines with all radii=0")
        df = df[~no_r64]

    return df


def write(df, ofile, append=False):
    if df.empty:
        logging.warning("afcf.write(): DataFrame is empty.")

    # assert wind radii thresholds are in ascending order
    assert df.groupby(["basin","cy","initial_time", "model", "fhr"]).apply(
                lambda x : x.rad.is_monotonic_increasing
            ).all(), "wind radii thresholds not in ascending order"


    models = df["model"].unique()
    logging.debug(f"writing {models} to {ofile}")
    logging.debug(df.head(1))

    assert 0 not in df.columns, 'TODO: stop putting 0 column'

    # Extra columns not in atcfcolumns.
    extras = [x for x in df.columns if x not in atcfcolumns]
    extras.remove("valid_time")

    # warn if r50 or r64 line has wind radii all zero
    for r in [50,64]:
        radii_all_zero = (df.rad.astype(int) == r) & (df.rad1 == 0) & (df.rad2 == 0) & (df.rad3 == 0) & (df.rad4 == 0)
        assert radii_all_zero.sum() == 0, f"Found {radii_all_zero.sum()} r{r} lines with all radii=0"

    atcf_lines = ""
    for index, row in df.iterrows():
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
        with open(ofile, mode) as f:
            f.write(atcf_lines)

        if append:
            logging.info(f"appended {len(models)} models to {ofile}")
        else:
            logging.info(f"wrote {len(models)} models as {ofile}")


def new_besttrack_times(best_track):
    if "technum" in best_track and best_track.technum.any():
        logging.info("add minutes from TECHNUM")
        # add minutes for BEST tracks. 2-digit minutes are in the TECHNUM column for BEST tracks.
        # TECHNUM means something else for non-BEST tracks and shouldn't be added like a timedelta.
        extra_minutes = pd.to_timedelta(best_track.technum.fillna(
            0).replace('', 0).astype(int), unit="minutes")
        best_track['initial_time'] += extra_minutes
        best_track['valid_time'] += extra_minutes
        # Set technum to zero so this operation is not applied again.
        best_track["technum"] = 0

    # return unchanged best_track if this operation has been done already
    if "initial_time" in best_track and best_track.initial_time.nunique() == 1:
        expected_valid_time = best_track.initial_time + \
            pd.to_timedelta(best_track.fhr, unit='H')
        assert (best_track.valid_time == expected_valid_time).all(
        ), 'atcf.besttrack_fhr(): initial_time, fhr, and valid_time inconsistent'
        return best_track

    # Make initial_time the same across an entire best track and save the time difference in fhr.
    # Make initial_time all the same as first and store difference in fhr.
    # Helps when you group by unique_track later.
    first_time = best_track.valid_time.min()
    best_track["fhr"] = best_track.valid_time - first_time
    best_track["fhr"] /= np.timedelta64(1, 's')*3600
    best_track["initial_time"] = first_time
    return best_track


def decorate_ax(ax, bscale='50m'):
    # bscale = countries and ocean the same border scale to match
    # Create a feature countries from Natural Earth
    countries = cartopy.feature.NaturalEarthFeature(category='cultural',
                                                    name='admin_0_countries', scale=bscale,
                                                    facecolor=cartopy.feature.COLORS['land'])
    alpha = 0.5
    ocean = cartopy.feature.NaturalEarthFeature(category='physical',
                                                name='ocean', scale=bscale, edgecolor='face',
                                                # facecolor=cartopy.feature.COLORS['water'])
                                                facecolor=idl_water_color,
                                                alpha=alpha)

    #ax.add_feature(ocean)
    ax.set_aspect("auto")
    ax.add_feature(countries, edgecolor='gray', lw=0.375)
    ax.add_feature(cartopy.feature.LAKES, facecolor=idl_water_color, alpha=alpha)
    ax.add_feature(cartopy.feature.STATES.with_scale(bscale), lw=0.25, edgecolor='gray')
    # commented out gridlines because if you try creating another gridliner, it is hard to set parameters on it.

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

    return category


def iswarmcore(track, min_warmcore_percent=25):
    if 'warmcore' not in track.columns:
        logging.error("No warm core column.")
        sys.exit(1)
    s = warmcore = track.warmcore.str.strip()
    warmcore = s == 'Y'
    known = s != 'U'  # not unknown
    # If warmcore column exists, make sure at least one time is warmcore or unknown.
    if any(track.warmcore.str.strip() == 'U'):
        logging.warning("warm core unknown")
        return True
    warmcore_percent = 100*warmcore.sum()/known.sum()
    return warmcore_percent >= min_warmcore_percent


def iswind_radii_method(s:str):
    # make sure option is 'max', 'azimuthal_mean', or 'xpercentile' where x is a number
    if s in ('max', 'azimuthal_mean'):
        return s
    assert s.endswith('percentile'), "String must end with 'percentile'"
    s = s.replace('percentile','')
    s = float(s)
    return s


# 'cpsB':  Cyclone Phase Space "Parameter B" for thermal asymmetry. (Values are *10)
# 'cpsll':  Cyclone Phase Space lower level (600-900 mb) thermal wind parameter, for diagnosing low-level warm core. (Values are *10)
# 'cpsul': Cyclone Phase Space upper level (300-600 mb) thermal wind parameter, for diagnosing upper-level warm core. (Values are *10)
cyclone_phase_space_columns = ["cpsB", "cpsll", "cpsul"]

def V500c(Vmax: pint.Quantity, latitude: pint.Quantity) -> pint.Quantity:
    # Climatological tangential wind 500 km from the center
    # Zehr and Knaff, equations 4, 5, 6
    # https://journals.ametsoc.org/waf/article/22/1/71/38805/Reexamination-of-Tropical-Cyclone-Wind-Pressure
    # Added units so shape_factor_x is dimensionless
    # Make sure you subtract 25.0 * degN and not just 25.0 or elsel it assumes 25 is in radians and you get
    # radians in return.
    shape_factor_x = 0.1147 + 0.0055 / kt * Vmax - 0.001 / degN * (latitude - 25.0 * degN)
    Rmax = (
        66.785 * units.km
        - 0.09102 * units.km / kt * Vmax
        + 1.0619 * units.km / degN * (latitude - 25.0 * degN)
    )
    climatological_tangential_wind_500_km_from_the_center = (
        Vmax * (Rmax / (500.0 * units.km)) ** shape_factor_x
    )

    if np.ndim(Vmax) == 0:  # deal with scalar
        if Vmax < 15.0 * kt:
            return Vmax
    else:  # deal with iterables
        if any(Vmax < 15.0 * kt):
            logging.info("atcf.V500c(): leave Vmax < 15kt unchanged")
            climatological_tangential_wind_500_km_from_the_center[Vmax < 15 * kt] = Vmax[
                Vmax < 15 * kt
            ]
    return climatological_tangential_wind_500_km_from_the_center

def Knaff_Zehr_Pmin(
    Vsrm1: pint.Quantity,
    storm_size_S: pint.Quantity,
    latitude: pint.Quantity,
    environmental_pressure: pint.Quantity,
) -> pint.Quantity:
    """
    Eq 1 in Courtney and Knaff http://rammb.cira.colostate.edu/resources/docs/Courtney&Knaff_2009.pdf
    Unitize coefficients so Vsrm1, latitude, environmental_pressure can have any units
    """
    Pc = (
        23.286 * hPa
        - 0.483 * hPa / kt * Vsrm1
        - (Vsrm1 / (24.254 * kt / units.parse_expression("hPa**0.5"))) ** 2.0
        - 12.587 * hPa * storm_size_S
        - 0.483 * hPa / degN * latitude
        + environmental_pressure
    )
    return Pc

def Vsrm1(Vmax: pint.Quantity, storm_motion_C: pint.Quantity) -> pint.Quantity:
    """Eq 2 in Courtney and Knaff (2009)
    1-min average Vmax adjusted for storm motion

    Parameters
    ----------
    Vmax : pint.Quantity
        1-min average Vmax
    storm_motion_C : pint.Quantity
        storm motion

    Returns
    -------
    quantity : pint.Quantity

    TODO: Replace NaN with zero?
    Vmax and storm_motion_C are unitized speeds. Doesn't matter what units.
    """
    return Vmax - 1.5 * kt**(1-0.63) * storm_motion_C**0.63

def fill_speed_heading(track):
    assert track.groupby(
        unique_track).ngroups == 1, 'atcf.fill_speed_heading() given multiple tracks'
    # reduce to one wind radius for each valid_time
    oneradtrack = track.groupby("valid_time").first().reset_index()
    if len(oneradtrack) == 1:
        return track  # no way to get distance and heading from one point.
    # get speed and heading of track for one wind radius.
    lon = oneradtrack.lon.values * degE
    lat = oneradtrack.lat.values * degN
    # .values so speed_heading can use numpy array indices, not iloc
    time = oneradtrack["valid_time"].values
    # return speed and heading from previous point to next point
    speed, bearing = speed_heading(lon, lat, time)
    oneradtrack.loc[:, "speed"] = speed.to("knots").m
    oneradtrack.loc[:, "heading"] = bearing
    # Copy oneradtrack speed and heading to original track (possibly with multiple wind radii lines)
    track = track.set_index("fhr")
    track.loc[:, ["speed", "heading"]] = oneradtrack.set_index("fhr")[
        ["speed", "heading"]]
    track = track.reset_index()  # put fhr back in column
    return track


def speed_heading(lon, lat, time):
    assert lon.units == degE, "atcf.speed_heading(): lon not deg E"
    assert lat.units == degN, "atcf.speed_heading(): lat not deg N"
    # return speed and heading from previous point to next point
    speed = np.zeros_like(lon) * units.m / units.s
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
        d, b = dist_bearing(lon[first_point], lat[first_point],
                            lon[second_point], lat[second_point])
        bearing[t] = b
        dt = time[second_point] - time[first_point]
        if isinstance(dt, np.timedelta64):
            dt /= np.timedelta64(1, "s")  # np.timedelta64
        else:
            dt = dt.total_seconds()  # datetime.timedelta
        dt *= units.s
        speed[t] = d/dt

    return speed, bearing


def get_stormname(df):
    from stormevents.nhc import VortexTrack

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        first = df.iloc[0]
        stormname = VortexTrack(
            f"{first.basin}{first.cy}{first.valid_time.year}").name
        return stormname


def mean_track(df):
    """
    When aggregating numbers, take the mean.
    When aggregating strings or categories, take the max.
    Tried mode but it sometimes returned more than one value.
    """
    agg_dict = {
        "technum": 'mean',
        "lat": 'mean',
        "lon": 'mean',
        "vmax": 'mean',
        "minp": 'mean',
        "ty": pd.Series.max,
        "rad1": 'mean',
        "rad2": 'mean',
        "rad3": 'mean',
        "rad4": 'mean',
        "pouter": 'mean',
        "router": 'mean',
        "rmw": 'mean',
        "gusts": 'mean',
        "eye": 'mean',
        "subregion": pd.Series.max,
        "maxseas": 'mean',
        "initials": pd.Series.max,
        "heading": 'mean',
        "speed": 'mean',
        'stormname': pd.Series.max,
        'depth': pd.Series.max,
        "seas": 'mean',
        "seas1": 'mean',
        "seas2": 'mean',
        "seas3": 'mean',
        "seas4": 'mean',
        "userdefine1": pd.Series.max,
        "userdata1": pd.Series.max,
    }

    # Optional columns. If they are not in df, you get KeyError in aggregate.
    # Aggregate with pd.Series.max
    agg_dict.update(
        {x: pd.Series.max for x in df.columns if x.startswith("user")})
    # made fhr optional - it may be part of index
    if "fhr" in df:
        agg_dict["fhr"] = 'mean'

    dfg = df.groupby(['basin', 'cy', 'initial_time',
                     'valid_time', 'windcode', 'rad', 'seascode'])
    df = dfg.agg(agg_dict)
    df.reset_index(inplace=True)
    df["model"] = "MEAN"
    return df

def plot_track(ax,
        start_label,group,end_label, scale=1,
        label_interval_hours=1, onecolor=None,
        label=None,
        **kwargs):
    logging.debug(f"plot_track: {start_label} {group}")
    group = group.sort_values("valid_time")

    lformat = None
    group["TCcategory"] = [vmax2category(x*kt) for x in group.vmax]
    group["color"] = [colors[c] for c in group["TCcategory"]]
    if onecolor:
        group["color"] = onecolor
    for i in range(0, len(group)):
        row = group.iloc[i]
        if i == 0:
            # first half-segment
            row_1 = group.iloc[i]
            if len(group) == 1:  # track is 1-point-long
                row1 = group.iloc[i]
            else:
                row1 = group.iloc[i+1]
            ax.text(row.lon, row.lat, start_label, clip_box=ax.bbox, clip_on=True,
                    ha='center', va='center', fontsize="small", transform=cartopy.crs.PlateCarree())
        elif i == len(group)-1:
            # last half-segment
            row_1 = group.iloc[i-1]
            row1 = group.iloc[i]
            ax.text(row.lon, row.lat, end_label, clip_box=ax.bbox, clip_on=True,
                    ha='center', va='center', fontsize="x-small", transform=cartopy.crs.PlateCarree())
        else:
            # middle segments
            row_1 = group.iloc[i-1]
            row1 = group.iloc[i+1]
        lat0 = (row_1.lat + row.lat) / 2.
        lon0 = circmean([row_1.lon, row.lon], low=-180, high=180)
        lat1 = (row.lat + row1.lat) / 2.
        lon1 = circmean([row.lon, row1.lon], low=-180, high=180)
        color = row.color
        # Include middle point in line segments. Full segment is a bent line.
        lons = np.array([lon0, row.lon, lon1])
        lats = [lat0, row.lat, lat1]
        dlon = max(lons) - min(lons)
        span_dateline = dlon > 180.
        if span_dateline:
            logging.warning(f"{row.cy} spans dateline")
            lons[lons < 0] += 360
        logging.debug(f"{i} plot segment {lons} {lats}")

        ax.plot(lons, lats, c=color,
                transform=cartopy.crs.PlateCarree(),
                label=label, **kwargs)

    if label_interval_hours is None:
        return ax
    # label if hour is multiple of label_interval_hours
    # outside segment loop so it remains on top of segments
    for i, row in group[group.valid_time.dt.hour % label_interval_hours == 0].iterrows():
        color = row.color
        ax.plot(row.lon, row.lat, 'o', markersize=scale*3, markerfacecolor=color, color=color,
                transform=cartopy.crs.PlateCarree(), **kwargs)
        lformat = "%-d"
        if row.valid_time.hour != 0:
            lformat = "%-Hz"
        ax.text(row.lon, row.lat, row.valid_time.strftime(lformat),
                color=contrasting_color(color), clip_box=ax.bbox, clip_on=True,
                ha='center', va='center_baseline', fontsize=scale*3.6, transform=cartopy.crs.PlateCarree())
    return ax

def TClegend(**kwargs):
    n = len(colors)
    ax = plt.gcf().get_axes()  # all the axes (1 or more)
    cbar = plt.cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=0, vmax=n), cmap=cmap)
    axCbar = plt.colorbar(cbar, ax=ax, orientation="horizontal", drawedges=True,
                          **kwargs)
    axCbar.set_ticks(np.array(range(n))+0.5)
    axCbar.set_ticklabels(colors.keys())
    return axCbar


def interpolate_by_rad(rad, interval):
    """
    Interpolate in time
    This should be from one wind speed threshold
    """
    logging.debug(f"atcf.interpolate(): padding {rad.name}")
    rad = rad.set_index('valid_time').resample(
        interval).interpolate(method='pad')
    # redo model track circular heading
    speed, heading = speed_heading(
        rad.lon.values*degE, rad.lat.values*degN, rad.index)
    rad["heading"] = heading
    rad = rad.reset_index()  # return valid_time from index to column
    return rad


def interpolate(df, interval):

    assert interval in ['1H', '3H', '6H', '12H', '24H'], (
        f"atcf.interpolate(): unexpected time interval {interval}"
        "Expected one of ['1H','3H','6H','12H','24H']"
    )

    # Copied from ~ahijevyc/bin/interpolate_atcf.py on Mar 2, 2020. This method should supercede that script.

    logging.debug(f"atcf.interpolate() {interval}")

    nbad = df.valid_time.isna().sum()
    if nbad > 0:
        logging.info(f"Dropping {nbad} wind radii line(s) with NaN valid_time")
        df.dropna(how='all', subset=['valid_time'], inplace=True)

    # Include 34, 50 and 64 knot wind radii for each time, so missing radii are explicitly zero km when interpolating.
    df = return_expandedwindradii(df, wind_threshes.m.astype(str))

    # Interpolate in time
    # handle multiple models, init_times, etc.
    byrad = ['basin', 'cy', 'initial_time', 'model', 'rad']
    df = df.groupby(byrad, as_index=False, group_keys=False).apply(
        interpolate_by_rad, interval)
    df = df.sort_values(
        by=['basin', 'cy', 'initial_time', 'model', 'valid_time', 'rad'])
    return df


def stringlatlon2float(slon, slat):
    """
    Extract last character of slon and slat input.
    Convert to float and divide by 10.
    Multiply by -1 if slon ends with "W"
    Multiply by -1 if slat ends with "S"
    """
    lon = slon.str[:-1].astype(float) / 10.
    lat = slat.str[:-1].astype(float) / 10.
    lon[slon.str.endswith('W')] *= -1
    lat[slat.str.endswith('S')] *= -1

    logging.debug("converted string lat/lons to float")
    return lon, lat



def expand_wind_radii(df, rads):
    """ Define all wind radii, regardless of wind speed. """
    # rad1-4 and seas1-4 columns
    rad_seas = []
    for f in ["rad", "seas"]:
        for r in ["1", "2", "3", "4"]:
            rad_seas.append(f+r)

    df = df.set_index("rad").reindex(rads)
    # first fill missing rad1-4 and seas1-4 with zeros before forward-filling na's
    df[rad_seas] = df[rad_seas].fillna(value=0)
    df = df.ffill()
    return df


def return_expandedwindradii(df, rads):
    """change 0kt to 34kt"""
    df.loc[df["rad"] == "0", "rad"] = "34"
    df = df.groupby(['basin', 'cy', 'initial_time', 'model', 'fhr'],
                    group_keys=False).apply(expand_wind_radii, rads)
    df = df.reset_index()  # return index (rad) to column
    return df


def add_missing_dummy_columns(df, columns):
    dummy = {}
    for col in columns:
        if col in df.columns:
            continue
        logging.debug(f"{col} not in DataFrame. Fill with appropriate value.")

        # if column doesn't exist make it zeroes
        if col in ['rad1', 'rad2', 'rad3', 'rad4', 'pouter', 'router', 'seas', 'seas1', 'seas2', 'seas3', 'seas4']:
            dummy[col] = 0.

        # if rad column doesn't exist make it string zero.
        elif col in ['rad']:
            dummy[col] = '0'

        # Initialize other default values.
        elif col in ['windcode', 'seascode']:
            dummy[col] = '   '

        # Numbers are NaN - change to 3*' ' in write
        elif col in ['rmw', 'gusts', 'eye', 'maxseas', 'heading', 'speed']:
            dummy[col] = np.NaN

        # Strings are empty
        elif col in ['subregion', 'stormname'] or col.startswith('user'):
            dummy[col] = ''

        elif col in ['ty']:
            dummy[col] = 'XX'

        elif col in ['initials', 'depth']:
            dummy[col] = 'X'
        else:
            logging.error(
                f"atcf.add_missing_dummy_columns(): unexpected col {col}")
            sys.exit(1)
    dummy = pd.DataFrame(dummy, index=df.index)  # propagate values to all rows
    df = pd.concat([df, dummy], axis="columns")
    return df


def read_aswip(ifile):
    # Read data into Pandas Dataframe
    logging.debug(f'Reading {ifile}')

    # https://adcirc.org/home/documentation/users-manual-v50/input-file-descriptions/single-file-meteorological-forcing-input-fort-22/
    #           1      2         3              4         5       6      7      8      9      10     11
    names = ["basin", "cy", "initial_time", "technum", "model", "fhr", "lat", "lon", "vmax", "minp", "ty",
             # 12      13         14      15       16     17       18        19       20     21       22
             "rad", "windcode", "rad1", "rad2", "rad3", "rad4", "pouter", "router", "rmw", "gusts", "eye",
             #    23          24          25         26         27        28
             "subregion", "maxseas", "initials", "heading", "speed", "stormname",
             #        29               30         31      32      33      34      35       36       37       38       39
             "time_record_number", "nisotachs", "use1", "use2", "use3", "use4", "rmax1", "rmax2", "rmax3", "rmax4", "HollandB",
             #40    41    42    43     44       45       46       47
             "B1", "B2", "B3", "B4", "vmax1", "vmax2", "vmax3", "vmax4"]

    df = pd.read_csv(ifile, index_col=None, header=None, delimiter=",", names=names, converters=converters,
                     na_values=na_values, dtype=dtype, skipinitialspace=True, engine='c')  # engine='c' is faster than engine="python"

    # convert string lat lon column to float. So we can write atcf file. valid_time not needed by write() method..
    df["lon"], df["lat"] = stringlatlon2float(df.lon, df.lat)

    # Add missing ATCF columns with dummy data.
    df = add_missing_dummy_columns(df, atcfcolumns)

    # Derive valid time.   valid_time = initial_time + fhr (used in read_aswip too)
    df['valid_time'] = df.initial_time + pd.to_timedelta(df.fhr, unit='h')

    return df


def read(ifile):
    """ Read data into Pandas Dataframe"""
    logging.debug(f'Reading {ifile}')

    # make a copy of list, not a copy of the reference to the list.
    names = list(atcfcolumns)

    with csv.reader(open(ifile), delimiter=',', encoding="utf-8") as reader:
        testline = next(reader)
        num_cols = len(testline)
        logging.debug(f"test line num_cols {num_cols}")
        logging.debug(testline)
    with open(ifile) as f:
        max_num_cols = max(len(line.split(',')) for line in f)
        logging.debug(f"max number of columns {max_num_cols}")

    # Output from GFDL vortex tracker, fort.64 and fort.66
    # are mostly ATCF format but have subset of columns
    if num_cols == 43:
        logging.info(
            f'assume GFDL tracker fort.64-style output with 43 columns in {ifile}')
        TPstr = "THERMO PARAMS"
        assert testline[35].strip(
        ) == TPstr, f"expected 36th column to be {TPstr}. got {testline[35].strip()}"
        for ii in range(20, 35):
            # duplicate names not allowed
            names[ii] = "space filler" + str(ii-19)
        names = names[0:35]
        names.append(TPstr)
        names.extend(cyclone_phase_space_columns)
        names.append('warmcore')
        names.append("warmcore_strength")
        names.append("string1")
        names.append("string2")

    # fort.66 has track id in the 3rd column.
    if num_cols == 31:
        logging.info(
            f'assume GFDL track fort.66-style with 31 columns in {ifile}')
        # There is a cyclogenesis ID column for fort.66
        logging.debug('inserted ID for cyclogenesis in column 2 (zero-based)')
        names.insert(2, 'id')  # ID for the cyclogenesis
        logging.info('Using 1st 21 elements of names list')
        names = names[0:21]
        logging.debug('redefining columns 22-31')
        names.extend(cyclone_phase_space_columns)
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
        logging.info(f"assume {ifile} is simple adeck with 11 columns")
        if not ifile.endswith('.dat'):
            logging.warning(f"even though file doesn't end in .dat")
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
    logging.debug(names)
    logging.debug(testline)
    # logging.debug(f"converters={converters}")
    # logging.debug(f"dype={dtype}")
    # logging.debug(f"column_units={column_units}")

    df = pd.read_csv(ifile,index_col=None,header=None, delimiter=",", usecols=usecols, names=names,
            converters=converters, na_values=na_values, dtype=dtype, skipinitialspace=True, engine='c') # engine='c' is faster than engine="python"

    # fort.64 has asterisks sometimes. Problem with hwrf_tracker.
    badlines = df['lon'].str.contains("*", regex=False)
    if any(badlines):
        logging.warning(f"tossing {len(badlines)} lines with asterisk in lon")
        df = df[~badlines]

    df["lon"], df["lat"] = stringlatlon2float(df.lon, df.lat)

    if max_num_cols != num_cols and df.model.nunique() > 1:
        logging.warning(
            f"test line has {num_cols} columns, but another line has {max_num_cols} columns. Unexpected results may occur.")
        logging.warning(
            "It's hard to deal with userdefined columns and data in a file with multiple types of models.")

    # Derive valid time.   valid_time = initial_time + fhr (used in read_aswip too)
    df.loc[:, 'valid_time'] = df.initial_time + \
        pd.to_timedelta(df.fhr, unit='h')

    # adjust best track times: 1) add technum (2-digit minutes) and set technum to zero
    # 2) make initial_times all the same as the first and put time offset in fhr.
    # This makes it possible to groupby best tracks with same unique_track list as model tracks.
    isbesttrack = df.model == 'BEST'
    df.loc[isbesttrack] = df.loc[isbesttrack].groupby(
        ["basin", "cy", "model"], group_keys=False).apply(new_besttrack_times)

    # Prior to 1999, rad column is blank. Fill with numeric string.
    # Downstream programs assume rads are convertable to floats. Empty strings are not convertable to floats.
    if "rad" not in df.columns or all(df.rad == ''):
        logging.warning(
            "atcf.read(): No wind radii. This happens in pre-2000 files. Change to '34' so we can convert to float downstream")
        df["rad"] = '34'

    # sanity check for rad values
    if not all(df.rad.isin(['0', '34', '50', '64'])):
        logging.warning(
            f"atcf.read(): unexpected rad value(s) in atcf file {ifile}")
        logging.warning(df.rad.value_counts())
        # adecks before 2002 had 35-knot lines, not 34. That's okay.
        assert df.valid_time.min(
        ).year <= 2001, "atcf.read(): this should not happen with post-2001 files."

    # Add missing ATCF columns with dummy data.
    df = add_missing_dummy_columns(df, atcfcolumns)

    missing_speed_heading = all(((df.speed == 0) | pd.isnull(df.speed)) & (
        (df.heading == 0) | pd.isnull(df.heading)))  # assume bad if everything is zero
    if missing_speed_heading:
        logging.debug("derive speed and heading")
        df = df.groupby(unique_track, group_keys=False).apply(
            fill_speed_heading)

    # Put userdefine/userdata column pairs into single columns.
    # Allow unaligned userdefine columns from multiple atcf files.
    # ["1", "2", "3", ... ]
    for x in [str(x) for x in range(1, usercolumnindex)]:
        ndefinitions = df["userdefine"+x].nunique()
        if ndefinitions > 10:
            logging.info(
                f"{ndefinitions} definitions for userdefine{x} column. Skipping")
            continue
        # for each unique userdefine value in column
        for v in df["userdefine"+x].unique():
            if v == "":
                continue  # Ignore empty strings
            # identify rows with this userdefine value
            rows = df["userdefine"+x] == v
            # identify userdata associated with this userdefine value
            userdata = df.loc[rows, "userdata"+x]
            # set values of a column (named after the value in userdefine) to userdata value.
            df.loc[rows, v] = userdata
        # Drop this pair of userdefine/userdata columns.
        df = df.drop(columns=[u+x for u in ["userdefine", "userdata"]])

    return df


def f2s(x):  # float to string
    """
    Convert absolute value of float to integer number of tenths for ATCF lat/lon
    called by lat2s and lon2s
    """
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


def dist_bearing(lon1, lat1, lons, lats, Rearth=earth_avg_radius):
    """
    function to compute great circle distance between point lat1 and lon1 and arrays of points 
    INPUT:
        lon1 - longitude of origin with units
        lat1 - latitude of origin with units
        lons - longitudes of points to get distance to. Could be DataArray or numpy array with units
        lats - latitudes of points to get distance to
    Returns 2 things:
        1) distance (pint quantity with units km)
        2) initial bearing from 1st pt (lon1, lat1) to an array of other points (lons, lats). (also pint quantity)
    """

    if False:
        # TODO: get array shapes working, and xarray handling
        geo = cartopy.geodesic.Geodesic(radius=Rearth.to('meter').m)
        lons = [x.m for x in lons]
        lats = [x.m for x in lats]
        n3 = geo.inverse((lon1.m, lat1.m), np.stack([lons, lats], axis=-1))
        dist = n3[:, 0] * units.m
        bearing = n3[:, 1] * units.deg
        return dist, bearing

    # MAYBE DELETE EVERYTHING BELOW IN THIS FUNCTION
    assert lat1 <= 90*deg, f"lat1 {lat1} >  90deg"
    assert lat1 >= -90*deg, f"lat1 {lat1} < -90deg"
    # allow scalar lons, lats and xarrays
    if np.ndim(lons) == 0 and np.ndim(lats) == 0:
        lons, lats = np.atleast_1d(lons, lats)
    # Don't lose xarray characteristics with np.atleast_1d
    # lons, lats = np.atleast_1d(lons, lats)
    assert lats.max() < 90, "lats element > 90"
    assert lats.min() > -90, "lats element < -90"
    if hasattr(lons, 'metpy'):
        lons = lons.metpy.quantify()
        lats = lats.metpy.quantify()
    # great circle distance.
    arg = np.sin(lat1)*np.sin(lats)+np.cos(lat1)*np.cos(lats)*np.cos(lon1-lons)
    # arg = np.where(np.fabs(arg) < 1., arg, 0.999999) # sometimes arg = 1.000000000000002
    if (np.fabs(arg) > 1).any():
        logging.debug(
            f"atcf.dist_bearing(): minarg={arg.min()} maxarg={arg.max()}")

    if hasattr(arg, "where"):
        # Use xarray.DataArray.where to preserve DataArray coordinates
        # sometimes arg = 1.000000000000002
        arg = arg.where(arg <= 1., other=1.)
        arg = arg.where(arg >= -1., other=-1.)
    else:
        # sometimes arg = 1.000000000000002
        arg = np.where(arg <= 1., arg,  1.)
        arg = np.where(arg >= -1., arg, -1.)

    dlon = lons-lon1
    bearing = np.arctan2(np.sin(dlon)*np.cos(lats), np.cos(lat1)
                         * np.sin(lats) - np.sin(lat1)*np.cos(lats)*np.cos(dlon))

    # -180 - 180 -> 0 - 360
    # parentheses around 360*deg are important
    bearing = (bearing + 360*deg) % (360*deg)

    assert (np.fabs(arg) <= 1).all(), f"atcf.dist_bearing(): arg={arg}"

    distance_from_center = np.arccos(arg)*Rearth
    # Treating DataArrays and pint arrays separately sure is getting kludgy.
    if hasattr(lons, 'metpy'):
        bearing = bearing.metpy.convert_units("degrees")
        distance_from_center = distance_from_center.metpy.convert_units("km")
    else:
        bearing = bearing.to("degrees")
        distance_from_center = distance_from_center.to("km")

    return distance_from_center, bearing


def get_azimuthal_mean(x, distance, binsize=25.*units.km):
    # Same units for distance and binsize
    distance = distance.metpy.convert_units(binsize.units)
    # Don't use xarray or pint quanity, because these aren't implemented in np.histogram.
    bins = np.arange(0, distance.data.max().m, binsize.m)
    n, bin_edges = np.histogram(distance, bins=bins)
    assert (bin_edges == bins).all()
    if (n == 0).any():
        logging.error(
            f"get_azimuthal_mean: no pts b/t {bins[n == 0]} and {bins[n == 0] + binsize}")
        sys.exit(1)
    if (n == 1).any():
        logging.info(
            f"get_azimuthal_mean: only 1 pt b/t {bins[n == 1]} and {bins[n == 1] + binsize}")
    # Depreciation warning about ragged arrays if you add units here
    h, _ = np.histogram(distance, bins=bins, weights=x)
    x_vs_radius = h/n
    bin_centers = bins[:-1] * binsize.units + binsize/2
    bin_centers *= binsize.units
    x_vs_radius *= x.metpy.units
    da = xarray.DataArray(data=x_vs_radius, coords={"radius": bin_centers})
    return da


def get_ext_of_wind(wind_speed, distance, bearing, raw_vmax, windcode='NEQ', wind_threshes=wind_threshes,
                    rad_search_radius=300.*nmi, lonCell=None, latCell=None, wind_radii_method='max'):
    """
    Return wind_radii dictionary where
    wind_radii = {
      wind_radii_method : wind_radii_method,
               windcode : windcode,
               raw_vmax : raw_vmax,
                    rads: ordered_dict{
                              wind_threshes[0]:  [rad1,rad2,rad3,rad4],
                              ...
                              wind_threshes[-1]: [rad1,rad2,rad3,rad4],
                          }
    }
    """
    assert wind_radii_method in ["azimuthal_mean", "max", "percentile"], (
        f"unexpected wind_radii_method: {wind_radii_method}")

    wind_radii = {"wind_radii_method": wind_radii_method}
    wind_radii['raw_vmax'] = raw_vmax
    wind_radii['windcode'] = windcode
    wind_radii['rads'] = OrderedDict()
    if raw_vmax < wind_threshes[0]:
        # write out zero rads for first wind threshold and return
        wind_radii['rads'][wind_threshes[0]] = [0, 0, 0, 0]*units.km
        return wind_radii
    # Originally had distance < 800km, but Chris D. suggested 300nm in Sep 2018 email
    # This was to deal with Irma and the unrelated 34 knot onshore flow in Georgia
    # Looking at HURDAT2 R34 sizes (since 2004), ex-tropical storm Karen 2015 had 710nm.
    # Removing EX-tropical storms, the max was 480 nm in Hurricane Sandy 2012
    # see /glade/work/ahijevyc/atcf/R34noEX.png and R34withEX.png
    # wind_speed is masked beyond rad_search_radius
    wind_speed = wind_speed.where(distance < rad_search_radius)

    logging.debug(
        f'  get_ext_of_wind(): method {wind_radii_method} windcode {windcode}')
    logging.debug(
        '  get_ext_of_wind(): wind_thresh     azimuth       npts    dist   bearing      lat        lon')

    for wind_thresh in wind_threshes:
        if (wind_speed >= wind_thresh).sum() == 0:
            return wind_radii
        if distance.ndim == 2:
            imax = distance.where(
                wind_speed >= wind_thresh).argmax(distance.dims)
            # warn if max_dist_of_wind_threshold is on edge of 2-d domain (like nested WRF grid)
            logging.debug(f"  get_ext_of_wind(): imax {imax}")
            for dim, i in imax.items():
                if i == 0 or i == distance[dim].size-1:
                    print(
                        f"  get_ext_of_wind(): R{wind_thresh} at edge of domain. {imax} shape: {distance.shape}")
        wind_radii['rads'][wind_thresh] = []
        for az in quads[windcode]:
            daz = 90*deg
            iquad = (az <= bearing) & (bearing < az+daz)
            if wind_radii_method == "azimuthal_mean":
                # Compute azimuthal mean
                wind_speed_vs_radius_km = get_azimuthal_mean(
                    wind_speed[iquad], distance[iquad], binsize=25.*units.km)
                if any(wind_speed_vs_radius_km >= wind_thresh):
                    max_dist_of_wind_threshold_nm = np.max(
                        radius_km[wind_speed_vs_radius_km >= wind_thresh])
                    wind_radii['rads'][wind_thresh].append(
                        max_dist_of_wind_threshold_nm.data)
                else:
                    wind_radii['rads'][wind_thresh].append(0.*units.km)
            else:
                # percentile method or max method
                iquad = iquad & (wind_speed >= wind_thresh)
                if iquad.sum():
                    x_km = distance.where(iquad)
                    if wind_radii_method.endswith("percentile"):
                        # assume wind_radii_method is a number followed by the string "percentile".
                        distance_percentile = float(
                            wind_radii_method.replace("percentile", ""))
                        # index of array entry nearest to percentile value
                        idist_of_wind_threshold = abs(
                            x_km-np.percentile(x_km, distance_percentile, interpolation='nearest')).argmin()
                        wind_radii['rads'][wind_thresh].append(
                            np.percentile(x_km, distance_percentile))
                    else:
                        assert wind_radii_method == "max"
                        idist_of_wind_threshold = x_km.argmax(x_km.dims)
                        wind_radii['rads'][wind_thresh].append(x_km.isel(idist_of_wind_threshold).data)
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
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    xlim = row.lon + np.array([-2.25, 2.25]) / np.cos(np.radians(row.lat))
    ylim = row.lat + [-2.25, 2.25]
    xi = np.linspace(*xlim, npts)
    yi = np.linspace(*ylim, npts)
    if where is not None:
        x = lonCell.where(where, drop=True).to_numpy()
        y = latCell.where(where, drop=True).metpy.dequantify()
    else:
        x, y = lonCell, latCell.metpy.dequantify()
    x[x >= 180] -= 360
    for ax, z in zip(axes.flatten(), [*zs]):
        if hasattr(z, "name"):
            label = z.name
        if hasattr(z, "long_name"):
            label = z.long_name
        z = z.where(where, drop=True).metpy.dequantify()
        zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')
        # ax.contour(xi, yi, zi)
        cntr1 = ax.contourf(xi, yi, zi, cmap="RdBu_r")
        ax.set_title(label)
        fig.colorbar(cntr1, ax=ax)
        # ax.plot(x,y, 'ko')
        ax.set(xlim=xlim, ylim=ylim)

    return fig

def derived_winds(u10, v10, mslp, lonCell, latCell,
        row: pd.Series,
        vmax_search_radius=250.*units.km,
        mslp_search_radius=100.*units.km,
        wind_radii_method="max",
        pouter_search_radius=800*units.km,
        debug=False) -> dict:
    # Given a row (with row.lon and row.lat)...
    # Derive cell distances and bearings
    distance, bearing = dist_bearing(
        row.lon*degE, row.lat*degN, lonCell, latCell)

    # Derive 10m wind speed and tangential wind speed Vt from u10 and v10
    speed = metpy.calc.wind_speed(u10, v10)

    # Tangential (cyclonic) wind speed
    # v * sin(bearing) - u * cos(bearing)
    Vt = (v10 * np.sin(np.radians(bearing)) -
          u10 * np.cos(np.radians(bearing)))
    if row.lat < 0:
        Vt = -Vt

    # Restrict Vmax search to vmax_search_radius
    vmaxrad = distance < vmax_search_radius
    ispeed_max = speed.where(vmaxrad).argmax(dim=speed.dims)
    raw_vmax = speed.isel(ispeed_max).data

    # If vmax > 34, check if tangential component of max wind is negative (anti-cyclonic)
    if row.vmax > 34 and Vt.isel(ispeed_max) < 0:
        print(" TC center", row.valid_time, row.lat, row.lon)
        print(f"  max wind is anti-cyclonic! {Vt.isel(ispeed_max).data:~3.2f}")
        print(
            f"  max wind lat/lon           {latCell.isel(ispeed_max).data:~3.1f} {lonCell.isel(ispeed_max).data:~4.1f}")
        print(
            f"  max wind U/V               {u10.isel(ispeed_max).data:~3.2f} {v10.isel(ispeed_max).data:~3.2f}")
        if debug:
            fig = debugplot(row, lonCell, latCell, u10,
                            v10, speed, mslp, where=vmaxrad)
            ofile = f"negVt.{lon2s(row.lon)}.{lat2s(row.lat)}.{row.valid_time.strftime('%Y%m%d%H')}.png"
            plt.savefig(ofile)
            print(f"created {os.path.realpath(ofile)}")

    # Check if average tangential wind within search radius is negative (anti-cyclonic)
    average_tangential_wind = Vt.where(vmaxrad).mean()
    if average_tangential_wind < 0:
        print(" TC center", row.valid_time, row.lat, row.lon,
              f" avg wind within vmax search radius is anti-cyclonic! {average_tangential_wind.data:~.2f}")
        if debug:
            fig = debugplot(row, lonCell, latCell, u10,
                            v10, speed, mslp, where=vmaxrad)
            ofile = f"negaverageVt.{lon2s(row.lon)}.{lat2s(row.lat)}.{row.valid_time.strftime('%Y%m%d%H')}.png"
            plt.savefig(ofile)
            logging.info(f"created {os.path.realpath(ofile)}")

    # Get radius of max wind
    raw_rmw = distance.isel(ispeed_max).data
    logging.debug(
        f" TC center {row.valid_time} {row.lat} {row.lon},  max wind {raw_vmax:~3.1f} @  {latCell.isel(ispeed_max).data:~3.1f}  {lonCell.isel(ispeed_max).data:~4.1f}  {raw_rmw:~3.0f}")

    # Restrict min mslp search
    mslprad = distance < mslp_search_radius
    iraw_minp = mslp.where(mslprad).argmin(dim=mslp.dims)# .min() drops units, turning Quantity into numpy array
    raw_minp = mslp.isel(iraw_minp).data

    # Get max extent of wind at thresh_kts thresholds.
    wind_radii = get_ext_of_wind(speed, distance, bearing, raw_vmax,
                                 latCell=latCell, lonCell=lonCell, wind_radii_method=wind_radii_method)

    # Restrict pressure of last closed isobar and radius of last closed isobar search to a certain radius.
    pouter_rad = distance < pouter_search_radius
    imslp_max = mslp.where(pouter_rad).argmax(dim=mslp.dims)
    raw_pouter = mslp.isel(imslp_max).data
    raw_router = distance.isel(imslp_max).data
    # Option 2
    # Zehr and Knaff (2007) https://journals.ametsoc.org/waf/article/22/1/71/38805/Reexamination-of-Tropical-Cyclone-Wind-Pressure
    # define pouter_rad as the annular average sea level pressure 800-1000 km.
    # I guess raw_router_nm (ROCI) is fixed at 900 km in this case.

    Vt_azimuthal_mean = get_azimuthal_mean(Vt, distance, binsize=200*units.km)
    Vt_500km = Vt_azimuthal_mean.sel(radius=500*units.km).data

    if Vt_500km < 0:
        logging.info(
            " atcf.derived_winds(): mean tangential wind 400-600km out is negative! TODO: figure out how to define storm_size_S.")
    # Should I use the input row.vmax or the raw_vmax from the raw model?
    storm_size_S = Vt_500km / V500c(row.vmax*kt, row.lat*degN)
    # I originally put raw_vmax, but I changed to row.vmax. It depends which one you think is more accurate.
    # For NARR at least, I think the input row.vmax is more accurate because NARR is coarse and biased low.
    penv_azimuthal_mean = get_azimuthal_mean(
        mslp, distance, binsize=200*units.km)
    penv = penv_azimuthal_mean.sel(radius=900*units.km).data

    derived_winds_dict = { # these values were xarrays, but I applied .data() to them already.
            "raw_vmax"     : raw_vmax,
            "raw_rmw"      : raw_rmw,
            "raw_minp"     : raw_minp,
            "wind_radii"   : wind_radii, 
            "raw_pouter"   : raw_pouter,
            "raw_router"   : raw_router,
            "Vt_500km"     : Vt_500km, # used for Zehr and Knaff Pmin derivation if storm_size_S not already calculated.
            "storm_size_S" : storm_size_S.to_base_units(), # used for Zehr and Knaff Pmin derivation.
            "penv"         : penv # used for Zehr and Knaff Pmin derivation.
            }
    return derived_winds_dict


def add_wind_rad_lines(row, wind_radii):
    lines = []
    for thresh in wind_radii['rads']:
        # row with 34, 50, or 64 knot radii
        newrow = row.copy()
        # 'rad' must be string. It is written as a string in write() method.
        # It is a category not a float.
        newrow['rad'] = str(thresh.to("knot").m)
        newrow['windcode'] = wind_radii['windcode']
        newrow[['rad1', 'rad2', 'rad3', 'rad4']] = [
            np.round(x.to("nautical_mile").m) for x in wind_radii['rads'][thresh]]
        lines.append(newrow)
    return pd.DataFrame(lines)


def origgridWRF(df, griddir, wind_radii_method="max"):
    """
    Get vmax, minp, radius of max wind, max radii of wind thresholds from WRF by Alex Kowaleski
    """

    wregex = r'WF((\d\d)|(CO))'
    WRFmember = df.model.str.extract(wregex, flags=re.IGNORECASE)
    # column 0 will have match or null
    if pd.isnull(WRFmember.iloc[:, 0]).any():
        logging.warning(
            f'assume WRF ensemble member, but not all model strings match {wregex}')
        pdb.set_trace()
    ens = WRFmember.iloc[0, 0]
    df = df.groupby('fhr').apply(WRFraw_vitals, griddir,
                                 ens, wind_radii_method=wind_radii_method)
    df = df.droplevel('fhr')
    return df


def WRFraw_vitals(fhr, griddir, ens, wind_radii_method='max'):
    if 'origmesh' in time.userdata1:
        logging.info(
            f"wait. fhr {fhr.name} already has original mesh values. Skipping.")
        return fhr
    row = fhr.head(1).squeeze()
    gridfile = os.path.join(griddir, "/EPS_"+str(ens)+"/E"+str(ens)+"_"+row.initial_time.strftime('%m%d%H') +
                            "_"+grid+"_" + row.valid_time.strftime('%Y-%m-%d_%H:%M:%S') + "_ll.nc")
    logging.debug(f'opening {gridfile}')
    ds = xarray.open_dataset(gridfile)
    ds = ds.isel(time=0)
    u10 = ds['u10']
    v10 = ds['v10']
    mslp = ds['slp']
    lonCell, latCell = np.meshgrid(ds.lon_0, ds.lat_0)
    # units and coordinates needed for derived_winds()
    lonCell = xarray.DataArray(
        lonCell*units(ds.lon_0.units), coords=u10.coords)
    # or else derived_winds() chokes on distance.isel()
    latCell = xarray.DataArray(
        latCell*units(ds.lat_0.units), coords=u10.coords)

    logging.debug(
        f"Extract vmax, RMW, minp, and radii of wind thresholds from row {row.name}")
    derived_winds_dict = derived_winds(
        u10, v10, mslp, lonCell, latCell, row, wind_radii_method=wind_radii_method)
    return row


def origgrid(df, griddir, ensemble_prefix="ens_", wind_radii_method="max"):
    """
    # Get vmax, minp, radius of max wind, max radii of wind thresholds from ECMWF grid, not from tracker.
    # Assumes
    #   ECMWF data came from TIGGE and were converted from GRIB to netCDF with ncl_convert2nc.
    #   4-character model string in ATCF file is "EExx" (where xx is the 2-digit ensemble member).
    #   ECMWF ensemble member in directory named "<ensemble_prefix>xx" (where xx is the 2-digit ensemble member).
    #   File path is "<ensemble_prefix>xx/${gs}yyyymmddhh.xx.nc", where ${gs} is the grid spacing (0p125, 0p15, 0p25, or 0p5).
    # ensemble_prefix may be a single string or a list of strings
    """

    # assert this is a single track
    assert df.groupby(['basin', 'cy', 'initial_time', 'model']
                      ).ngroups == 1, 'got more than 1 track'
    initial_time, model = df.iloc[0][[
        'initial_time', 'model']]

    if isinstance(ensemble_prefix, str):
        ensemble_prefixes = [ensemble_prefix]
    elif isinstance(ensemble_prefix, (list, tuple)):
        ensemble_prefixes = ensemble_prefix

    m = re.search(r'EE(\d\d)', model)
    if not m:
        logging.debug(
            'assume ECMWF ensemble member, but did not find EE\\d\\d in model string')
        logging.debug(f'no original grid for {model} - skipping')
        return df
    ens = int(m.group(1))  # strip leading zero

    # used to skip EE00 because I didn't know how to handle control run. Now it is handled.
    # if ens < 1:
    #    continue

    # Allow some naming conventions
    # ens_n/yyyymmddhh.n.nc
    # ens_n/0p15yyyymmddhh_sfc.nc
    # ens_n/0p25yyyymmddhh_sfc.nc
    # ens_n/0p5yyyymmddhh_sfc.nc
    yyyymmddhh = initial_time.strftime('%Y%m%d%H')
    yyyymmdd_hhmm = initial_time.strftime('%Y%m%d_%H%M')
    potential_gridfiles = []
    for ensemble_prefix in ensemble_prefixes:
        # If first filename doesn't exist, try the next one, and so on...
        # List in order of most preferred to least preferred.
        potential_gridfiles.extend([
            ensemble_prefix+str(ens)+"/SFC_"+yyyymmdd_hhmm +
            ".nc",  # Linus-style
            ensemble_prefix+str(ens)+"/" + "0p125" + \
            yyyymmddhh+"."+str(ens)+".nc",
            ensemble_prefix+str(ens)+"/" + "0p15" + \
            yyyymmddhh+"."+str(ens)+".nc",
            ensemble_prefix+str(ens)+"/" + "0p25" + \
            yyyymmddhh+"."+str(ens)+".nc",
            ensemble_prefix+str(ens)+"/" + "0p5"+yyyymmddhh+"."+str(ens)+".nc",
            ensemble_prefix+str(ens)+"/" + yyyymmddhh+"."+str(ens)+".nc"
        ])
    for gridfile in potential_gridfiles:
        # gridfile now has full path. Important for saving in atcf file. Or else os.path.realpath might use current directory.
        gridfile = os.path.join(griddir, gridfile)
        if os.path.isfile(gridfile):
            break
        logging.debug(f"no {gridfile}")

    df["originalmeshfile"] = gridfile
    logging.info(f'opening {gridfile}')
    ds = xarray.open_dataset(gridfile).metpy.quantify()
    df = df.groupby('fhr', group_keys=False).apply(
        ECMWFraw_vitals, ds, wind_radii_method=wind_radii_method)
    return df


def ECMWFraw_vitals(row, ds, wind_radii_method=None):
    row = row.head(1).squeeze() # make multiple wind rad lines one series.
    forecast_time0 = pd.to_timedelta(row.fhr, unit='h')
    if forecast_time0 not in ds.forecast_time0:
        print(
            f"atcf.ECMWFraw_vitals(): fhr {row.fhr} not in original mesh. Dropping time.")
        return None  # Don't return squeezed DataFrame (which is a Series now)
    ds = ds.sel(forecast_time0=forecast_time0)
    u10 = ds["10u_P1_L103_GLL0"]
    v10 = ds["10v_P1_L103_GLL0"]
    mslp = ds["msl_P1_L101_GLL0"]
    lonCell, latCell = np.meshgrid(ds.lon_0, ds.lat_0)
    # units and coordinates needed for derived_winds()
    lonCell = xarray.DataArray(
        lonCell*units(ds.lon_0.units), coords=u10.coords)
    # or else derived_winds() chokes on distance.isel()
    latCell = xarray.DataArray(
        latCell*units(ds.lat_0.units), coords=u10.coords)

    # Extract vmax, RMW, minp, and radii of wind thresholds
    derived_winds_dict = derived_winds(
        u10, v10, mslp, lonCell, latCell, row, wind_radii_method=wind_radii_method)

    row = unitless_row(derived_winds_dict, row)

    row = add_wind_rad_lines(row, derived_winds_dict["wind_radii"])

    return row

# unitless_row() used in ECMWFraw_vitals() and mpas.raw_vitals()


def unitless_row(derived_winds_dict, row):
    for c in ["vmax", "minp", "rmw", "pouter", "router"]:
        row[c] = derived_winds_dict["raw_"+c].to(column_units[c]).m
    row["wind_radii_method"] = derived_winds_dict["wind_radii"]["wind_radii_method"]
    # Used in Knaff and Zehr wind pressure relationship.
    for c in ["penv", "Vt_500km"]:
        row[c] = derived_winds_dict[c].to(column_units[c]).m
    return row


def ll_arc_distance(lat0=0*degN, lon0=0*degE, heading=0*deg, distance=0.*units.km, Rearth=earth_avg_radius):

    # Don't want to deal with Pandas indices. it will try to match them up, and you get NaNs and a bigger Series if they don't.
    lon0 = np.radians(lon0)
    lat0 = np.radians(lat0)
    heading_rad = np.radians(heading)
    d_rad = distance/Rearth

    newlat = np.arcsin(np.sin(lat0)*np.cos(d_rad) +
                       np.cos(lat0)*np.sin(d_rad)*np.cos(heading_rad))

    newlon = lon0 + np.arctan2(np.sin(heading_rad)*np.sin(d_rad)*np.cos(lat0),
                               np.cos(d_rad)-np.sin(lat0)*np.sin(newlat))

    return np.degrees(newlon), np.degrees(newlat)


def cross_track(track, cross):
    track = track.set_index("valid_time").sort_index()
    # TODO: keep repeated valid_times (multiple wind radii)
    track = track[~track.index.duplicated(keep='first')]
    ptrack = track.copy()

    for valid_time, row in track.iterrows():
        future_times = track.index > valid_time
        if future_times.sum() == 0:
            break
        # Find the first future time.
        inext = future_times.argmax()  # first True
        next_time = track.index[inext]
        # Get lon/lat and elapsed time at next location.
        lon1 = track.lon[inext] * degE
        lat1 = track.lat[inext] * degN
        # days since start of track
        dt = next_time - track.index.min()
        dt = dt.total_seconds()/24/3600 * units("day")
        cross_track_error = cross * dt
        perpendicular_heading = ptrack.heading[inext] * \
            units("deg") + 90*units("deg")

        # head off perpendicular to track, starting from end of control segment
        newlon, newlat = ll_arc_distance(
            lon0=lon1, lat0=lat1, distance=cross_track_error, heading=perpendicular_heading)
        # Assign one value to (possibly) multiple rows (for different wind radii).
        ptrack.loc[next_time, "lon"] = newlon.m
        ptrack.loc[next_time, "lat"] = newlat.m

    return ptrack.reset_index()


def veer_track(track, veer):
    track = track.set_index("valid_time").sort_index()
    # TODO: keep repeated valid_times (multiple wind radii)
    track = track[~track.index.duplicated(keep='first')]
    ptrack = track.copy()

    for valid_time, row in track.iterrows():
        lon0 = row.lon * degE
        lat0 = row.lat * degN
        future_times = track.index > valid_time
        if future_times.sum() == 0:
            logging.debug(f"no times later than {valid_time}. stop veering.")
            break
        # Find the first future time.
        inext = future_times.argmax()  # first True
        # Find the first future time.
        next_time = track.index[inext]
        # get distance and heading of original segment from this time to next time
        lon1 = track.lon[inext] * degE
        lat1 = track.lat[inext] * degN
        distance, heading = spc.gdist_bearing(lon0, lat0, lon1, lat1)
        logging.info(f"lon0={lon0} lat0={lat0} lon1={lon1} lat1={lat1}")

        # days since start of track
        dt = next_time - track.index.min()
        dt = dt.total_seconds()/24/3600 * units("day")
        new_heading = heading + veer * dt
        logging.info(f"{heading} dt={dt} new_heading={new_heading}")

        # start of perturbed segment is end of previous perturbed segment
        lon0 = ptrack.loc[valid_time, "lon"] * degE
        lat0 = ptrack.loc[valid_time, "lat"] * degN
        # head off in new direction for distance of control segment
        newlon, newlat = ll_arc_distance(
            lon0=lon0, lat0=lat0, distance=distance, heading=new_heading)
        # Assign one value to (possibly) multiple rows (for different wind radii)
        logging.info(
            f"perturbed lon0/lat0={lon0}/{lat0} dist={distance} next_time={next_time} newlon={newlon} newlat={newlat}")
        # convert quantity to magnitude or it will be treated as radians
        ptrack.loc[next_time, "lon"] = newlon.m
        ptrack.loc[next_time, "lat"] = newlat.m

    return ptrack.reset_index()
