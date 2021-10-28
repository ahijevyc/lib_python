import atcf # for atcf.dist_bearing(). Maybe delete in favor of cartopy.geodesic.inverse()
import cartopy
import cartopy.geodesic
import datetime
import glob # used to locate stormEvents files 
import matplotlib.pyplot as plt
from   metpy.units import units # used to normalize polarplot range
import numpy as np
import os # for basename
import pandas as pd
from   pandas.api.types import CategoricalDtype
import pdb
import pytz # Time zone-aware datetimes
import requests # for stormEvents()
from   scipy import spatial
import scipy.ndimage as ndimage
from scipy.stats import circmean # To average longitudes
import sqlite3
import sys # for stderr output


def getTCTOR(rename=True):
    # Tropical Cyclone Tornado (TCTOR) database from Roger Edwards (SPC)
    ifile = "/glade/work/ahijevyc/share/SPC/TCTor95-18.xls - Full List.csv"
    REdf = pd.read_csv(ifile, delimiter=',', parse_dates=[["Year","UTC-Mo.","UTC-Date","UTC time"]], 
            index_col=0, skiprows=1, header=[0], nrows=1506) # skipfooter=23)
    # skipfooter=23 could be used, but 'c' engine doesn't handle it. 'c' engine is faster than 'python' engine.
    REdf = REdf.tz_localize(tz='UTC')

    if rename:
        REdf.index.set_names('datetime', inplace=True)
        # Roger's columns have nice, descriptive names, but
        # rename some of Roger's columns to match terse SPC database.
        RE2SPC = {
                "LAT-start (path)" : "slat",
                "LON-start (path)" : "slon",
                "LAT-end (path)"   : "elat",
                "LON-end (path)"   : "elon"
                }

        REdf = REdf.rename(columns=RE2SPC)


    # Fix probable typo. "ST" should be "TS"
    REdf.loc[REdf["TC Cat"] == "ST","TC Cat"] = "TS"
    # Convert to dtype="category" - could do this at read_csv step, but then "ST" exists as a category.
    cat_type = CategoricalDtype(categories=["N","TD","TS","H","MH"], ordered=True)
    REdf["TC Cat"] = REdf["TC Cat"].astype(cat_type)


    return REdf


def RyanSobash(start=None, end=None, event_type="torn", debug=False):
    sobash_db = "/glade/u/home/sobash/2013RT/REPORTS/reports_all.db"
    conn = sqlite3.connect(sobash_db)
    sqltable = "reports_" + event_type
    # Could apply a datetime range here (WHERE datetime BETWEEN yyyy/mm/dd hh:mm:ss and blah), but converting from UTC to CST is tricky.
    sqlcommand = "SELECT * FROM "+sqltable
    sql_df = pd.read_sql_query(sqlcommand, conn, parse_dates=['datetime'])
    conn.close()
    # Add 6 hours to datetime. This converts to UTC.
    sql_df["datetime"] = sql_df["datetime"] + pd.to_timedelta(6, unit='h')
    # make it aware of its UTC timezone.
    sql_df["datetime"] = sql_df["datetime"].dt.tz_localize(pytz.UTC)
    sql_df = sql_df[(sql_df.datetime >= start) & (sql_df.datetime < end)]
    sql_df["dbfile"] = sobash_db
    if debug:
        print("From Ryan's SQL database",sobash_db)
        print(sqlcommand)
        print(sql_df.to_string())

    return sql_df, sobash_db # Yes, sobash is in the dbfile column of sql_df, but sql_df may have zero rows, so return sobash_db separately.

def spc_lsr_filename(event_type, latestyear=2019, debug=False):
    assert event_type in ["wind","hail","torn"]
    # Get wind, hail, or tornado local storm reports (LSR). If LSR do not exist locally, download from SPC.
    # Cache in TMPDIR, which by default is /glade/scratch/$USER/temp/.

    # Input
    # event_type: "wind", "hail", or "torn"
    # latestyear: most recent (4-digit) year in SPC dataset. Part of SPC filename. Update when SPC releases new year.

    import urllib.request
    idir = os.getenv("TMPDIR", "/glade/scratch/"+os.getenv("USER")+"/temp")
    if debug:
        print("spc.spc_lsr_filename: idir=",idir)

    if event_type=='torn':
        filename = idir + f"/1950-{latestyear}_actual_tornadoes.csv"
    else:
        filename = idir + f"/1955-{latestyear}_"+event_type+".csv"

    if debug:
        print("spc.spc_lsr_filename: local filename=",filename)
    if not os.path.exists(filename):
        url = "https://www.spc.noaa.gov/wcm/data/" + os.path.basename(filename)
        if debug:
            print("spc.spc_lsr_filename: local file not found. downloading",url)
        urllib.request.urlretrieve(url, filename)

    nominal_last_time = datetime.datetime(latestyear+1,1,1,0,0,0,0,pytz.UTC) 
    return filename, nominal_last_time


def get_storm_reports( start = datetime.datetime(2016,6,10,tzinfo=pytz.UTC), end = datetime.datetime(2016,7, 1,tzinfo=pytz.UTC), 
        event_types = ["torn", "wind", "hail"], latestyear = 2019, debug = False):

    # Return a DataFrame with local storm reports (LSRs) downloaded from SPC.
    # Choices are tornado, hail, and/or wind.
    assert isinstance(event_types,list)

    # INPUT
    #   start - start of time window, includes start. timezone aware datetime
    #   end   - end of time window, includes end. timezone aware datetime
    #   event_types - list of event types
    #   latestyear - most recent year in SPC file. 4-digit year
    #   debug - If True, print debugging information.
    # OUTPUT
    #   rpts - DataFrame with time, location, and description of event (LSR)


    if debug:
        print("get_storm_reports: start:",start)
        print("get_storm_reports: end:",end)
        print("get_storm_reports: event types:",event_types)


    # Create one DataFrame with all requested LSR event types.
    all_rpts = pd.DataFrame()
    for event_type in event_types:
        # Locate data file and nominal last time in data file.
        rpts_file, nominal_last_time = spc_lsr_filename(event_type, latestyear=latestyear, debug=debug)

        # Make sure the requested last time isn't later than the nominal last time.
        if end > nominal_last_time:
            print(f"spc.get_storm_reports({event_type}): requested end time",end,"is later than nominal last time in ",rpts_file,nominal_last_time)
            print("Do you need to download a new",event_type,"database from SPC?")
            sys.exit(1)

        # csv format described in http://www.spc.noaa.gov/wcm/data/SPC_severe_database_description.pdf
        # SPC storm report files downloaded from http://www.spc.noaa.gov/wcm/#data to 
        # cheyenne:/glade/work/ahijevyc/share/ Mar 2019.
        # Multi-year zip files have headers; pre-2016 single-year csv files have no header. 

        dtype = {
                "om": np.int64,
                "yr": np.int32,
                "mo": np.int32,
                "dy": np.int32,
                "date": str,
                "tz": np.int32,
                "st": str,
                "stf": np.int64, # State FIPS number. some Puerto Rico codes are incorrect
                "stn": np.int64,  # State number - number of this tornado in this state in this year
                "mag": np.float64, # you might think there is "sz" for hail and "f" for torn, but all "mag"
                "inj": np.int64,
                "fat": np.int64,
                "loss": np.float64,
                "closs": np.float64,
                "slat": np.float64,
                "slon": np.float64,
                "elat": np.float64,
                "elon": np.float64,
                "len": np.float64,
                "wid": np.float64,
                "ns": np.int64,
                "sn": np.int64,
                "sg": np.int64,
                "f1": np.int64,
                "f2": np.int64,
                "f3": np.int64,
                "f4": np.int64,
                "mt": str,
                "fc": np.int64,
                }

        # Unify "date" and "time" to make a naive datetime object (datetime with no timezone information).
        # The unified column is called "date_time".
        rpts = pd.read_csv(rpts_file, parse_dates=[['date','time']], dtype=dtype, infer_datetime_format=True)
        if debug:
            print("input file:",rpts_file)
            print("read",len(rpts),"lines")

        rpts["event_type"] = event_type
        rpts["source"] = rpts_file # Used to take basename of this. But why dispose dirname information?

        # -9 = unknown tornado F-scale
        # Change -9 to NaN
        rpts["mag"].replace(to_replace=-9, value = np.nan, inplace=True)

        # "Significant" is defined as: tornadoes rated EF2 or greater, thunderstorm wind gusts of hurricane force (74 mph) or higher, or hail 2 inches or larger in diameter.
        rpts["significant"] = ((event_type == "torn") & (rpts.mag >= 2.)) | ((event_type == "wind") & (rpts.mag >= 65.)) | ((event_type == "hail") & (rpts.mag >= 2.)) 
        if event_type == "hail":
            largehail = rpts.mag >= 2.
            if any(largehail):
                rpts.loc[largehail,"event_type"] = "large hail"
        if event_type == "wind":
            highwind = rpts.mag >= 65.
            if any(highwind):
                rpts.loc[highwind, "event_type"] = "high wind"



        # Derive timezone-aware datetime "time" from "date_time" and "tz" columns. Convert to UTC.
        # According to www.spc.noaa.gov/wcm/data/SPC_severe_database_description.pdf,
        # all date_times except for tz ?=unknown and 9=GMT were converted to 3=CST. 
        # But there are several tz=0 and tz=6. tz=6 is MDT in the NECI Storm Events database.

        # MDT is equivalent to CST. Therefore, change tz=6 (MDT) to tz=3 (CST).
        # Wrote to SPC Apr 1 2019 about fixing these lines.
        MDT = rpts['tz'] == 6
        if debug:
            print(f"spc.get_storm_reports(): found",MDT.sum(),f"{event_type} in MDT")
            print(rpts.loc[MDT, ['om','date_time','tz','event_type', 'source']], file=sys.stderr)
            print("changing tz from 6 to 3 because CST=MDT")
        rpts.loc[MDT, "tz"] = 3

        # Convert GMT(UTC) date_times to CST
        GMT = rpts.tz == 9
        if debug:
            print(f"spc.get_storm_reports(): found",GMT.sum(),f"{event_type} in GMT")
        rpts.loc[GMT, "date_time"] = rpts.loc[GMT, "date_time"] - datetime.timedelta(hours=6)
        rpts.loc[GMT, "tz"] = 3

        # Don't know what to with date_times with unknown time zones. They're treated like CST when converted to UTC below. 
        unknown_tz = rpts['tz'] == 0
        if debug:
            print(f"spc.get_storm_reports(): found",unknown_tz.sum(),f"{event_type} in unknown time zone")
            print(rpts.loc[unknown_tz, ['om','date_time','tz','event_type', 'source']], file=sys.stderr)

        """
        Wrote to SPC Apr 1 2019 about fixing tz!=3 lines.
        Email20190401 PatrickMarsh took over database in 2017 and has no record or documentation as to what those timezones are. 
        Each year I append new information to the end of the old information, so timezones will continue to exist as is until I can learn what those time zones are. 
        take a look at the NCEI version of storm data. They may have information I do not.
        
        When timezone = 0, it is 'UNK' (unknown?) in the NCEI Storm Events database.
        When timezone = 6, it is 'MDT' (Mountain Daylight Time?) in the NCEI Storm Events database.
        
        LSR with tz=0 or tz=6, followed by the zone (if any) according to NCEI Storm Events database:
                om       yr_mo_dy_time  tz zone
        2507   245 1956-06-01 11:33:00   0 UNK
        2855    89 1957-04-02 23:45:00   0 UNK
        9569   158 1967-04-21 12:33:00   0 UNK
        13409  264 1972-05-13 18:08:00   0 UNK
        15792  804 1974-08-13 15:03:00   0 UNK
        20640  458 1980-06-04 16:30:00   0 UNK
        22101  271 1982-05-11 14:25:00   0 UNK
        24007  200 1984-04-26 19:32:00   0 CST
        8145   216 1965-05-05 14:45:00   6 MDT
        25899  501 1986-07-01 22:15:00   6 MDT
        26054  656 1986-09-04 18:55:00   6 MDT
        28793  419 1990-05-24 15:00:00   6 MDT
        28797  422 1990-05-24 16:00:00   6 MDT
        28810  433 1990-05-24 18:33:00   6 MDT
        29192  815 1990-06-27 20:00:00   6 MDT
        29232  855 1990-07-05 21:10:00   6 MDT
        29347  971 1990-08-15 18:30:00   6 MDT
        33262  151 1994-04-22 18:06:00   6 MDT
        33557  446 1994-05-31 15:00:00   6 MDT
        33572  461 1994-06-06 14:40:00   6 MDT
        33574  463 1994-06-06 15:00:00   6 MDT
        33585  474 1994-06-07 14:47:00   6 MDT
        33586  476 1994-06-07 15:57:00   6 MDT
        33587  477 1994-06-07 16:10:00   6 MDT
        33589  478 1994-06-07 16:35:00   6 MDT
        33592  482 1994-06-07 18:50:00   6 not in Storm Events database
        33783  672 1994-06-29 15:45:00   6 MDT
        33834  722 1994-07-06 18:17:00   6 MDT
        33855  744 1994-07-12 14:30:00   6 MDT
        33886  775 1994-07-18 15:30:00   6 MDT
        33887  776 1994-07-18 16:00:00   6 MDT
        33888  777 1994-07-18 16:25:00   6 MDT
        33889  778 1994-07-18 16:40:00   6 MDT
        33890  779 1994-07-18 16:55:00   6 not in Storm Events database
        33891  780 1994-07-18 16:55:00   6 not in Storm Events database
        33892  781 1994-07-18 17:00:00   6 MDT
        """            

        # Now all date_times, except for ?=unknown, were converted to 3=CST.
        # Convert to UTC by adding 6 hours.
        rpts["time"] = rpts['date_time'] + pd.to_timedelta(6, unit='h')
        # Make time-zone aware datetime object (UTC) "time" . 
        rpts["time"] = rpts["time"].dt.tz_localize(pytz.UTC)

        # Apply requested time window start <= time <= end
        # Used to say rpts.time < end. But if rpts.time == end it would not get included. 
        time_window = (rpts.time >= start) & (rpts.time <= end)
        rpts = rpts[time_window]
        if debug:
            print("found",len(rpts),event_type,"reports")


        RyanSobashSanityCheck = debug
        if RyanSobashSanityCheck:
            # Verify I get the same thing as Ryan Sobash's sqlite3 database
            sql_df, RyanSobash_file = RyanSobash(start=start, end=end, event_type=event_type)

            # See if they have the same number of rows
            if len(sql_df) != len(rpts):
                print(f'spc.get_storm_reports(): {event_type} {start}-{end}')
                print("SPC has", len(rpts),"reports, but Ryan's SQL database has",len(sql_df),".")
                epoch1 = os.path.getmtime(rpts_file)
                epoch2 = os.path.getmtime(RyanSobash_file)
                if epoch2 > epoch1:
                    print("Ryan's database modified more recently but it may have duplicate reports.")
                if debug:
                    print("Mod date of SPC reports file:    ", datetime.datetime.fromtimestamp(epoch1).strftime('%c'))
                    print("Mod date of Ryan's SQL database: ", datetime.datetime.fromtimestamp(epoch2).strftime('%c'))
            elif any(sql_df.datetime.reset_index(drop=True) != rpts.time.reset_index(drop=True)): # Times don't all match
                print(f'spc.get_storm_reports(): {event_type} {start}-{end}')
                print("spc.read(): my database and Ryan's have same # of reports in the requested time window but the times aren't equal")
                print("so I can't easily compare the lat and lons")
                print(rpts)
                print(sql_df)
                print("oh well")
            else:
                # See if they have the same times (must have same number of lines to test like this)
                if (sql_df["datetime"].values != rpts["time"].values).any():
                    print("spc.get_storm_reports(): My data times don't match Ryan's SQL database")
                # See if they have the same locations
                same_columns = ["slat", "slon", "elat", "elon"]
                if (sql_df[same_columns].values != rpts[same_columns].values).any():
                    print("SPC locations don't match Ryan's SQL database")
                    # TODO: just show offending rows, but all columns
                    mine = rpts[same_columns]
                    his  = sql_df[same_columns]
                    print("SPC minus Ryan")
                    print(mine-his)
                    max_abs_difference = np.abs(his.values-mine.values).max()
                    print("max abs difference", max_abs_difference)
                    if max_abs_difference < 0.000001:
                        print("who cares about such a small difference?")
                    elif max_abs_difference < 0.1:
                        pass
                    else:
                        pdb.set_trace()
                        sys.exit(1)

        all_rpts = all_rpts.append(rpts, ignore_index=True, sort=False) # Append this storm report type

    return all_rpts

def symbol_dict(scale=1):
    # Color, size, marker, and label of wind, hail, and tornado storm reports
    return {
            "wind":       {"c" : 'blue',  "s": 8*scale, "marker":"s", "label":"Wind"},
            "high wind":  {"c" : 'black', "s":12*scale, "marker":"s", "label":"Wind/HI"},
            "hail":       {"c" : 'green', "s":12*scale, "marker":"^", "label":"Hail"},
            "large hail": {"c" : 'black', "s":16*scale, "marker":"^", "label":"Hail/LG"},
            "torn":       {"c" : 'red',   "s":12*scale, "marker":"v", "label":"Torn"}
            }


def plotgridded(storm_reports, ax, gridlat2D=None, gridlon2D=None, scale=1, sigma=1, alpha=0.5, event_types=["wind", "high wind", "hail", "large hail", "torn"], 
        debug=False):


    # Default dx and dy of 81.2705km came from https://www.nco.ncep.noaa.gov/pmb/docs/on388/tableb.html#GRID211
    if storm_reports.empty:
        if debug:
            print("spc.plotgridded(): storm reports DataFrame is empty. data_gridded will be all zeros.")

    storm_rpts_gridded = {}

    for event_type in event_types:
        if debug:
            print("looking for "+event_type)
        xrpts = storm_reports[storm_reports.event_type == event_type]
        print("plot",len(xrpts),event_type,"reports")
        # Used to skip to next iteration if there were zero reports, but you want grid even if it is all zeros.
        obslons, obslats = xrpts.slon.values, xrpts.slat.values

        grid_points = list(zip(gridlon2D.ravel(),gridlat2D.ravel()))# tried using zip but in python 3 this returns a 0-dimensional zip object, not 2 elements
        tree = spatial.KDTree(grid_points) 

        data_gridded = np.zeros_like(gridlon2D).flatten()
        if len(obslons) > 0:
            dist, indices = tree.query(list(zip(obslons,obslats)))
            data_gridded[indices] = 1

        data_gridded = data_gridded.reshape(gridlon2D.shape)

        if sigma > 0:
            if debug:
                print("about to run Gaussian smoother")
                pdb.set_trace()
            # I'm uncomfortable doing this in map space but it is fast
            data_gridded = ndimage.gaussian_filter(data_gridded, sigma=sigma)

        storm_rpts_gridded[event_type] = data_gridded

        if False:
            # TODO: fix extent so it is plotted correctly
            color = symbol_dict()[event_type]["c"]
            # get the color of the symbol, tack on an "s" and use that as the color map name.
            if color=="black": # there is no "blacks" colortable
                color="grey"
            color = color.capitalize()
            # add gridded storm events array to dictionary

            img = ax.imshow(np.ma.masked_less(data_gridded,0.001), extent=(gridlon2D.min(),gridlon2D.max(),gridlat2D.min(),gridlat2D.max()),
                    transform=cartopy.crs.PlateCarree(), cmap=plt.get_cmap(color+"s"), label=event_type, alpha=0.85)

    return storm_rpts_gridded

def get_event_type_from_label(event_type_plot):
    event_type = event_type_plot.get_label()
    event_type = event_type.split(" ")
    event_type = "".join(event_type[:-1]) # leave off the (count) word at the end of the label
    return event_type


def centroid_polar(theta_deg, r, debug=False):
    # locate centroid of this event type
    east = r * np.sin(np.radians(theta_deg))
    north = r * np.cos(np.radians(theta_deg))
    if debug:
        print(f"east {east} north {north}")
    east = east.mean()
    north = north.mean()
    if debug:
        print(f"east {east} north {north}")
    az = np.degrees(np.arctan2(east,north))
    r = np.sqrt(east**2 + north**2)
    return az, r

def polarplot(originlon, originlat, storm_reports, ax, zero_azimuth=0, normalize_range_by_value=None, scale=1.5, alpha=0.5, debug=False):

    if storm_reports.empty:
        if debug:
            print("spc.polarplot(): storm reports DataFrame is empty. Returning None")
        return None


    # Color, size, marker, and label of wind, hail, and tornado storm reports
    kwdict = symbol_dict(scale=scale)
    storm_rpts_plots = []

    geo = cartopy.geodesic.Geodesic()

    for event_type in ["wind", "high wind", "hail", "large hail", "torn"]:
        kwdict[event_type]["edgecolors"]="black"
        kwdict[event_type]["linewidths"] = 0.2
        if debug:
            print("spc.polarplot(): looking for",event_type)
        xrpts = storm_reports[storm_reports.event_type == event_type]
        if debug:
            print("spc.polarplot(): found",len(xrpts),event_type,"reports")
        if len(xrpts) == 0:
            continue
        lons, lats = xrpts.slon.values, xrpts.slat.values
        r_km, heading = atcf.dist_bearing(originlon, originlat, lons, lats)
        n3 = geo.inverse((originlon, originlat), np.column_stack((lons, lats)))
        n3 = np.asarray(n3) # convert cartopy MemoryView to ndarray
        r_km_geo = n3[:,0]/1000.
        start_heading = n3[:,1]
        start_heading[start_heading < 0] += 360.
        if (np.abs(r_km - r_km_geo).max() > 5):
            print("spc.polarplot(): distances", r_km, r_km_geo)
            if (np.abs(r_km - r_km_geo).max() > 10):
                pdb.set_trace()
        if (np.abs(heading - start_heading).max() > 2):
            print("spc.polarplot(): headings", heading, start_heading)
            if (np.abs(heading - start_heading).max() > 5):
                pdb.set_trace()
        if normalize_range_by_value:
            r_km = r_km * units("km") / normalize_range_by_value
            r_km = r_km.m
        # Filter out points beyond the max range of axis
        maxr = ax.get_ylim()[1]
        if all(r_km >= maxr):
            if debug:
                print("spc.polarplot():",event_type,"reports all outside axis range.")
            continue
        inrange = r_km < maxr
        r_km = r_km[inrange]
        heading = heading[inrange]
        if debug:
            print("spc.polarplot(): found",inrange.sum(),event_type,"reports inside axis range")
        kwdict[event_type]["label"] += " (%d)" % inrange.sum()
        theta = (heading - zero_azimuth + 360 ) % 360
        ax.set_autoscale_on(False) # Don't rescale the axes with far-away reports (thought unneeded after filtering out pts beyond maxr, but symbol near maximum range autoscales axis to larger range.)
        storm_rpts_plot = ax.scatter(np.radians(theta), r_km, alpha = alpha, **kwdict[event_type])

        az, r = centroid_polar(theta, r_km)

        if debug:
            print(f"centroid of {event_type} x,y {east:7.2f}km,{north:7.2f}km  az,r {az:5.1f}deg,{r:6.2f}km")

        storm_rpts_plots.append(storm_rpts_plot)
    return storm_rpts_plots

def plot(storm_reports, ax, scale=1, drawrange=0, alpha=0.5, debug=False):

    if storm_reports.empty:
        # is this the right thing to return? what about empty list []? or rpts?
        if debug:
            print("spc.plot(): storm reports DataFrame is empty. Returning")
        return None

    # Color, size, marker, and label of wind, hail, and tornado storm reports
    kwdict = symbol_dict(scale=scale)
    storm_rpts_plots = []

    for event_type in ["wind", "high wind", "hail", "large hail", "torn"]:
        if debug:
            print("looking for "+event_type)
        xrpts = storm_reports[storm_reports.event_type == event_type]
        if debug:
            print("plot",len(xrpts),event_type,"reports")
        kwdict[event_type]["label"] += " (%d)" % len(xrpts)
        if len(xrpts) == 0:
            continue
        lons, lats = xrpts.slon.values, xrpts.slat.values
        storm_rpts_plot = ax.scatter(lons, lats, alpha = alpha, edgecolors="None", **kwdict[event_type],
                transform=cartopy.crs.PlateCarree()) # ValueError: Invalid transform: Spherical scatter is not supported with crs.Geodetic
        storm_rpts_plots.append(storm_rpts_plot)
        if drawrange > 0:
            if debug:
                print("about to draw tissot circles for "+event_type)
            # With lons and lats, specifying more than one dimension allows individual points to be drawn. 
            # Otherwise a grid of circles will be drawn.
            # It warns about using PlateCarree to approximate Geodetic. It still warps the circles
            # appropriately, so I think this is okay.
            within_range = ax.tissot(rad_km = drawrange.to("km").magnitude, lons=lons[np.newaxis], lats=lats[np.newaxis], 
                facecolor=kwdict[event_type]["c"], alpha=0.4, label=str(drawrange)+" range")
            # TODO: Legend does not support tissot cartopy.mpl.feature_artist.
            # A proxy artist may be used instead.
            # matplotlib.org/users/legend_guide.html#
            # creating-artists-specifically-for-adding-to-the-legend-aka-proxy-artists
            # storm_rpts_plots.append(within_range)

    return storm_rpts_plots



def to_MET(df, gribcode=None, debug=False):
    # INPUT:
    #   df       - storm_reports DataFrame from spc.get_storm_reports()
    #   gribcode - Use gribcode if specified.
    # OUTPUT:
    #   txt - MET point observation format, one observation per line.
    #         encode event type (wind, hail, torn, significant or not) in "Grib_Code"
    #         assign EF-scale, wind speed, or hail size to "Observation_Value"
    # 
    # Each observation line will consist of the following 11 columns of data:
    met_columns = ["Message_Type", "Station_ID", "Valid_Time", "Lat", "Lon", "Elevation", "Grib_Code", "Level", "Height", "QC_String", "Observation_Value"]
    
    df["Message_Type"] = "StormReport" # as in https://github.com/dtcenter/METplus/blob/main_v3.1/parm/use_cases/model_applications/convection_allowing_models/read_ascii_storm.py
    df["Station_ID"]  = df.st
    df["Valid_Time"] = df.time.dt.strftime('%Y%m%d_%H%M%S') #df.time should be aware of UTC time zone
    df["Lat"]  = np.mean([df.slat,df.elat], axis=0)
    df["Lon"]  = circmean([df.slon,df.elon], low=-180, high=180, axis=0)
    df["Elevation"]  = 0.
    df["Height"] = 0.
    if gribcode:
        df["Grib_Code"] = gribcode
    else:
        df['Grib_Code'] = -9999
        # grib_code for tornado, hail and wind (regular and significant)
        # inspired by $MET_BASE/table_files/grib2_ndfd.txt
        # case-insensitive string matches
        df.loc[df['event_type'].str.lower() == 'torn', "Grib_Code"] = 197 # probability of tornado
        df.loc[df['event_type'].str.contains('hail',case=False), "Grib_Code"] = 198 # prob of hail
        df.loc[df['event_type'].str.contains('wind',case=False), "Grib_Code"] = 199 # prob of damaging wind
        df.loc[df['event_type'].str.lower() == 'large hail', "Grib_Code"] = 201 # prob of extreme hail
        df.loc[df['event_type'].str.lower() == 'high wind', "Grib_Code"] = 202 # prob of extreme wind
    df["Level"] = 0.
    df["QC_String"] = "NA" # DTC put 1,2,or 3 (torn, hail, or wind) in QC_String, but I like Grib_Code.
    df["Observation_Value"] = df.mag
    # index=False don't write index number
    # Change NaN to "NA" MET considers "NA" missing
    # This may not matter, but by adding a string 'NA', it changes the format of the entire column. Floats change to integers (or maybe strings). 
    df.replace(to_replace=np.nan, value = 'NA', inplace=True)
    txt = df.to_string(columns=met_columns,index=False,header=False)#used to append "\n" (for neatness?) but you can't split output on "\n" without empty string at end. 
    return txt


def listFD(url, ext=''):
    # Return list of files in url directory with given extension.
    from bs4 import BeautifulSoup
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]


# NCDC storm events homepage https://www.ncdc.noaa.gov/stormevents/ftp.jsp
#  event types: Hail, Thunderstorm Wind, Snow, Ice, etc. 
#               More types in https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/Storm-Data-Bulk-csv-Format.pdf
# There are 3 types of information: 1) details, 2) fatalities, and 3) locations
# NCDC storm event "details" don't have a lat and lon. Just a city, a range, and an azimuth.
# Perhaps the lat and lon are in the "location" files (as opposed to "details").
# Recommend using spc.get_storm_reports() instead.

def stormEvents(year, info="details", version="1.0", debug = False):
    # INPUT
    #   year   - 4-digit year
    # OUTPUT
    #   events - DataFrame with storm events
   
    # File name convention: StormEvents_details-ftp_v1.0_dyyyy_cyyyymmdd.csv
    # where dyyyy is the data year and cyyyymmdd is the creation date.

    url = f"https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles"
    for filename in listFD(url, "csv.gz"):
        if f"StormEvents_{info}-ftp_v{version}_d{year}_c" in filename:
            events = pd.read_csv(filename)
            break
    events["url"] = filename
    return events

def events2met(df):
    # Point Observation Format
    # input ASCII MET point observation format contains one observation per line. 
    # Each input observation line should consist of the following 11 columns of data:
    met_columns = ["Message_Type", "Station_ID", "Valid_Time", "Lat", "Lon", "Elevation", "Grib_Code", "Level", "Height", "QC_String", "Observation_Value"]
    
    df["Message_Type"] = "StormReport"
    df["Station_ID"]  = df.WFO
    df["Valid_Time"] = df.BEGIN.dt.strftime('%Y%m%d_%H%M%S')
    df["Lat"]  = df.lat # I don't know how to get this. 
    df["Lon"]  = df.lon
    df["Elevation"]  = "NA"
    df["Grib_Code"] = "NA"
    df["Level"] = "NA"
    df["Height"] = "NA"
    df["QC_String"] = "NA"
    df["Observation_Value"] = df.TOR_F_SCALE
    print(df.to_string(columns=met_columns))

