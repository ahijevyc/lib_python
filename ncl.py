import logging
from metpy.units import units
import pandas as pd
import pdb
import subprocess


def steering_flow(args = dict(stormname="Joaquin", storm_heading=45*units("degrees"), storm_speed=3.2*units("m/s"), 
    lat0=24.3*units("degrees_north"), lon0=-74.3*units("degrees_east"), 
    force_new=False, fhr=000, rx=4.5*units("degrees"), pbot=850*units("hPa"), ptop=200*units("hPa"), 
    file_ncl="/glade/scratch/ahijevyc/NARR/merged_AWIP32.2015100300.3D", ensmember="JOAQUIN2015", nc_output=False), **kwargs):

    assert not pd.isnull(kwargs["storm_speed"]), "ncl.steering_flow: missing storm_speed"
    # Ensure correct units for unit-unaware NCL script
    kwargs["storm_speed"] = kwargs["storm_speed"].to("m/s").m
    kwargs["lat0"] = kwargs["lat0"].to("degrees_N").m
    kwargs["lon0"] = kwargs["lon0"].to("degrees_E").m
    kwargs["pbot"] = kwargs["pbot"].to("hPa").m
    kwargs["ptop"] = kwargs["ptop"].to("hPa").m
    kwargs["storm_heading"] = kwargs["storm_heading"].to("degrees").m
    # return Pandas Series with results of steering_flow.ncl
    # For example, wind_shear_heading, steering_flow_speed, etc. 

    # Replace default keyword arguments with provided ones (in kwargs dictionary). 
    args.update(kwargs)
    arglist = ["ncl", "-Q"] # -Q Turn off echo of NCL version and copyright info

    # Put args into strings suitable for ncl command line.
    for k in args:
        v = args[k]
        if isinstance(v, str):
            v = f'"{v}"'
        arglist.append(f"{k}={v}")


    nclscript = "/glade/work/ahijevyc/ncl/steering.ncl"
    arglist.append(nclscript)
    logging.info(f"ncl.steering_flow(): args={args}")
    logging.debug(f"ncl.steering_flow(): {arglist}")
    # PIPE stderr to eliminate warnings from running ncl script. ncl script does not warn if you run from command line. And NCARG_ROOT looks ok.
    # warning:Unable to load System Resource File /glade/work/ahijevyc/20201220_daa_casper/lib/python3.7/site-packages/PyNIO/ncarg/sysresfile
    # warning:WorkstationClassInitialize:Unable to access rgb color database - named colors unsupported:[errno=2]
    # warning:["Palette.c":1850]:NhlLoadColormapFiles: Invalid colormap directory: /glade/work/ahijevyc/20201220_daa_casper/lib/python3.7/site-packages/PyNIO/ncarg/colormaps
    ncls = subprocess.run(arglist, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    logging.debug(f"ncl.steering_flow: exit code was {ncls.returncode}")
    logging.debug(f"ncl.steering_flow() stdout {ncls.stdout}")
    # Parse standard output for csv output filename.
    for line in ncls.stdout.split("\n"):
        if 'csv_output_file:' in line:
            words = line.split()
            csv_file = words[-1]
            # Once we've found and read csv file, stop parsing standard output.
            return pd.read_csv(csv_file)
