from metpy.units import units
import pandas as pd
import subprocess


def steering_flow(args = dict(stormname="Joaquin", storm_heading=45*units("degrees"), storm_speed=3.2*units("m/s"), 
    lat0=24.3*units("degrees_north"), lon0=-74.3*units("degrees_east"), 
    force_new=False, fhr=000, rx=4.5*units("degrees"), pbot=850*units("hPa"), ptop=200*units("hPa"), 
    file_ncl="/glade/scratch/ahijevyc/NARR/merged_AWIP32.2015100300.3D", ensmember="JOAQUIN2015", nc_output=False), debug=False, **kwargs):

    # Ensure correct units for unit-unaware NCL script
    kwargs["storm_speed"] = kwargs["storm_speed"].to("m/s").m
    # return Pandas Series with results of steering_flow.ncl
    # For example, wind_shear_heading, steering_flow_speed, etc. 

    # Replace default keyword arguments with provided ones (in kwargs dictionary). 
    for k in kwargs:
        args[k] = kwargs[k]
    arglist = ["ncl", "-Q"] # -Q Turn off echo of NCL version and copyright info
    if debug:
        arglist.remove("-Q")

    # Put args into strings suitable for ncl command line.
    for k in args:
        v = args[k]
        if isinstance(v, str):
            v = '"'+v+'"'
        arglist.append(k+"="+str(v))



    nclscript = "/glade/work/ahijevyc/ncl/steering.ncl"
    arglist.append(nclscript)
    if debug:
        print("ncl.steering_flow():", arglist)
    ncls = subprocess.run(arglist, stdout=subprocess.PIPE, encoding='utf-8')
    if debug:
        print("ncl.steering_flow: exit code was %d" % ncls.returncode)
    if debug or len(ncls.stdout.split("\n")) > 10: # force print if there's more standard output than usual
        print("ncl.steering_flow() stdout ", ncls.stdout)
    # Parse standard output for csv output filename.
    for line in ncls.stdout.split("\n"):
        if 'csv_output_file:' in line:
            words = line.split()
            csv_file = words[-1]
            #Tried squeeze=True to return a Series, but still DataFrame
            df = pd.read_csv(csv_file)
            df = df.squeeze()
            # Once we've found and read csv file, stop parsing standard output.
            break
    return df 
