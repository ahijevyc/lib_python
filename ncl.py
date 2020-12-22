import subprocess
import pandas as pd


def steering_flow(args = dict(stormname="Joaquin", storm_heading=45, storm_speed=3.2, lat0=24.3, lon0=-74.3, 
    force_new=False, fhr=000, rx=4.5, pbot=850, ptop=200, 
    file_ncl="/glade/scratch/ahijevyc/NARR/merged_AWIP32.2015100300.3D", ensmember="JOAQUIN2015", nc_output=False), debug=False, **kwargs):
    # return DataFrame with results of steering_flow.ncl
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
            df = pd.read_csv(csv_file)
            # Once we've found and read csv file, stop parsing standard output.
            break
    return df 
