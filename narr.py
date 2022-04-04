import datetime
from fieldinfo import fieldinfo, readNCLcm
import glob
import itertools
import logging
import matplotlib.colors as colors
from metpy.units import units 
import metpy.calc as mcalc
from metpy.interpolate import log_interpolate_1d
import numpy as np
import os 
import pandas as pd # used to label values along 'uv' dimension of vector dataset
import pdb
import pytz
import re
import subprocess
import sys
import tarfile
import xarray

# (vartype, file_suffix)
narrSfc = ('sfc', 'RS.sfc')
narrFlx = ('flx', 'RS.flx')
narrPBL = ('pbl', 'RS.pbl')
narr3D  = ('3D', '3D')
narrFixed  = "/glade/scratch/"+os.getenv("USER")+"/NARR/rr-fixed.grb.nc"

# Modify fieldinfo dictionary for NARR.
# fieldinfo keys are "human-readable" string nicknames like "u700" and "speed10m".
# Their values are also dictionaries with information about how to find the data, read the data, extract vertical level(s), and plot the data. 
#   For example the following key : value pairs
#   key      : value
#   cmap     : color map
#   fname    : field variable name
#   levels   : contour levels
#   vertical : vertical level(s) to extract. These are pint quantities with units. For layers: [bot, top]

# Remove 'filename' from all fieldinfo values. .pop will not return a KeyError if 'filename' is not present.
[fieldinfo[f].pop('filename',None) for f in fieldinfo]

# Define pressure-level variables
levs = [l * units.meters for l in [10,30]]
levs.extend(np.arange(100,1025,25) * units.hPa) # large range useful for Chris Rozoff's CM1 model. Use wide range to prevent "out of contour range" error in NARR_composite.py.
levs.extend([l * units["dimensionless"] for l in ['lev1', 'trop']])
for lev in levs:
    for ws in ['wind', 'speed', 'u', 'v', 'vort', 'div']:
        f = ws+str(lev.m)
        if lev == 10 * units.meters:
            f = ws+"10m" # field names of variables at 10 meters have 'm' suffix. Pressure-level variables have nothing (no "hPa" or "mb")
        if f not in fieldinfo: fieldinfo[f] = {}
        fieldinfo[f]['levels'] = range(2,36,2)
        fieldinfo[f]['cmap'] = readNCLcm('wind_17lev')
        if ws == 'vort':
            fieldinfo[f]['levels'] = np.arange(-4,40,4) 
        if ws == 'div':
            fieldinfo[f]['levels'] = np.arange(-27,33,6) 
            fieldinfo[f]['cmap'] = readNCLcm('BlueWhiteOrangeRed')
        if lev.units == units.hPa:
            fieldinfo[f]['fname'] = ['U_GRD_221_ISBL','V_GRD_221_ISBL'] # changed to get vector data
        elif lev.units == units.meters:
            fieldinfo[f]['fname'] = ['U_GRD_221_HTGL', 'V_GRD_221_HTGL']
        else:
            fieldinfo[f]['fname'] = [None, None]
        fieldinfo[f]['vertical'] = lev
        #fieldinfo[f]['units'] = 'knots' # should remain m/s for publications
        if ws == 'u' or ws == 'v': # wind component can be negative or positive
            fieldinfo[f]['levels'] = range(-22,26,4)
            fieldinfo[f]['cmap'] = readNCLcm('cmocean_balance')
        if ws == 'u':
            fieldinfo[f]['sel'] =  [fieldinfo[f]['fname'][0]] # make 1-element list so you can use info['sel'][0] for attributes
        if ws == 'v':
            fieldinfo[f]['sel'] =  [fieldinfo[f]['fname'][1]] # make 1-element list so you can use info['sel'][0] for attributes

    hgt = 'hgt'+str(lev.m)
    if hgt not in fieldinfo: fieldinfo[hgt] = {}
    fieldinfo[hgt]['levels'] = range(0,17000,500)
    fieldinfo[hgt]['cmap'] =  readNCLcm('nice_gfdl')[3:193]
    fieldinfo[hgt]['fname'] = 'HGT_221_ISBL'
    fieldinfo[hgt]['vertical'] = lev
    sh = 'sh'+str(lev.m)
    if sh not in fieldinfo: fieldinfo[sh] = {}
    fieldinfo[sh]['levels'] = [1e-11,0.01,0.1,1,5,10,15,20,25]
    fieldinfo[sh]['cmap'] =  readNCLcm('nice_gfdl')[3:193]
    fieldinfo[sh]['fname'] = 'SPF_H_221_ISBL'
    fieldinfo[sh]['vertical'] = lev
    fieldinfo[sh]['units'] = 'g/kg'
    temp = 'temp'+str(lev.m)
    if temp not in fieldinfo: fieldinfo[temp] = {}
    fieldinfo[temp]['levels'] = range(-65,30,5)
    fieldinfo[temp]['cmap'] =  readNCLcm('nice_gfdl')[3:193]
    fieldinfo[temp]['fname'] = 'TMP_221_ISBL'
    fieldinfo[temp]['vertical'] = lev
    fieldinfo[temp]['units'] = 'degC'
    rh = 'rh'+str(lev.m)
    if rh not in fieldinfo: fieldinfo[rh] = {}
    fieldinfo[rh]['fname'] = ['TMP_221_ISBL','SPF_H_221_ISBL']
    fieldinfo[rh]['vertical'] = lev
    fieldinfo[rh]['units'] = 'percent'
    vvel = 'vvel'+str(lev.m)
    fieldinfo[vvel] = {'levels' : [-250,-100,-25,-10,-2.5,-1,1,2.5,10,25,100,250], 'cmap': readNCLcm('cmocean_balance')[::-1], 'fname': 'V_VEL_221_ISBL'}
    fieldinfo[vvel]['cmap'][127] = "white"
    fieldinfo[vvel]['vertical'] = lev
    fieldinfo[vvel]['units'] = 'microbar/second'
fieldinfo['bunkers']['fname'] = ['USTM_221_HTGY','VSTM_221_HTGY']
fieldinfo['CAPE_221_SFC'] = fieldinfo["sbcape"].copy()
fieldinfo['CAPE_221_SFC'].update(dict(fname = 'CAPE_221_SFC', vertical = 'surface-based'))
fieldinfo['CAPE_221_SPDY'] = fieldinfo["sbcape"].copy()
fieldinfo['CAPE_221_SPDY'].update(dict(fname = 'CAPE_221_SPDY', vertical = '180-0mb above gnd'))
#fieldinfo['hfx'] = {'levels' : [-640,-320,-160,-80,-40,-20,-10,0,5,10,15,20,40,60,80], 'cmap':readNCLcm('amwg256')[::-1], 'fname'  : ['SHTFL_221_SFC'] }
fieldinfo['hfx'] = {'levels' : list(range(-600,125,25)), 'cmap':readNCLcm('amwg256')[::-1], 'fname'  : 'SHTFL_221_SFC'} # NARR sfc flux is upward (highly negative in day)
fieldinfo['lcl'] = {}
fieldinfo['lcl']['cmap'] = [readNCLcm('nice_gfdl')[i] for i in [3,20,37,54,72,89,106,123,141,158,175,193]]
fieldinfo['lcl']['cmap'].reverse()
fieldinfo['lcl']['fname'] = ['PRES_221_HTGL','TMP_221_HTGL','DPT_221_HTGL']
fieldinfo['lcl']['levels'] = [400, 500, 600, 700, 750, 800, 850, 875, 900, 925, 950, 975, 1000]
fieldinfo['lcl']['units'] = 'hPa'
fieldinfo['lcl']['parcel'] = 2*units.meter
#fieldinfo['lh']  = {'levels' : [-1280,-640,-320,-160,-80,-40,-20,0,10,20,40], 'cmap':readNCLcm('MPL_BrBG')[127:40:-1], 'fname'  : ['LHTFL_221_SFC'] }
fieldinfo['lh']  = {'levels' : list(range(-700,150,50)), 'cmap':readNCLcm('MPL_BrBG')[127:30:-1], 'fname'  : 'LHTFL_221_SFC'}
fieldinfo['mslp']['fname'] = 'PRMSL_221_MSL'
fieldinfo['mslp']['levels'] = np.arange(956,1028,4)
fieldinfo['mslp']['units'] = 'hPa'
fieldinfo['mslet'] = fieldinfo['mslp'].copy()
fieldinfo['mslet'] = 'MSLET_221_MSL'
fieldinfo['sbcape']['fname'] = 'CAPE_221_SFC'
fieldinfo['sbcape']['vertical'] = 'surface-based'
fieldinfo['sbcinh']['cmap'].reverse()
fieldinfo['sbcinh']['fname'] = 'CIN_221_SFC'
fieldinfo['sbcinh']['levels'].reverse()
fieldinfo['sbcinh']['levels'] = [-np.round(x/2) for x in fieldinfo['sbcinh']['levels']] # NARR cin is negative, and halve levels
fieldinfo['sbcinh']['vertical'] = 'surface-based'
fieldinfo['mlcape']['fname'] = 'CAPE_221_SPDY' # 180-0 mb above ground (according to grib1 in /glade/collections/rda/data/ds608.0/3HRLY and https://rda.ucar.edu/datasets/ds608.0/#!docs)
fieldinfo['mlcape']['vertical'] = 'mixed-layer'
fieldinfo['mlcinh'] = fieldinfo['sbcinh'].copy()
fieldinfo['mlcinh']['fname'] = 'CIN_221_SPDY'
fieldinfo['mlcinh']['vertical'] = 'mixed-layer'
fieldinfo['mucape']['vertical'] = 'most unstable'
fieldinfo['pblh']['fname'] = 'HPBL_221_SFC'
fieldinfo['precipacc']['fname'] = 'RAINNC'
#fieldinfo['pwat']['levels'] = [5,10,20,25,30,40,50,60,70,80,90,100] # precipitable water in kg/m**2 not depth-of-water
fieldinfo['pwat']['levels'] = [20,25,30,35,40,45,50,55,60,65,70] # precipitable water in kg/m**2 not depth-of-water
fieldinfo['pwat']['fname'] = 'P_WAT_221_EATM'
fieldinfo['pwat']['temporal'] = 0
fieldinfo['rh_0deg'] = fieldinfo['rh700'].copy() 
fieldinfo['rh_0deg']['fname'] = 'R_H_221_0DEG'
fieldinfo['rh_0deg']['vertical'] = 'freezing level' # Remember to overwrite 'vertical' from rh700
fieldinfo['rh2'] = fieldinfo['rh700'].copy() 
fieldinfo['rh2']['fname'] = 'R_H_221_HTGL'
fieldinfo['rh2']['vertical'] = 2*units.meters
fieldinfo['rhlev1'] = fieldinfo['rh700'].copy() 
fieldinfo['rhlev1']['fname'] = 'R_H_221_HYBL'
fieldinfo['rhlev1']['vertical'] = 'lowest model level'
fieldinfo['scp'] = fieldinfo['stp'] # stp is in fieldinfo.py not scp 
fieldinfo['scp']['fname'] = ['CAPE_221_SFC','CIN_221_SFC','HLCY_221_HTGY']
fieldinfo['scp']['shear_layer'] = 'shr10_500'
fieldinfo['sh2']    = {'levels' : [0.5,1,2,4,8,12,14,16,17,18,19,20,21,22,23,24], 'cmap':fieldinfo['td2']['cmap'], 'fname': 'SPF_H_221_HTGL', 'vertical':2*units.meters, 'units':'g/kg'}
fieldinfo['shlev1'] = {'levels' : [0.5,1,2,4,8,12,14,16,17,18,19,20,21,22,23,24], 'cmap':fieldinfo['td2']['cmap'], 'fname': 'SPF_H_221_HYBL', 'vertical':'lowest model level', 'units':'g/kg'}
fieldinfo['shr10_30']  = fieldinfo['speed10m'].copy()
fieldinfo['shr10_30']['levels'] = range(0,54,3)
#shrlev1_trop = wind shear between tropopause and lowest model level
for bot, top in itertools.permutations(levs, 2): # create shear fieldinfo entry for every permutation of levels
    shr = f"shr{bot.m}_{top.m}"
    fieldinfo[shr] = fieldinfo['shr10_30'].copy()
    fieldinfo[shr]['vertical'] = [bot,top]
fieldinfo['shrtrop'] = {'levels':np.array([2,5,10,15,20,30,50])*1e-3, 'cmap': fieldinfo['speed700']['cmap'], 
                        'fname': 'VWSH_221_TRO', 'vertical':'tropopause'} # shear at tropopause. https://www.emc.ncep.noaa.gov/mmb/rreanl/faq.html 
fieldinfo['srh'] = fieldinfo['srh1'].copy()
fieldinfo['srh']['levels'].extend([750])
fieldinfo['srh']['cmap'].extend(readNCLcm('wind_17lev')[-6:-4])
fieldinfo['srh']['fname'] = 'HLCY_221_HTGY'
fieldinfo['stp'] = fieldinfo['scp']
fieldinfo['surface_height'] = fieldinfo['sbcape'].copy()
fieldinfo['surface_height']['fname'] = 'HGT_221_SFC'
fieldinfo['t2']['fname'] = 'TMP_221_SFC'
fieldinfo['t2']['units'] = 'degF'
fieldinfo['tctp'] = fieldinfo['scp']
fieldinfo['tctp']['shear_layer'] = 'shr10_700'
fieldinfo['thetasfc'] = {'levels' : np.arange(290,320,2), 'cmap': ['#eeeeee', '#dddddd', '#cccccc', '#aaaaaa']+readNCLcm('precip2_17lev')[3:-1], 'fname'  : 'POT_221_SFC'}
fieldinfo['theta2']   = {'levels' : np.arange(294,313,1), 'cmap': ['#eeeeee', '#dddddd', '#cccccc', '#aaaaaa']+readNCLcm('precip2_17lev')[3:-1], 'fname'  : ['PRES_221_HTGL', 'TMP_221_HTGL'], 'vertical':2*units.meters}
fieldinfo['thetae2']  = {'levels' : np.arange(321,375,3), 'cmap': ['#eeeeee', '#dddddd', '#cccccc', '#aaaaaa']+readNCLcm('precip2_17lev')[3:-1], 'fname'  : ['PRES_221_HTGL', 'TMP_221_HTGL','DPT_221_HTGL'], 'vertical':2*units.meters}
fieldinfo['vvellev1'] = fieldinfo['vvel700'].copy()
fieldinfo['vvellev1']['fname'] = 'V_VEL_221_HYBL'
fieldinfo['wvflux'] = dict(fname=['WVUFLX_221_ISBY_acc3h','WVVFLX_221_ISBY_acc3h'],cmap=[],levels=[])
fieldinfo['wvflux']['sel'] = fieldinfo['wvflux']['fname']
fieldinfo['wvfluxconv'] = {'fname':'WVCONV_221_ISBY_acc3h','levels':np.array(fieldinfo['precip']['levels'])*10,'cmap': readNCLcm('prcp_1')}
fieldinfo['wcflux'] = dict(fname=['WCUFLX_221_ISBY_acc3h','WCVFLX_221_ISBY_acc3h'],cmap=[],levels=[])
fieldinfo['wcflux']['sel'] = fieldinfo['wcflux']['fname']
fieldinfo['wcfluxconv'] = {'fname':'WCCONV_221_ISBY_acc3h','levels':np.array(fieldinfo['precip']['levels'])*2,'cmap': readNCLcm('prcp_1')}
fieldinfo['zlcl']['fname'] = ['PRES_221_HTGL','TMP_221_HTGL','DPT_221_HTGL',"HGT_221_ISBL"] 
fieldinfo['zlcl']['parcel'] = 2 * units.meter


#######################################################################
idir = "/glade/collections/rda/data/ds608.0/3HRLY/" # path to NARR
#######################################################################

# Get static NARR file, convert to netCDF.
# Return netCDF filename.
def get_static(fieldname, debug=False):
    narr = narrFixed
    targetdir = os.path.basedir(narr)
    # Convert to netCDF if netCDF file doesn't exist.
    if not os.path.exists(narr):
        narr_grb ='/glade/collections/rda/data/ds608.0/FIXED/rr-fixed.grb'
        call_args = ["ncl_convert2nc", narr_grb, "-e", "grb", "-o", targetdir]
        print(call_args)
        subprocess.check_call(call_args)

    ncStatic = xarray.open_dataset(narr)
    return ncStatic[fieldname]


# Get NARR file from tar file, convert to netCDF.
# Return netCDF filename.
def get(valid_time, targetdir='.', narrtype=narr3D, idir=idir, debug=False):
    if narrtype == narrFixed:
        return narrFixed
    assert isinstance(valid_time, datetime.datetime) # make sure valid_time is a datetime object
    if debug:
        print("narr.get(): valid_time=",valid_time)
    vartype, file_suffix = narrtype # 3D, clm, flx, pbl, or sfc 
    narr = targetdir + '/' + valid_time.strftime("merged_AWIP32.%Y%m%d%H."+file_suffix+".nc")
    # Convert to netCDF if netCDF file doesn't exist.
    if os.path.exists(narr):
        #TODO: Make sure file is complete. Another instance of narr.get() may still be writing it.
        pass
    else:
        print("narr.get(): previously existing narr file not found",narr)
        narr_grb = narr[:-3] # drop '.nc' suffix
        # Extract NARR grib file from tar file if it doesn't exist.
        if not os.path.exists(narr_grb):
            search_str = idir + valid_time.strftime('%Y/NARR'+vartype+'_%Y%m_') + '*.tar'
            if debug:
                print("narr.get(): search_str=",search_str)
            narrtars = glob.glob(search_str)
            for n in narrtars:
                prefix, yyyymm, dddd = n.split('_')
                dd1 = datetime.datetime.strptime(yyyymm + dddd[0:2], '%Y%m%d')
                # NARR3D_201704_0103.tar starts on 1st at 0Z and ends just before 4th at 0Z. so add one to eday
                dd2 = datetime.datetime.strptime(yyyymm + dddd[2:4], '%Y%m%d') + datetime.timedelta(days=1)
                # make dd1 and dd2 timezone-aware if valid_time is timezone-aware
                if valid_time.tzinfo is not None and valid_time.tzinfo.utcoffset(valid_time) is not None:
                    if debug:
                        print("making dd1 and dd2 timezone-aware cause valid_time is")
                    dd1 = pytz.utc.localize(dd1)
                    dd2 = pytz.utc.localize(dd2)
                if valid_time >= dd1 and valid_time < dd2:
                    break
            narrtar = n
            print("Found NARR tar file: "+narrtar)
            tar = tarfile.open(narrtar, mode='r')
            print("extracting "+os.path.basename(narr_grb))
            ret = tar.extract(os.path.basename(narr_grb),path=targetdir)
        call_args = ["ncl_convert2nc", narr_grb, "-e", "grb", "-o", targetdir]
        print(call_args)
        subprocess.check_call(call_args)
    return narr 

def get_ll(data):
    # TODO: maybe delete. but kept for interesting notes below
    lon, lat = data.gridlon_221.metpy.quantify(), data.gridlat_221.metpy.quantify()
    # lon, lat = data.metpy.longitude, data.metpy.latitude # Thought about using metpy accessor, but I was warned "x" and "y" coordinates must be 1-D not 2-D in met-1.0 and beyond.
    # subtract 360 from positive longitudes west of dateline or else you get straight spagehetti across the plot
    # This fix may only work for NARR grid because all the problem points are in the northwest part of the grid
    # and the northeast has no positive longitudes. Considered warning metpy developers about this, but I think it's
    # a problem with the original grib or the way ncl_convert2nc converts it. 
    lon = lon.where(lon<0,lon-360*units.deg) # use xarray method instead of numpy to keep as xarray. (lat is still xarray)

    return lon, lat

def myunits(data, info, debug=False):
    # Quantify xarray.
    data = data.metpy.quantify() # quantify in MetPy>=1
    # Convert to info["units"].
    if "units" in info:
        if debug:
            print("converting to "+info["units"])
        data = data.metpy.convert_units(info["units"]) # Don't .sel(lv_HTGL3=10) before .metpy.convert_units('kt'). You get None in return.
    return data



def temporal(data, info, debug=False):

    logging.debug(f"narr.temporal(): info={info}")

    # Define 'timetitle' attribute of data, no matter what.
    if 'timetitle' not in data.attrs:
        data.attrs['timetitle'] = ''

    if 'temporal' in info:
        time0 = info['temporal']
        if hasattr(data.metpy, 'time'): 
            temporal, = data.metpy.coordinates('time')
            data = data.metpy.sel(time=time0)
        elif len(data.shape) <= 2: # If data has only 2 dimensions assume it has no temporal dimension (like tropopause-level, or max-wind-level)
            data.attrs['timetitle'] = info['temporal']
            logging.debug(f"narr.temporal(): assume no temporal dimension. setting timetitle={info['temporal']}")
            return data
        else:
            temporal_str = data.dims[0]
            assert 'time' in temporal_str
            if debug:
                print("narr.temporal(): metpy does not identify temporal coordinate. assume 1st dim is ("+temporal_str+")")
            data = data[time0]
        data.attrs['timetitle'] = str(time0) + "-h" # TODO: only works with zero. convert [ns] to hours. 

    return data



def vertical(data, info, debug=False):
    logging.debug(f"narr.vertical(): data {data.name} info {info}")

    if 'verttitle' in data.attrs:
        return data

    # Define 'verttitle' attribute of data, no matter what.
    data.attrs['verttitle'] = ''

    # If data has a vertical coordinate dimension, select the appropriate level.

    if 'vertical' in info:
        vlevel = info['vertical']
        if hasattr(data.metpy, 'vertical'): 
            vertical, = data.metpy.coordinates('vertical')
            if vertical.name in data.dims:
                # If vertical has already been run on this DataArray, the vertical dimension may be length 1.
                # This zero-dimensional dimension was dropped and can't be selected again.
                # Tried DataArray.expand_dims() but the degenerate vertical dimension confused the ax.contour command.
                # contour expects a 2D array not 3D.
                data = data.metpy.sel(vertical=vlevel)
            else:
                logging.debug(f"narr.vertical(): {data.name} {vertical.name} not in {data.dims}. assuming vertical has already had its way")
            verttitle = str(vlevel)
        elif len(data.dims) <= 2: # If data has only 2 dimensions assume it has no vertical dimension (like tropopause-level, or max-wind-level)
            logging.debug(f'narr.vertical(): {data.name} is 2D already.')
            logging.info(f'narr.vertical(): setting verttitle={vlevel}')
            data.attrs['verttitle'] = vlevel
            return data
        else:
            # using list comprehension  # to get element with substring  
            res = [i for i in data.dims if 'lv_' == i[0:3]]
            vertical = res[0]
            logging.debug(f"narr.vertical(): metpy does not identify vertical coordinate. assume it is ({vertical})")
            data = data.sel({vertical:vlevel})
            vertical = data.coords[vertical]
            verttitle = str(vlevel) + vertical.units
        logging.debug(f"narr.vertical(): setting verttitle {verttitle}")
        data.attrs['verttitle'] = verttitle

    return data



def shear(field, valid_time=None, targetdir=None, debug=False):
    # bottom and top vertical level are in fieldinfo[field][vertical]
    bot, top = fieldinfo[field]['vertical']

    logging.debug(f"narr.shear(): bot={bot} top={top}")

    # winds are found in the flx or 3D file. Open both.
    ifiles = [get(valid_time, targetdir=targetdir, narrtype=narrtype) for narrtype in [narrFlx,narr3D]]
    ds = xarray.open_mfdataset(ifiles)

    # ubot and vbot
    if bot.units == units.meters:
        ubot = ds['U_GRD_221_HTGL'].sel(lv_HTGL3=bot)
        vbot = ds['V_GRD_221_HTGL'].sel(lv_HTGL3=bot)
    elif bot.units == units.hPa:
        ubot = ds['U_GRD_221_ISBL'].sel(lv_ISBL0=bot)
        vbot = ds['V_GRD_221_ISBL'].sel(lv_ISBL0=bot)
    elif bot == 'lev1': # lowest model level
        ubot = ds['U_GRD_221_HYBL']
        vbot = ds['V_GRD_221_HYBL']
        ubot.attrs['verttitle'] = 'lowest model level'
        vbot.attrs['verttitle'] = 'lowest model level'
    else:
        print("narr.shear(): unexpected bot {bot}")
        sys.exit(1)

    # utop and vtop
    if top.units == units.meters:
        utop = ds['U_GRD_221_HTGL'].sel(lv_HTGL3=top)
        vtop = ds['V_GRD_221_HTGL'].sel(lv_HTGL3=top)
    elif top.units == units.hPa:
        utop = ds['U_GRD_221_ISBL'].sel(lv_ISBL0=top)
        vtop = ds['V_GRD_221_ISBL'].sel(lv_ISBL0=top)
    elif top == 'trop': #  tropopause
        utop = ds['U_GRD_221_TRO']
        vtop = ds['V_GRD_221_TRO']
        utop.attrs['verttitle'] = 'tropopause'
        vtop.attrs['verttitle'] = 'tropopause'
    else:
        print("narr.shear(): unexpected top {top}")
        sys.exit(1)

    du =  utop-ubot
    dv =  vtop-vbot
    du.attrs = utop.attrs
    dv.attrs = vtop.attrs
    if du.name is None: # xarray.DataArray.name used in vectordata()
        du.name = utop.name
    if dv.name is None:
        dv.name = vtop.name
    du.attrs['long_name'] += ' shear'
    dv.attrs['long_name'] += ' shear'
    du.attrs['verttitle'] = f"{bot} to {top}"
    dv.attrs['verttitle'] = f"{bot} to {top}"
    return du, dv


def multiInterp(x, xp, fp):
    # x = target vertical coordinate (2D) 
    # xp = vertical coordinates of data to interpolate (1D)
    # fp = data to interpolate (3D)
    xp = np.broadcast_to(xp[:,None,None],fp.shape) # broadcast 1D xp array across 2 new spatial dimensions of fp
    assert xp.shape == fp.shape, 'narr.multiInterp(): shapes of xp and fp differ'
    # xp>x is False below the vertical layer that encompasses target
    # Once xp>x turns True, the np.diff function keeps the first occurrence of True in the vertical.
    bb = np.diff(xp>x,axis=0,append=True) # 3d boolean array. True at start of vertical layer that encompasses target
    k = bb.argmax(axis=0) # vertical index of start of vertical layer that encompasses target
    ij = np.indices(x.shape)
    kij = (k, ij[0], ij[1]) # 3d indices of start of vertical layer 
    rateofchange = np.diff(fp, axis=0, append=np.nan) /  np.diff(xp, axis=0, append=np.nan) # rate of change of f with respect to x
    rateofchange = rateofchange[kij] # just for the layer that encompasses target
    fstart = fp[kij] # f at start of layer 
    dx = x - xp[kij] # difference betweeen target and start of layer
    data = fstart + dx * rateofchange
    return data

def pressure_to_height(target_p, hgt3D, debug=False):

    # target_p and vertical coordinate of hgt3D are both in hPa before taking natural log.
    lv_ISBL0 = hgt3D.lv_ISBL0.metpy.unit_array.to('hPa').m
    log_lv_ISBL0 = np.log(lv_ISBL0)
    log_target_p = np.log(target_p.metpy.unit_array.to('hPa').m)

    data = multiInterp(log_target_p, log_lv_ISBL0, hgt3D.values)

    # numpy array to xarray
    hgt2D = xarray.zeros_like(hgt3D.metpy.dequantify().mean(dim='lv_ISBL0',keep_attrs=True)) # dequantify moves units to attributes
    hgt2D.values = data
    # with quantified data
    return hgt2D.metpy.quantify()

def scalardata(field, valid_time, targetdir=".", debug=False):
    # Get color map, levels, and netCDF variable name appropriate for requested variable (from fieldinfo module).
    info = fieldinfo[field]

    # Make cmap a colors.ListedColormap, if it is not already.
    if not isinstance(info['cmap'], (colors.ListedColormap)):
        info['cmap'] = colors.ListedColormap(info['cmap']) 
    logging.debug(f"scalardata: found {field} info={info}")

    # Get narr file and filename.
    ifiles = [get(valid_time, targetdir=targetdir, narrtype=narrtype) for narrtype in [narrSfc, narrFlx, narrPBL, narr3D]]
    # TODO: fix hack
    ifile_basename = ifiles[0].replace(narrSfc[1]+".nc","")

    logging.debug(f"About to open {ifiles}")

    nc = xarray.open_mfdataset(ifiles)

    # .load() to avoid UserWarning: Passing an object to dask.array.from_array which is already a Dask collection. This can lead to unexpected behavior.
    data = nc[info["fname"]].load().metpy.quantify()
    # Define data array. Speed and shear derived differently.
    # Define 'long_name' attribute
    
    if field.startswith("speed"):
        u = data[info["fname"][0]]
        v = data[info["fname"][1]]
        data = mcalc.wind_speed(u,v)
        data.name = field
        data.attrs['long_name'] = "wind speed"
    elif field.startswith("div"):
        u = data[info["fname"][0]]
        v = data[info["fname"][1]]
        data = mcalc.divergence(u,v) * 1e5
        data.name = field
        data.attrs['long_name'] = "divergence * 1e5"
    elif field.startswith("vort"):
        u = data[info["fname"][0]]
        v = data[info["fname"][1]]
        data = mcalc.vorticity(u,v) * 1e5
        data.name = field
        data.attrs['long_name'] = "vorticity * 1e5"
    elif field[0:3] == 'shr' and '_' in field:
        du, dv = shear(field, valid_time=valid_time, targetdir=targetdir, debug=debug)
        data = mcalc.wind_speed(du, dv)
        data.name = field
        data.attrs.update({'long_name':'wind shear', 'verttitle' :du.attrs["verttitle"]})
    elif field[0:2] == 'rh' and 'lv_ISBL0' in data.coords: # could be 2-m RH or rh_0deg
        pres = data['lv_ISBL0']
        temp = data[info["fname"][0]]
        sh   = data[info["fname"][1]]
        data = mcalc.relative_humidity_from_specific_humidity(pres, temp, sh)
        data.name = field 
        data.attrs['long_name'] = "relative humidity"
    elif field == 'theta2':
        prs = data[info["fname"][0]]
        tmp = data[info["fname"][1]]
        data = mcalc.potential_temperature(prs, tmp) # Tried being clever and using *data, but complains about no units
        data = xarray.DataArray(data=data, name=field)
        data.attrs['long_name'] = "potential temperature"
    elif field == 'thetae2':
        prs = data['PRES_221_HTGL']
        tmp = data['TMP_221_HTGL']
        dpt = data['DPT_221_HTGL']
        data = mcalc.equivalent_potential_temperature(prs, tmp, dpt)
        data = xarray.DataArray(data=data, name=field)
        data.attrs['long_name'] = "equivalent potential temperature"
    elif field == 'scp' or field == 'stp' or field == 'tctp':
        cape, cin, srh = data.data_vars.values()
        bulk_shear = scalardata(info['shear_layer'], valid_time, targetdir=targetdir, debug=debug)
        lifted_condensation_level_height = scalardata('zlcl', valid_time, targetdir=targetdir, debug=debug)
        if field == 'scp':
            # In SPC help, cin is positive in SCP formulation.
            cin_term = -40 * units.parse_expression("J/kg")/cin
            cin_term = cin_term.where(cin < -40*units.parse_expression("J/kg"), other=1)
            scp = mcalc.supercell_composite(cape, srh, bulk_shear) * cin_term.metpy.unit_array
            attrs = {'long_name': 'supercell composite parameter'}
            data = xarray.DataArray(data=scp, name=field, attrs=attrs) 
        if field == 'stp':
            cin_term = (200*units.parse_expression("J/kg") +cin)/(150*units.parse_expression("J/kg"))
            cin_term = cin_term.where(cin <= -50*units.parse_expression("J/kg"), other=1)
            cin_term = cin_term.where(cin >= -200*units.parse_expression("J/kg"), other=0)
            # CAPE, srh, bulk_shear, cin may be one vertical level, but LCL may be multiple heights.
            # xarray.broadcast() makes them all multiple heights with same shape, so significant_tornado doesn't 
            # complain about expecting lat/lon 2 dimensions and getting 3 dimensions..
            (cape, lifted_condensation_level_height, srh, bulk_shear, cin_term) = xarray.broadcast(cape, lifted_condensation_level_height, srh, bulk_shear, cin_term)
            # Caveat, NARR storm relative helicity (srh) is 0-3 km AGL, while STP expects 0-1 km AGL. 
            # Tried to ignore non-finite elements to avoid RuntimeWarning: invalid value encountered in greater/less but couldn't use 2-d boolean indexing with cape
            # cape and bulk_shear have different nans
            stp = mcalc.significant_tornado(cape, lifted_condensation_level_height, srh, bulk_shear) * cin_term.metpy.unit_array
            attrs = {'long_name': 'significant tornado parameter'} # , 'verttitle':lifted_condensation_level_height.attrs['verttitle']} # don't want "2 meter" verttitle
            data = xarray.DataArray(data=stp, name=field, attrs=attrs) 
        if field == 'tctp':
            tctp = srh/(40*units.parse_expression('m**2/s**2')) * bulk_shear/(12*units.parse_expression('m/s')) * (2000*units.meters - lifted_condensation_level_height)/(1400*units.meters)
            # NARR storm relative helicity (srh) is 0-3 km AGL, while original TCTP expects 0-1 km AGL. 
            # So the shear term is too large using the NARR srh. Normalize the srh term with a larger denominator. 
            # In STP, srh is normalized by 150 m**2/s**2. Use that.
            tctp_0_3kmsrh = srh/(150*units.parse_expression('m**2/s**2')) * bulk_shear/(12*units.parse_expression('m/s')) * (2000*units.meters - lifted_condensation_level_height)/(1400*units.meters)
            attrs = {'long_name': 'TC tornado parameter'}
            data = xarray.DataArray(data=tctp_0_3kmsrh, name=field, attrs=attrs)
    elif field=='lcl':
        parcel = info["parcel"]
        # .fillna() to eliminate RuntimeWarning: overflow encountered in exp and RuntimeWarning: invalid value encountered in true_divide
        # metpy.calc.lcl() takes and returns pint.Quantitys not xarrays.
        pres = data['PRES_221_HTGL'].metpy.dequantify().fillna(101315).metpy.sel(vertical=parcel)
        temp = data['TMP_221_HTGL'].metpy.dequantify().fillna(273).metpy.sel(vertical=parcel)
        dwpt = data['DPT_221_HTGL'].metpy.dequantify().fillna(273) # no vertical coordinate to select, for some reason
        LCL_pressure, LCL_temperature = mcalc.lcl(pres, temp, dwpt)
        # Transfer coords and dims from pres because LCL_pressure is only a pint.Quantity.
        data = xarray.zeros_like(data['DPT_221_HTGL']) # dpt has no vertical, for some reason
        data.name=field
        data.values = LCL_pressure
        data.attrs["long_name"] = f"pressure of lifted condensation level from metpy.calc.lcl using {parcel} parcel"
    elif field=='zlcl':
        LCL_pressure = scalardata('lcl', valid_time, targetdir=targetdir, debug=debug)
        hgt3D = data["HGT_221_ISBL"] 
        ifile = get(None, targetdir=targetdir, narrtype=narrFixed)
        nc = xarray.open_dataset(ifile)
        surface_height = nc[fieldinfo["surface_height"]["fname"]].metpy.quantify()
        nc.close()
        data = pressure_to_height(LCL_pressure, hgt3D)
        data = data - surface_height
        data.attrs['long_name']=LCL_pressure.attrs["long_name"].replace("pressure of", "height AGL of")
    else:
        if 'sel' in info: # this is a component of a vector
            attrs = data[info["sel"][0]].attrs # remember attributes of sel component before .to_array removes them
            data = data.to_array(dim="uv") # convert from Dataset with 2 DataArrays to single DataArrray
            data.attrs = attrs # for long_name 
    data = myunits(data, info, debug=debug)
    data = vertical(data, info, debug=debug)
    data = temporal(data, info, debug=debug)

    data.attrs['field'] = field
    data.attrs['ifile'] = ifile_basename
    # use np.array to allow for levels to be a range
    levels = np.array(info['levels'])
    data.attrs['levels'] = levels
    data.attrs.update(info)

    return data


def vectordata(field, valid_time, targetdir=".", debug=False):
    # Get color map, levels, and netCDF variable name appropriate for requested variable (from fieldinfo module).
    info = fieldinfo[field]
    logging.debug(f"vectordata(): field={field} info={info}")
    if debug:
        pdb.set_trace()
    if field[0:3] == "shr":
        u,v = shear(field, valid_time, targetdir=targetdir, debug=debug)
        u = temporal(u, info, debug=debug) # shear doesn't apply temporal like scalardata does.
        v = temporal(v, info, debug=debug)
        uv = xarray.merge([u,v]).to_array(dim="uv") # Tried concat, but didn't preserve the dataarray names or uv coordinate values (needed for uvsel).
        uv.attrs.update(info) # shear() doesn't copy over attributes like scalardata does
        uv.attrs.update(u.attrs)
    elif field.endswith("flux"):
        uv = scalardata(field, valid_time, targetdir=targetdir, debug=debug)
    else:
        uname = 'u'+str(info['vertical'].m)
        if uname == 'u10': uname = 'u10m' 
        uv = scalardata(uname, valid_time, targetdir=targetdir, debug=debug)

    # Fix sel attribute
    uv.attrs["sel"] = uv.uv.values # select all (both) dimensions
    # Fix long_name, which was copied from u-component DataArray when you requested scalardata(uname).
    uv.attrs["long_name"] = uv.attrs["long_name"].replace("u-component of ","").replace("zonal ","")
    uv.attrs['field'] = field # 'field' attribute should have been added to u and v separately in scalardata().
    return uv



def fromskewtds(nc, field, debug=False):
    # Used by NARR_lineplot.py
    # input:
    # nc: xarray Dataset with u, v, t, sh, hgt
    # field: field to derive
    # Returns:
    # derived DataArray
   
    temperature = nc["temp"].compute() # remove dask. problems with mixing dask and ndarrays, using len().
    pressure = nc.lv_ISBL0
    # for some reason temperature and sh get different shapes if I don't broadcast 1-D pressure first
    pressure = nc.lv_ISBL0.broadcast_like(temperature) # avoid ValueError: operands could not be broadcast together with shapes (6, 27, 18, 3) (27, 6, 18, 3)
    specific_hum = nc["sh"].compute()
    if field[0:2] == 'rh':
        hPa = field[2:]
        assert hPa.isnumeric()
        relative_humidity = mcalc.relative_humidity_from_specific_humidity(pressure, temperature, specific_hum)
        return relative_humidity.sel(lv_ISBL0 = int(hPa)*units.hPa) # pressure level units ignored but included for clarity
    # Don't derive fields here that can easily be created by NARR_composite.py
    # for example, speed, shr10_700, theta2, thetae2, etc.
    if debug:
        pdb.set_trace()
    # Also avoid ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has 2 dimension(s) and the array at index 1 has 1 dimension(s)
    if field == 'scp' or field == 'stp' or field == 'tctp':
        hgts = nc['hgt'].compute()
        us   = nc['u'].compute()
        vs   = nc['v'].compute()
        dewpoint = mcalc.dewpoint_from_specific_humidity(pressure, temperature, specific_hum)
        from time import perf_counter
        t1_start = perf_counter()
        mucape = xarray.DataArray(coords=[nc.hrs,nc.storm,nc.point])
        mucin  = xarray.DataArray(coords=[nc.hrs,nc.storm,nc.point])
        srh    = xarray.DataArray(coords=[nc.hrs,nc.storm,nc.point])
        for point in nc.point: # cape_cin is only a 1-D thing in MetPy. 
            for hrs in nc.hrs:
                # .loc doesn't work with storm coordinate
                mucapes= xarray.DataArray(coords=[nc.storm])
                mucins = xarray.DataArray(coords=[nc.storm])
                srhs   = xarray.DataArray(coords=[nc.storm])
                for istorm, storm in enumerate(nc.storm):
                    kwargs = dict(point=point, hrs=hrs)
                    # .sel doesn't work with storm coordinate because there are 2 hrs for each storm. storms are not unique
                    t  = temperature.sel(**kwargs).isel(storm=istorm)
                    td =   dewpoint.sel(**kwargs).isel(storm=istorm)
                    u  =         us.sel(**kwargs).isel(storm=istorm)
                    v  =         vs.sel(**kwargs).isel(storm=istorm)
                    h  =       hgts.sel(**kwargs).isel(storm=istorm)
                    cc = mcalc.most_unstable_cape_cin(nc.lv_ISBL0, t, td)
                    mucapes[istorm], mucins[istorm] = cc[0].m, cc[1].m # .m avoids AttributeError: Neither Quantity object nor its magnitude (0) has attribute...  
                    # srh is 1-D. If you supply higher dim vars, it tries to allocate 73.1 TiB for array (27, 18, 3, 27, 18, 3, 4723921) 
                    _,_,srhs[istorm] = mcalc.storm_relative_helicity(h, u, v, 3*units.km) 
                print(point.values, hrs.values, storm.values, cc, srhs[istorm].values)
                mucape.loc[kwargs], mucin.loc[kwargs] = mucapes*units("J/kg"), mucins* units("J/kg")
                srh.loc[kwargs] = srhs * units("m**2/s**2")

        t1_stop = perf_counter()
        print("Elapsed time:", t1_stop-t1_start, 's')
        pdb.set_trace()
        u6, v6 = u.sel(hgt=6*units('km')), v.sel(hgt=6*units('km'))
        u0, v0 = u.sel(hgt=0*units('km')), v.sel(hgt=0*units('km'))
        bulk_shear = mcalc.wind_speed(u6-u0, v6-v0)
        lifted_condensation_level_height = scalardata('zlcl', valid_time, targetdir=targetdir, debug=debug)
       
        if field == 'scp':
            # In SPC help, cin is positive in SCP formulation.
            cin_term = -40/cin
            cin_term = cin_term.where(cin < -40, other=1)
            scp = mcalc.supercell_composite(cape, srh, bulk_shear) * cin_term.metpy.unit_array
            attrs = {'long_name': 'supercell composite parameter'}
            data = xarray.DataArray(data=scp, name=field, attrs=attrs) 
        if field == 'stp':
            cin_term = (200+cin)/150
            cin_term = cin_term.where(cin <= -50, other=1)
            cin_term = cin_term.where(cin >= -200, other=0)
            # CAPE, srh, bulk_shear, cin may be one vertical level, but LCL may be multiple heights.
            # xarray.broadcast() makes them all multiple heights with same shape, so significant_tornado doesn't 
            # complain about expecting lat/lon 2 dimensions and getting 3 dimensions..
            (cape, lifted_condensation_level_height, srh, bulk_shear, cin_term) = xarray.broadcast(cape, lifted_condensation_level_height, srh, bulk_shear, cin_term)
            # Caveat, NARR storm relative helicity (srh) is 0-3 km AGL, while STP expects 0-1 km AGL. 
            # Tried to ignore non-finite elements to avoid RuntimeWarning: invalid value encountered in greater/less but couldn't use 2-d boolean indexing with cape
            # cape and bulk_shear have different nans
            stp = mcalc.significant_tornado(cape, lifted_condensation_level_height, srh, bulk_shear) * cin_term.metpy.unit_array
            attrs = {'long_name': 'significant tornado parameter'} # , 'verttitle':lifted_condensation_level_height.attrs['verttitle']} # don't want "2 meter" verttitle
            data = xarray.DataArray(data=stp, name=field, attrs=attrs) 
        if field == 'tctp':
            tctp = srh/(40*units['m**2/s**2']) * bulk_shear/(12*units['m/s']) * (2000*units.meters - lifted_condensation_level_height)/(1400*units.meters)
            # NARR storm relative helicity (srh) is 0-3 km AGL, while original TCTP expects 0-1 km AGL. 
            # So the shear term is too large using the NARR srh. Normalize the srh term with a larger denominator. 
            # In STP, srh is normalized by 150 m**2/s**2. Use that.
            tctp_0_3kmsrh = srh/(150*units['m**2/s**2']) * bulk_shear/(12*units['m/s']) * (2000*units.meters - lifted_condensation_level_height)/(1400*units.meters)
            attrs = {'long_name': 'TC tornado parameter'}
            data = xarray.DataArray(data=tctp_0_3kmsrh, name=field, attrs=attrs)
    elif field=='lcl':
        pres = nc['p']
        temp = temperature
        dwpt = nc[info['fname'][2]]
        LCL_pressure, LCL_temperature = mcalc.lcl(pres.fillna(pres.mean()), temp.fillna(temp.mean()), dwpt.fillna(dwpt.mean()))
        attrs = {"long_name":"pressure of lifted condensation level", "from":"metpy.calc.lcl"}
        # assign coords and dims manually because LCL_pressure is only an array of Pint quantities. 
        data = xarray.DataArray(data = LCL_pressure, coords=pres.coords, dims=pres.dims, name='LCL', attrs=attrs)
    elif field=='zlcl':
        LCL_pressure = scalardata('lcl', valid_time, targetdir=targetdir, debug=debug)
        ifile = get(valid_time, targetdir=targetdir, narrtype=narr3D)
        hgt3D = data["HGT_221_ISBL"] 
        data = pressure_to_height(LCL_pressure, hgt3D)
        ds = xarray.open_dataset(ifile)
        surface_height = ds[fieldinfo["surface_height"]["fname"]].metpy.quantify()
        ds.close()
        print('subtract surface height')
        data = data - surface_height
        data.attrs['long_name']=LCL_pressure.attrs["long_name"].replace("pressure of", "height AGL of")
    elif field=='srh1':
        print(f"Can't derive {field} yet")
    elif field=='srh3':
        print(f"Can't derive {field} yet")
    else:
        data = nc[fvar]

    return data

