import datetime
from fieldinfo import fieldinfo, readNCLcm
import glob
import matplotlib.colors as colors
from metpy.units import units as munits
from metpy.calc import wind_speed, potential_temperature, significant_tornado, pressure_to_height_std
import metpy.calc as mcalc
from metpy.interpolate import log_interpolate_1d
import numpy as np
import os # for NCARG_ROOT
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
#   filename : surface, flux, pbl, 3d, or fixed NARR file
#   fname    : field variable name
#   levels   : contour levels
#   vertical : vertical level(s) to extract. These are pint quantities with units. For layers: [bot, top]

# Make filename narrSfc by default.
for k in fieldinfo:
    fieldinfo[k]['filename'] = narrSfc


# Define pressure-level variables
levs = [10*munits.m]
levs.extend(np.arange(100,1025,25) * munits("hPa"))  # large range useful for Chris Rozoff's CM1 model. Use wide range to prevent "out of contour range" error in NARR_composite.py.
for lev in levs:
    for ws in ['wind', 'speed', 'u', 'v']:
        f = ws+str(lev.m)
        if lev == 10 * munits.m:
            f = ws+"10m" # field names of variables at 10 meters have 'm' suffix. Pressure-level variables have nothing (no "hPa" or "mb")
        if f not in fieldinfo: fieldinfo[f] = {}
        fieldinfo[f]['levels'] = range(2,36,2)
        fieldinfo[f]['cmap'] = readNCLcm('wind_17lev')
        fieldinfo[f]['filename'] = narr3D
        fieldinfo[f]['fname'] = ['U_GRD_221_ISBL','V_GRD_221_ISBL'] # changed to get vector data
        fieldinfo[f]['vertical'] = lev
        #fieldinfo[f]['units'] = 'knots' # should remain m/s for publications
        if lev == 10 * munits.m:
            fieldinfo[f]['fname'] = ['U_GRD_221_HTGL', 'V_GRD_221_HTGL']
            fieldinfo[f]['filename'] = narrFlx
        if ws == 'u' or ws == 'v': # wind component can be negative or positive
            fieldinfo[f]['levels'] = range(-22,26,4)
            fieldinfo[f]['cmap'] = readNCLcm('cmocean_balance')
        if ws == 'v':
            fieldinfo[f]['fname'].reverse() # put v-component first so scalardata method works
    sh = 'sh'+str(lev.m)
    if sh not in fieldinfo: fieldinfo[sh] = {}
    fieldinfo[sh]['levels'] = [1e-11,0.01,0.1,1,5,10,15,20,25]
    fieldinfo[sh]['cmap'] =  readNCLcm('nice_gfdl')[3:193]
    fieldinfo[sh]['filename'] = narr3D
    fieldinfo[sh]['fname'] = ['SPF_H_221_ISBL']
    fieldinfo[sh]['vertical'] = lev
    fieldinfo[sh]['units'] = 'g/kg'
    temp = 'temp'+str(lev.m)
    if temp not in fieldinfo: fieldinfo[temp] = {}
    fieldinfo[temp]['levels'] = range(-65,30,5)
    fieldinfo[temp]['cmap'] =  readNCLcm('nice_gfdl')[3:193]
    fieldinfo[temp]['filename'] = narr3D
    fieldinfo[temp]['fname'] = ['TMP_221_ISBL']
    fieldinfo[temp]['vertical'] = lev
    fieldinfo[temp]['units'] = 'degC'
    hgt = 'hgt'+str(lev.m)
    if hgt not in fieldinfo: fieldinfo[hgt] = {}
    fieldinfo[hgt]['levels'] = range(0,17000,500)
    fieldinfo[hgt]['cmap'] =  readNCLcm('nice_gfdl')[3:193]
    fieldinfo[hgt]['filename'] = narr3D
    fieldinfo[hgt]['fname'] = ['HGT_221_ISBL']
    fieldinfo[hgt]['vertical'] = lev
    vvel = 'vvel'+str(lev.m)
    fieldinfo[vvel] = {'levels' : [-250,-100,-25,-10,-2.5,-1,1,2.5,10,25,100,250], 'cmap': readNCLcm('cmocean_balance')[::-1], 'fname'  : ['V_VEL_221_ISBL'], 'filename': narr3D}
    fieldinfo[vvel]['cmap'][127] = "white"
    fieldinfo[vvel]['vertical'] = lev
    fieldinfo[vvel]['units'] = 'microbar/second'
    rh = 'rh'+str(lev.m)
    if rh not in fieldinfo: fieldinfo[rh] = {}
    fieldinfo[rh]['fname'] = ['TMP_221_ISBL'] # for some reason scalardata checks for fname[0] in nc
    fieldinfo[rh]['filename'] = narr3D
    fieldinfo[rh]['vertical'] = lev
    fieldinfo[rh]['units'] = 'percent'
fieldinfo['bunkers']['fname'] = ['USTM_221_HTGY','VSTM_221_HTGY']
fieldinfo['bunkers']['filename'] = narrFlx
fieldinfo['bunkers']['arrow'] = True
#fieldinfo['hfx'] = {'levels' : [-640,-320,-160,-80,-40,-20,-10,0,5,10,15,20,40,60,80], 'cmap':readNCLcm('amwg256')[::-1], 'fname'  : ['SHTFL_221_SFC'], 'filename':narrSfc}
fieldinfo['hfx'] = {'levels' : list(range(-600,125,25)), 'cmap':readNCLcm('amwg256')[::-1], 'fname'  : ['SHTFL_221_SFC'], 'filename':narrSfc} # NARR sfc flux is upward (highly negative in day)
fieldinfo['lcl'] = {}
fieldinfo['lcl']['cmap'] = [readNCLcm('nice_gfdl')[i] for i in [3,20,37,54,72,89,106,123,141,158,175,193]]
fieldinfo['lcl']['cmap'].reverse()
fieldinfo['lcl']['filename'] = narrFlx
fieldinfo['lcl']['fname'] = ['PRES_221_HTGL','TMP_221_HTGL','DPT_221_HTGL']
fieldinfo['lcl']['levels'] = [400, 500, 600, 700, 750, 800, 850, 875, 900, 925, 950, 975, 1000]
fieldinfo['lcl']['units'] = 'hPa'
fieldinfo['lcl']['vertical'] = 2 * munits.m
#fieldinfo['lh']  = {'levels' : [-1280,-640,-320,-160,-80,-40,-20,0,10,20,40], 'cmap':readNCLcm('MPL_BrBG')[127:40:-1], 'fname'  : ['LHTFL_221_SFC'], 'filename':narrSfc}
fieldinfo['lh']  = {'levels' : list(range(-700,150,50)), 'cmap':readNCLcm('MPL_BrBG')[127:30:-1], 'fname'  : ['LHTFL_221_SFC'], 'filename':narrSfc}
fieldinfo['mslp']['fname'] = ['PRMSL_221_MSL']
fieldinfo['mslp']['filename'] = narrFlx
fieldinfo['mslp']['levels'] = np.arange(956,1028,4)
fieldinfo['mslp']['units'] = 'hPa'
fieldinfo['mslet'] = fieldinfo['mslp'].copy()
fieldinfo['mslet'] = ['MSLET_221_MSL']
fieldinfo['sbcape']['fname'] = ['CAPE_221_SFC']
fieldinfo['sbcape']['vertical'] = 'surface-based'
fieldinfo['sbcinh']['cmap'].reverse()
fieldinfo['sbcinh']['fname'] = ['CIN_221_SFC']
fieldinfo['sbcinh']['levels'].reverse()
fieldinfo['sbcinh']['levels'] = [-np.round(x/2) for x in fieldinfo['sbcinh']['levels']] # NARR cin is negative, and halve levels
fieldinfo['sbcinh']['vertical'] = 'surface-based'
fieldinfo['mlcape']['filename'] = narrPBL # 180-0 mb above ground
fieldinfo['mlcape']['fname'] = ['CAPE_221_SPDY'] # 180-0 mb above ground
fieldinfo['mlcape']['vertical'] = 'mixed-layer'
fieldinfo['mlcinh'] = fieldinfo['sbcinh'].copy()
fieldinfo['mlcinh']['filename'] = narrPBL
fieldinfo['mlcinh']['fname'] = ['CIN_221_SPDY']
fieldinfo['mlcinh']['vertical'] = 'mixed-layer'
fieldinfo['mucape']['vertical'] = 'most unstable'
fieldinfo['pblh']['fname'] = ['HPBL_221_SFC']
fieldinfo['precipacc']['fname'] = ['RAINNC']
#fieldinfo['pwat']['levels'] = [5,10,20,25,30,40,50,60,70,80,90,100] # precipitable water in kg/m**2 not depth-of-water
fieldinfo['pwat']['levels'] = [20,25,30,35,40,45,50,55,60,65,70] # precipitable water in kg/m**2 not depth-of-water
fieldinfo['pwat']['fname'] = ['P_WAT_221_EATM']
fieldinfo['pwat']['filename'] = narrSfc
fieldinfo['pwat']['temporal'] = 0
fieldinfo['rh_0deg'] = fieldinfo['rh700'] # thought about adding 'vertical' but there is no vertical dimension
fieldinfo['rh_0deg']['filename'] = narrFlx
fieldinfo['rh_0deg']['fname'] = ['R_H_221_0DEG']
fieldinfo['scp'] = fieldinfo['stp'].copy()
fieldinfo['scp']['fname'] = ['CAPE_221_SFC','CIN_221_SFC','HLCY_221_HTGY','shr10_700']
fieldinfo['scp']['filename'] = narrSfc
fieldinfo['sh2']    = {'levels' : [0.5,1,2,4,8,12,14,16,17,18,19,20,21,22,23,24], 'cmap':fieldinfo['td2']['cmap'], 'fname'  : ['SPF_H_221_HTGL'], 'filename':narrFlx, 'vertical':2*munits.m, 'units':'g/kg'}
fieldinfo['shlev1'] = {'levels' : [0.5,1,2,4,8,12,14,16,17,18,19,20,21,22,23,24], 'cmap':fieldinfo['td2']['cmap'], 'fname'  : ['SPF_H_221_HYBL'], 'filename':narrFlx, 'units':'g/kg'}
fieldinfo['shr10_30']  = fieldinfo['speed10m'].copy()
fieldinfo['shr10_30']['levels'] = range(0,54,3)
#shrlev1_trop = wind shear between tropopause and lowest model level
for shr in ['shr10_500', 'shr10_700', 'shr10_850', 'shr10_900', 'shr10_925', 'shr30_500', 'shr30_700', 'shr850_200', 'shrlev1_trop']:
    fieldinfo[shr] = fieldinfo['shr10_30'].copy()
    bot, top = shr[3:].split("_")
    if top.isnumeric():
        top = int(top) * munits("hPa")
    if bot.isnumeric():
        bot = int(bot)
        if bot <= 30:
            bot *= munits.m
        else:
            bot *= munits("hPa")
    fieldinfo[shr]['vertical'] = [bot,top]
fieldinfo['shrtrop'] = {'levels':np.array([2,5,10,15,20,30,50])*1e-3, 'cmap': fieldinfo['speed700']['cmap'], 
                        'filename': narrFlx, 'fname': ['VWSH_221_TRO'], 'vertical':'tropopause'} # shear at tropopause. https://www.emc.ncep.noaa.gov/mmb/rreanl/faq.html 
fieldinfo['srh'] = fieldinfo['srh1'].copy()
fieldinfo['srh']['levels'].extend([750])
fieldinfo['srh']['cmap'].extend(readNCLcm('wind_17lev')[-6:-4])
fieldinfo['srh']['fname'] = ['HLCY_221_HTGY']
fieldinfo['srh']['filename'] = narrFlx
fieldinfo['stp']['fname'] = ['CAPE_221_SFC','CIN_221_SFC','HLCY_221_HTGY','shr10_500']
fieldinfo['stp']['filename'] = narrSfc
fieldinfo['surface_height'] = fieldinfo['sbcape'].copy()
fieldinfo['surface_height']['fname'] = ['HGT_221_SFC']
fieldinfo['surface_height']['filename'] = narrFixed
fieldinfo['t2']['fname'] = ['TMP_221_SFC']
fieldinfo['t2']['units'] = 'degF'
fieldinfo['tctp'] = fieldinfo['stp'].copy()
fieldinfo['tctp']['fname'][-1] = 'shr10_700'
fieldinfo['thetasfc'] = {'levels' : np.arange(290,320,2), 'cmap': ['#eeeeee', '#dddddd', '#cccccc', '#aaaaaa']+readNCLcm('precip2_17lev')[3:-1], 'fname'  : ['POT_221_SFC'], 'filename': narrSfc}
fieldinfo['theta2']   = {'levels' : np.arange(294,313,1), 'cmap': ['#eeeeee', '#dddddd', '#cccccc', '#aaaaaa']+readNCLcm('precip2_17lev')[3:-1], 'fname'  : ['PRES_221_HTGL', 'TMP_221_HTGL'], 'filename': narrFlx, 'vertical':2*munits.m}
fieldinfo['thetae2']  = {'levels' : np.arange(321,375,3), 'cmap': ['#eeeeee', '#dddddd', '#cccccc', '#aaaaaa']+readNCLcm('precip2_17lev')[3:-1], 'fname'  : ['PRES_221_HTGL', 'TMP_221_HTGL','DPT_221_HTGL'], 'filename': narrFlx, 'vertical':2*munits.m}
fieldinfo['vvellev1'] = fieldinfo['vvel700'].copy()
fieldinfo['vvellev1']['fname'] = ['V_VEL_221_HYBL']
fieldinfo['wvflux'] = {'fname':['WVUFLX_221_ISBY_acc3h','WVVFLX_221_ISBY_acc3h'],'filename':narrFlx,'arrow':True}
fieldinfo['wvfluxconv'] = {'fname':['WVCONV_221_ISBY_acc3h'],'filename':narrFlx,'levels':np.array(fieldinfo['precip']['levels'])*10,'cmap': readNCLcm('prcp_1')}
fieldinfo['wcflux'] = {'fname':['WCUFLX_221_ISBY_acc3h','WCVFLX_221_ISBY_acc3h'],'filename':narrFlx,'arrow':True}
fieldinfo['wcfluxconv'] = {'fname':['WCCONV_221_ISBY_acc3h'],'filename':narrFlx,'levels':np.array(fieldinfo['precip']['levels'])*2,'cmap': readNCLcm('prcp_1')}
fieldinfo['zlcl']['filename'] = narrFlx
fieldinfo['zlcl']['fname'] = ['PRES_221_HTGL','TMP_221_HTGL','DPT_221_HTGL']
fieldinfo['zlcl']['vertical'] = 2 * munits.m


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
    lon, lat = data.gridlon_221, data.gridlat_221
    # lon, lat = data.metpy.longitude, data.metpy.latitude # Thought about using metpy accessor, but I was warned "x" and "y" coordinates must je 1-D not 2-D in met-1.0 and beyond.
    # subtract 360 from positive longitudes west of dateline or else you get straight spagehetti across the plot
    # This fix may only work for NARR grid because all the problem points are in the northwest part of the grid
    # and the northeast has no positive longitudes. Considered warning metpy developers about this, but I think it's
    # a problem with the original grib or the way ncl_convert2nc converts it. 
    lon = lon.where(lon<0,lon-360) # use xarray method instead of numpy to keep as xarray. (lat is still xarray)

    return lon, lat

def units(data, info, debug=False):
    # Quantify xarray.
    data = data.metpy.quantify() # quantify in MetPy>=1
    # Convert to info["units"].
    if "units" in info:
        if debug:
            print("converting to "+info["units"])
        data = data.metpy.convert_units(info["units"]) # Don't .sel(lv_HTGL3=10) before .metpy.convert_units('kt'). You get None in return.
    return data



def temporal(data, info, debug=False):
    if debug:
        print("narr.temporal(): info",info)

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
            if debug:
                print('narr.temporal(): assume no temporal dimension. setting timetitle=',info['temporal'])
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
    if debug:
        print("narr.vertical(): data",data.name,"info",info)

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
                if debug:
                    print("narr.vertical():",data.name,vertical.name,'not in', data.dims,". assuming vertical has already had its way")
            verttitle = str(vlevel)
        elif len(data.shape) <= 2: # If data has only 2 dimensions assume it has no vertical dimension (like tropopause-level, or max-wind-level)
            if debug:
                print('narr.vertical():',data.name,'is 2D already.')
            print('narr.vertical(): setting verttitle=',vlevel)
            data.attrs['verttitle'] = vlevel
            return data
        else:
            # using list comprehension  # to get element with substring  
            res = [i for i in data.dims if 'lv_' == i[0:3]]
            vertical = res[0]
            if debug:
                print("narr.vertical(): metpy does not identify vertical coordinate. assume it is ("+vertical+")")
            data = data.sel({vertical:vlevel})
            vertical = data.coords[vertical]
            verttitle = str(vlevel) + vertical.units
        if debug:
            print("narr.vertical(): setting verttitle",verttitle)
        data.attrs['verttitle'] = verttitle

    return data



def shear(field, valid_time=None, targetdir=None, debug=False):
    # assumes field is a string 'shr(bot)_(top)'
    # where top is a constant pressure level in mb or 'trop'opause
    # bot is lowest model level if bot is 'lev1' # no other model levels available in NARR
    # or a constant pressure level in mb if bot > 30
    # or height in meters AGL if bot <= 30.
    m = re.search('^shr(.*)_(.*)', field)
    if m:
        bot = m.group(1)
        top = m.group(2)
    else:
        print("narr.shear(): expected field to start with 'shr' and have underscore between bottom and top levels")
        sys.exit(1)

    if debug:
        print("narr.shear(): bot",bot,"top",top)

    # winds are found in the flx or 3D file. Open both.
    flx_file = get(valid_time, targetdir=targetdir, narrtype=narrFlx)
    nc_flx = xarray.open_dataset(flx_file)
    ifile = get(valid_time, targetdir=targetdir, narrtype=narr3D)
    nc3D = xarray.open_dataset(ifile)

    # ubot and vbot
    if bot == 'lev1': # lowest model level
        info = {} # no vertical dimension
        ubot = vertical(nc_flx['U_GRD_221_HYBL'], info, debug)
        vbot = vertical(nc_flx['V_GRD_221_HYBL'], info, debug)
        ubot.attrs['verttitle'] = 'lowest model level'
        vbot.attrs['verttitle'] = 'lowest model level'
    elif int(bot) <= 30:
        info = {'vertical':int(bot)*munits.m}
        ubot = vertical(nc_flx['U_GRD_221_HTGL'], info, debug)
        vbot = vertical(nc_flx['V_GRD_221_HTGL'], info, debug)
    else:
        info = {'vertical':int(bot)*munits("hPa")}
        ubot = vertical(nc3D['U_GRD_221_ISBL'], info, debug)
        vbot = vertical(nc3D['V_GRD_221_ISBL'], info, debug)

    # utop and vtop
    if top == 'trop':
        info = {} # no vertical key
        utop = vertical(nc_flx['U_GRD_221_TRO'], info, debug)
        vtop = vertical(nc_flx['V_GRD_221_TRO'], info, debug)
        utop.attrs['verttitle'] = 'tropopause'
        vtop.attrs['verttitle'] = 'tropopause'
    elif int(top) <= 30:
        info = {'vertical':int(top)*munits.m}
        utop = vertical(nc_flx['U_GRD_221_HTGL'], info, debug)
        vtop = vertical(nc_flx['V_GRD_221_HTGL'], info, debug)
    else:
        info = {'vertical':int(top)*munits("hPa")}
        utop = vertical(nc3D['U_GRD_221_ISBL'], info, debug)
        vtop = vertical(nc3D['V_GRD_221_ISBL'], info, debug)

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
    du.attrs['verttitle'] = f"{ubot.attrs['verttitle']} to {utop.attrs['verttitle']}"
    dv.attrs['verttitle'] = f"{vbot.attrs['verttitle']} to {vtop.attrs['verttitle']}"
    return du, dv

def pressure_to_height_AGL(target_p, hgt3D, targetdir=".", debug=False):
  
    junktime = datetime.datetime(2015,1,1)
    surface_height = scalardata('surface_height', junktime, targetdir=targetdir, debug=debug)

    height_MSL =  pressure_to_height_std(target_p)
    
    # subtract surface geopotential height to get height AGL
    # switch from xarray to pint quantities for now.
    height_AGL =  height_MSL - surface_height.metpy.unit_array

    lv_ISBL0 = hgt3D.lv_ISBL0.metpy.unit_array
    
    # subtract pint quantities. surface_height broadcasted to multiple vertical levels that hgt3D has. 
    AGL3D = hgt3D.metpy.unit_array - surface_height.metpy.unit_array

    shp = target_p.shape
    nz = len(lv_ISBL0)
    data = np.empty(shp)
    for i,p in enumerate(target_p.metpy.unit_array.flatten()):
        ituple = np.unravel_index(i,shp) # 277 lat , 349 lon  - ituple may have 2 or 3 dims
        # interpolate linearly in ln(p). convert to same units, remove units, before applying natural log 
        lclinterp = np.interp(np.log(p.to('hPa').m), np.log(lv_ISBL0.to('hPa').m), AGL3D[:,ituple[-2],ituple[-1]]) 
        if not hasattr(lclinterp, "units"):
            # TODO: figure out why lclinterp sometimes has and sometimes doesn't have this attribute.
            lclinterp *= AGL3D.units # Prior to pint 0.11 np.interp lost units
        if debug: # this slows things down
            if np.abs(lclinterp - height_AGL[ituple]) > 415 * munits.meter:
                print('ituple={} lclinterp={} lclstd={}'.format(ituple,lclinterp,height_AGL[ituple]))
                pdb.set_trace()
        # strip units with .m magnitude attribute or get ValueError: setting an array element with a sequence.
        data[ituple] = lclinterp.m # this is slow if you update an xarray. speed things up by updating a ndarray.

    # Change pint quantity to xarray
    attrs = hgt3D.attrs
    # Units were stripped earlier from lclinterp. Make they are what was expected.
    assert attrs["units"] == lclinterp.units or (attrs["units"] == "gpm" and lclinterp.units == "meter")
    #attrs['units'] = lclinterp.units # replaces 'gpm' with 'meters'. TODO: figure out why this causes AttributeError: 'NoneType' object has no attribute 'evaluate' later when significant tornado function is called. 
    height_AGL = xarray.DataArray(data = data, coords=target_p.coords, dims=target_p.dims, name='height_AGL', attrs=attrs)
    height_AGL.attrs['long_name'] = 'height AGL'
    return height_AGL

def scalardata(field, valid_time, targetdir=".", debug=False):
    # Get color map, levels, and netCDF variable name appropriate for requested variable (from fieldinfo module).
    info = fieldinfo[field]
    if not isinstance(info['cmap'], (colors.ListedColormap)):
        info['cmap'] = colors.ListedColormap(info['cmap']) 
    if debug:
        print("scalardata: found",field,"fieldinfo:",info)
    fvar = info['fname'][0]


    # Get narr file and filename.
    ifile = get(valid_time, targetdir=targetdir, narrtype=info['filename'])

    if debug:
        print("About to open "+ifile)
    nc = xarray.open_dataset(ifile)
    # Tried to rename vars and dimensions so metpy.parse_cf() would not warn "Found latitude/longitude values, assuming latitude_longitude for projection grid_mapping variable"
    # It didn't help. Only commenting out the metpy.parse_cf() line helped.
    # It didn't help with MetpyDeprecationWarning: Multidimensional coordinate lat assigned for axis "y". This behavior has been deprecated and will be removed in v1.0 (only one-dimensional coordinates will be available for the "y" axis) either
    #nc = nc.rename_vars({"gridlat_221": "lat", "gridlon_221" : "lon"})
    #nc = nc.rename_dims({"gridx_221": "x", "gridy_221" : "y"})
    #nc = nc.metpy.parse_cf() # TODO: figure out why filled contour didn't have .metpy.parse_cf()

    if fvar not in nc.variables:
        print(fvar,"not in",ifile,'. Try', nc.var())
        sys.exit(1)

    # Define data array. Speed and shear derived differently.
    # Define 'long_name' attribute
    # 
    if field[0:5] == "speed":
        u = nc[info['fname'][0]]
        v = nc[info['fname'][1]]
        data = wind_speed(u, v)
        data.name = field
        data.attrs['long_name'] = "wind speed"
    elif field[0:3] == 'shr' and '_' in field:
        du, dv = shear(field, valid_time=valid_time, targetdir=targetdir, debug=debug)
        data = wind_speed(du, dv)
        data.name = field
        data.attrs.update({'long_name':'wind shear', 'verttitle' :du.attrs["verttitle"]})
    elif field[0:2] == 'rh':
        pres = nc['lv_ISBL0']
        temp = nc['TMP_221_ISBL']
        sh   = nc['SPF_H_221_ISBL']
        data = mcalc.relative_humidity_from_specific_humidity(pres, temp, sh)
        data.name = field 
        data.attrs['long_name'] = "relative humidity"
    elif field == 'theta2':
        pres = nc[info['fname'][0]]
        temp = nc[info['fname'][1]]
        theta = potential_temperature(pres, temp)
        theta.attrs['long_name'] = 'potential temperature'
        data = theta
    elif field == 'thetae2':
        pres = nc[info['fname'][0]]
        temp = nc[info['fname'][1]]
        dwpt = nc[info['fname'][2]]
        thetae = mcalc.equivalent_potential_temperature(pres, temp, dwpt)
        thetae.attrs['long_name'] = 'equivalent potential temperature'
        data = thetae
    elif field == 'scp' or field == 'stp' or field == 'tctp':
        cape = nc[info['fname'][0]]
        cin  = nc[info['fname'][1]]
        ifile = get(valid_time, targetdir=targetdir, narrtype=narrFlx)
        ncFlx = xarray.open_dataset(ifile)
        # metpy.parse_cf() would warn "Found latitude/longitude values, assuming latitude_longitude for projection grid_mapping variable"
        #ncFlx = ncFlx.metpy.parse_cf() 
        srh  = ncFlx[info['fname'][2]].metpy.quantify()
        shear_layer = info['fname'][3]
        bulk_shear = scalardata(shear_layer, valid_time, targetdir=targetdir, debug=debug)
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
            stp = significant_tornado(cape, lifted_condensation_level_height, srh, bulk_shear) * cin_term.metpy.unit_array
            attrs = {'long_name': 'significant tornado parameter'} # , 'verttitle':lifted_condensation_level_height.attrs['verttitle']} # don't want "2 meter" verttitle
            data = xarray.DataArray(data=stp, name=field, attrs=attrs) 
        if field == 'tctp':
            tctp = srh/(40*munits['m**2/s**2']) * bulk_shear/(12*munits['m/s']) * (2000*munits.m - lifted_condensation_level_height)/(1400*munits.m)
            # NARR storm relative helicity (srh) is 0-3 km AGL, while original TCTP expects 0-1 km AGL. 
            # So the shear term is too large using the NARR srh. Normalize the srh term with a larger denominator. 
            # In STP, srh is normalized by 150 m**2/s**2. Use that.
            tctp_0_3kmsrh = srh/(150*munits['m**2/s**2']) * bulk_shear/(12*munits['m/s']) * (2000*munits.m - lifted_condensation_level_height)/(1400*munits.m)
            attrs = {'long_name': 'TC tornado parameter'}
            data = xarray.DataArray(data=tctp_0_3kmsrh, name=field, attrs=attrs)
    elif field=='lcl':
        pres = nc[info['fname'][0]]
        temp = nc[info['fname'][1]]
        dwpt = nc[info['fname'][2]]
        LCL_pressure, LCL_temperature = mcalc.lcl(pres.fillna(pres.mean()), temp.fillna(temp.mean()), dwpt.fillna(dwpt.mean()))
        attrs = {"long_name":"lifted condensation level", "from":"metpy.calc.lcl"}
        # assign coords and dims manually because LCL_pressure is only an array of Pint quantities. 
        data = xarray.DataArray(data = LCL_pressure, coords=pres.coords, dims=pres.dims, name='LCL', attrs=attrs)
    elif field=='zlcl':
        LCL_pressure = scalardata('lcl', valid_time, targetdir=targetdir, debug=debug)
        ifile = get(valid_time, targetdir=targetdir, narrtype=narr3D)
        nc3D = xarray.open_dataset(ifile)
        # metpy.parse_cf() would warn "Found latitude/longitude values, assuming latitude_longitude for projection grid_mapping variable"
        # nc3D = nc3D.metpy.parse_cf()
        hgt3D = nc3D["HGT_221_ISBL"] 
        data = pressure_to_height_AGL(LCL_pressure, hgt3D, targetdir=targetdir)
        data.attrs['long_name']="lifted condensation level height AGL"
    else:
        data = nc[fvar]
    data = units(data, info, debug=debug)
    data = vertical(data, info, debug=debug)
    data = temporal(data, info, debug=debug)

    data.attrs['field'] = field
    data.attrs['ifile'] = os.path.realpath(ifile)
    # use np.array to allow for levels to be a range
    levels = np.array(info['levels'])
    data.attrs['levels'] = levels
    data.attrs.update(info)

    if isinstance(data.data, munits.Quantity):
        pass
        # Used to Add units to levels array if data has units. 
        # but this screwed up NARR_composite plt.savefig(). pint.errors.DimensionalityError: Cannot convert from 'degree_Celsius' to 'dimensionless'
        #data.attrs['levels'] = data.attrs['levels'] * data.data.units
    # Assume units of levels array is same as data.
    if data.min() > (data.metpy.units * levels).max() or data.max() < (data.metpy.units * levels).min():
        print('levels',levels,'out of range of data')
        print(data.min())
        print(data.max())

    return data


def vectordata(field, valid_time, targetdir=".", combineuv=True, debug=False):
    # Get color map, levels, and netCDF variable name appropriate for requested variable (from fieldinfo module).
    info = fieldinfo[field]
    if debug:
        print("vectordata(): found",field,"fieldinfo. Using",info)
        pdb.set_trace()
    if field[0:3] == "shr":
        u,v = shear(field, valid_time, targetdir=targetdir, debug=debug)
        u = temporal(u, info, debug=debug) # shear doesn't apply temporal like scalardata does.
        v = temporal(v, info, debug=debug)
    else:
        uname = 'u'+str(info['vertical'].m)
        vname = 'v'+str(info['vertical'].m)
        # TODO: fix this hack
        if uname == 'u10': uname = 'u10m' 
        if vname == 'v10': vname = 'v10m' 
        u = scalardata(uname, valid_time, targetdir=targetdir, debug=debug)
        v = scalardata(vname, valid_time, targetdir=targetdir, debug=debug)
    if 'arrow' in info:
        u.attrs['arrow'] = True
        v.attrs['arrow'] = True
    if combineuv:
        # The second argument to concat can also be an Index or DataArray object as well as a string, in which case it is used to label values along the new dimension:
        uv = xarray.concat([u,v],pd.Index(["u","v"], name='uv'))
        uv.attrs.update(info) # shear() doesn't copy over attributes like scalardata does
        uv.attrs['long_name'] = u.attrs['long_name'].replace('u-component of ', '')
        uv.name = uv.name.replace("U_GRD","UV_GRD") 
        uv.attrs['field'] = field # 'field' attribute should have been added to u and v separately in scalardata().
        return uv
    return u, v



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
        return relative_humidity.sel(lv_ISBL0 = int(hPa)*munits("hPa")) # pressure level units ignored but included for clarity
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
                    _,_,srhs[istorm] = mcalc.storm_relative_helicity(h, u, v, 3*munits('km')) 
                print(point.values, hrs.values, storm.values, cc, srhs[istorm].values)
                mucape.loc[kwargs], mucin.loc[kwargs] = mucapes*munits("J/kg"), mucins* munits("J/kg")
                srh.loc[kwargs] = srhs * munits("m**2/s**2")

        t1_stop = perf_counter()
        print("Elapsed time:", t1_stop-t1_start, 's')
        pdb.set_trace()
        u6, v6 = u.sel(hgt=6*munits('km')), v.sel(hgt=6*munits('km'))
        u0, v0 = u.sel(hgt=0*munits('km')), v.sel(hgt=0*munits('km'))
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
            stp = significant_tornado(cape, lifted_condensation_level_height, srh, bulk_shear) * cin_term.metpy.unit_array
            attrs = {'long_name': 'significant tornado parameter'} # , 'verttitle':lifted_condensation_level_height.attrs['verttitle']} # don't want "2 meter" verttitle
            data = xarray.DataArray(data=stp, name=field, attrs=attrs) 
        if field == 'tctp':
            tctp = srh/(40*munits['m**2/s**2']) * bulk_shear/(12*munits['m/s']) * (2000*munits.m - lifted_condensation_level_height)/(1400*munits.m)
            # NARR storm relative helicity (srh) is 0-3 km AGL, while original TCTP expects 0-1 km AGL. 
            # So the shear term is too large using the NARR srh. Normalize the srh term with a larger denominator. 
            # In STP, srh is normalized by 150 m**2/s**2. Use that.
            tctp_0_3kmsrh = srh/(150*munits['m**2/s**2']) * bulk_shear/(12*munits['m/s']) * (2000*munits.m - lifted_condensation_level_height)/(1400*munits.m)
            attrs = {'long_name': 'TC tornado parameter'}
            data = xarray.DataArray(data=tctp_0_3kmsrh, name=field, attrs=attrs)
    elif field=='lcl':
        pres = nc['p']
        temp = temperature
        dwpt = nc[info['fname'][2]]
        LCL_pressure, LCL_temperature = mcalc.lcl(pres.fillna(pres.mean()), temp.fillna(temp.mean()), dwpt.fillna(dwpt.mean()))
        attrs = {"long_name":"lifted condensation level", "from":"metpy.calc.lcl"}
        # assign coords and dims manually because LCL_pressure is only an array of Pint quantities. 
        data = xarray.DataArray(data = LCL_pressure, coords=pres.coords, dims=pres.dims, name='LCL', attrs=attrs)
    elif field=='zlcl':
        LCL_pressure = scalardata('lcl', valid_time, targetdir=targetdir, debug=debug)
        ifile = get(valid_time, targetdir=targetdir, narrtype=narr3D)
        nc3D = xarray.open_dataset(ifile)
        # metpy.parse_cf() would warn "Found latitude/longitude values, assuming latitude_longitude for projection grid_mapping variable"
        # nc3D = nc3D.metpy.parse_cf()
        hgt3D = nc3D["HGT_221_ISBL"] 
        data = pressure_to_height_AGL(LCL_pressure, hgt3D, targetdir=targetdir)
        data.attrs['long_name']="lifted condensation level height AGL"
    elif field=='srh1':
        print(f"Can't derive {field} yet")
    elif field=='srh3':
        print(f"Can't derive {field} yet")
    else:
        data = nc[fvar]

    return data

def get_normalize_range_by_value(df, index, normalize_by, debug=False):
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
                print("get_normalize_range_by_value(): r34 is zero. Normalize by {:.0f} nautical miles".format(value))
        else:
            print("get_normalize_range_by_value(): Unexpected 'rad' value", rad)
            sys.exit(1)
    elif normalize_by == 'Vt500km':
        valid_time = df.loc[index, "valid_time"]
        # Had targetdir set to "." but it grabbed and converted NARR grb in the current directory
        data = narr.vectordata("wind10m", valid_time, targetdir=workdir, combineuv=True, debug=debug)
        lon, lat = narr.get_ll(data)
        u = data.sel(uv="u").values
        v = data.sel(uv="v").values
        derived_vitals_dict = atcf.derived_winds(u, v, np.full_like(u, 1013.), lon.values, lat.values, df.loc[index, :], debug=True)
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
        print(f'Vt_500km_kts {derived_vitals_dict["Vt_500km_kts"]:.2f}kts  S={storm_size_S:.3f}')
        value = storm_size_S * units("km") # This needs units when you divide distance_km by it. 
        # Originally took inverse of storm_size_S, but that is wrong. If you have a storm 10% larger than normal, 
        # you want to pull everything 10% closer to the origin, so it matches up with other storms that are normal sized.
        # The radial distance is divided by this value.
        assert value != 0, "can't be zero"+str(df.loc[index,:])
        if np.isnan(value):
            print ("normalize value can't be nan")
            pdb.set_trace()
        return value
    else:
        value = df.loc[index, normalize_by]
        if np.isnan(value):
            if normalize_by == 'rmw':
                value = 25. # 25 nautical miles is default rmw in aswip.
            else:
                print("get_normalize_range_by_value(): Null value for", normalize_by)
                print("not sure how to define")
                sys.exit(1)


    value = value * units["nautical_mile"].to("km")


    assert value != 0, "can't be zero"+str(df.loc[index,:])
    assert not np.isnan(value), "can't be nan"+str(df.loc[index,:])

    return value


