from fieldinfo import fieldinfo, readNCLcm
import os # for NCARG_ROOT
import numpy as np
import subprocess
import tarfile
import pytz
import re
import pdb
import xarray
import datetime
import sys
import glob
from metpy.units import units as munits
from metpy.calc import equivalent_potential_temperature, lcl, wind_speed, potential_temperature, significant_tornado, pressure_to_height_std, supercell_composite
from metpy.interpolate import log_interpolate_1d
import matplotlib.colors as colors

# (vartype, file_suffix)
narrSfc = ('sfc', 'RS.sfc')
narrFlx = ('flx', 'RS.flx')
narrPBL = ('pbl', 'RS.pbl')
narr3D  = ('3D', '3D')
narrFixed  = "/glade/scratch/"+os.getenv("USER")+"/NARR/rr-fixed.grb.nc"

# Modify fieldinfo dictionary for NARR. 
for v in fieldinfo:
    fieldinfo[v]['filename'] = narrSfc
plevs = ['200', '250', '300', '500', '700', '850', '900', '925','10m']
for plev in plevs:
    vertical = int(re.search(r'\d+', plev)[0]) # extract numeric part as integer
    for ws in ['wind', 'speed']:
        f = ws+plev
        if f not in fieldinfo: fieldinfo[f] = {}
        fieldinfo[f]['filename'] = narr3D
        fieldinfo[f]['fname'] = ['U_GRD_221_ISBL','V_GRD_221_ISBL']
        fieldinfo[f]['vertical'] = vertical
        #fieldinfo[f]['units'] = 'knots' # should remain m/s for publications
        if plev == "10m":
            fieldinfo[f]['fname'] = ['U_GRD_221_HTGL', 'V_GRD_221_HTGL']
            fieldinfo[f]['filename'] = narrFlx
    uwind = 'u'+plev
    vwind = 'v'+plev
    for wind_component in [uwind, vwind]:
        fieldinfo[wind_component] = fieldinfo["speed"+plev].copy() # use filename, and vertical from speed
        if plev == "10m":
            UorV = wind_component[0:1].upper()
            fieldinfo[wind_component]['fname'] = [UorV+'_GRD_221_HTGL']
            fieldinfo[wind_component]['filename'] = narrFlx
    hgt = 'hgt'+plev
    if hgt not in fieldinfo: fieldinfo[hgt] = {}
    fieldinfo[hgt]['filename'] = narr3D
    fieldinfo[hgt]['fname'] = ['HGT_221_ISBL']
    fieldinfo[hgt]['vertical'] = vertical
    vvel = 'vvel'+plev
    fieldinfo[vvel] = {'levels' : [-250,-100,-25,-10,-2.5,-1,1,2.5,10,25,100,250], 'cmap': readNCLcm('cmocean_balance')[::-1], 'fname'  : ['V_VEL_221_ISBL'], 'filename': narr3D}
    fieldinfo[vvel]['cmap'][127] = "white"
    fieldinfo[vvel]['vertical'] = vertical
    fieldinfo[vvel]['units'] = 'microbar/second'
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
fieldinfo['lcl']['vertical'] = 2
#fieldinfo['lh']  = {'levels' : [-1280,-640,-320,-160,-80,-40,-20,0,10,20,40], 'cmap':readNCLcm('MPL_BrBG')[127:40:-1], 'fname'  : ['LHTFL_221_SFC'], 'filename':narrSfc}
fieldinfo['lh']  = {'levels' : list(range(-700,150,50)), 'cmap':readNCLcm('MPL_BrBG')[127:30:-1], 'fname'  : ['LHTFL_221_SFC'], 'filename':narrSfc}
fieldinfo['mslp']['fname'] = ['PRMSL_221_MSL']
fieldinfo['mslp']['filename'] = narrFlx
fieldinfo['mslp']['units'] = 'hPa'
fieldinfo['mslet'] = fieldinfo['mslp'].copy()
fieldinfo['mslet'] = ['MSLET_221_MSL']
fieldinfo['sbcape']['fname'] = ['CAPE_221_SFC']
fieldinfo['sbcinh']['fname'] = ['CIN_221_SFC']
fieldinfo['sbcinh']['levels'].reverse()
fieldinfo['sbcinh']['levels'] = [-x for x in fieldinfo['sbcinh']['levels']] # NARR cin is negative
fieldinfo['sbcinh']['cmap'].reverse()
fieldinfo['mlcape']['filename'] = narrPBL # 180-0 mb above ground
fieldinfo['mlcape']['fname'] = ['CAPE_221_SPDY'] # 180-0 mb above ground
fieldinfo['mlcinh'] = fieldinfo['sbcinh'].copy()
fieldinfo['mlcinh']['filename'] = narrPBL
fieldinfo['mlcinh']['fname'] = ['CIN_221_SPDY']
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
fieldinfo['scp']['vertical'] = 2
fieldinfo['sh2']    = {'levels' : [0.2,0.5,1,2,4,8,12,14,16,18,20,22,24], 'cmap':fieldinfo['td2']['cmap'], 'fname'  : ['SPF_H_221_HTGL'], 'filename':narrFlx, 'vertical':2, 'units':'g/kg'}
fieldinfo['shlev1'] = {'levels' : [0.2,0.5,1,2,4,8,12,14,16,18,20,22,24], 'cmap':fieldinfo['td2']['cmap'], 'fname'  : ['SPF_H_221_HYBL'], 'filename':narrFlx, 'units':'g/kg'}
fieldinfo['shr10_30']  = fieldinfo['speed10m'].copy()
del(fieldinfo['shr10_30']['vertical']) # shear is 2 levels. vertical is undefined
fieldinfo['shr10_500'] = fieldinfo['shr10_30'].copy()
fieldinfo['shr10_700'] = fieldinfo['shr10_30'].copy()
fieldinfo['shr10_850'] = fieldinfo['shr10_30'].copy()
fieldinfo['shr10_900'] = fieldinfo['shr10_30'].copy()
fieldinfo['shr10_925'] = fieldinfo['shr10_30'].copy()
fieldinfo['shr30_500'] = fieldinfo['shr10_30'].copy()
fieldinfo['shr30_700'] = fieldinfo['shr10_30'].copy()
fieldinfo['shr850_200'] = fieldinfo['shr10_30'].copy()
fieldinfo['shrtrop'] = {'levels':np.array([2,5,10,15,20,30,50])*1e-3, 'cmap': fieldinfo['speed700']['cmap'], 
                        'filename': narrFlx, 'fname': ['VWSH_221_TRO'], 'vertical':'tropopause'} # shear at tropopause. https://www.emc.ncep.noaa.gov/mmb/rreanl/faq.html 
fieldinfo['shrlev1_trop'] = fieldinfo['shr10_700'].copy() # wind shear between tropopause and lowest model level
fieldinfo['srh'] = fieldinfo['srh1'].copy()
fieldinfo['srh']['levels'].extend([750])
fieldinfo['srh']['cmap'].extend(readNCLcm('wind_17lev')[-6:-4])
fieldinfo['srh']['fname'] = ['HLCY_221_HTGY']
fieldinfo['srh']['filename'] = narrFlx
fieldinfo['stp']['fname'] = ['CAPE_221_SFC','CIN_221_SFC','HLCY_221_HTGY','shr10_500']
fieldinfo['stp']['filename'] = narrSfc
fieldinfo['stp']['vertical'] = 2
fieldinfo['surface_height'] = fieldinfo['sbcape'].copy()
fieldinfo['surface_height']['fname'] = ['HGT_221_SFC']
fieldinfo['surface_height']['filename'] = narrFixed
fieldinfo['t2']['fname'] = ['TMP_221_SFC']
fieldinfo['t2']['units'] = 'degF'
fieldinfo['tctp'] = fieldinfo['stp'].copy()
fieldinfo['tctp']['fname'][-1] = 'shr10_700'
fieldinfo['thetasfc'] = {'levels' : np.arange(288,320,2), 'cmap': ['#eeeeee', '#dddddd', '#cccccc', '#aaaaaa']+readNCLcm('precip2_17lev')[3:-1], 'fname'  : ['POT_221_SFC'], 'filename': narrSfc}
fieldinfo['theta2'] = {'levels' : np.arange(288,320,2), 'cmap': ['#eeeeee', '#dddddd', '#cccccc', '#aaaaaa']+readNCLcm('precip2_17lev')[3:-1], 'fname'  : ['PRES_221_HTGL', 'TMP_221_HTGL'], 'filename': narrFlx, 'vertical':2}
fieldinfo['thetae2'] = {'levels' : np.arange(312,376,4), 'cmap': ['#eeeeee', '#dddddd', '#cccccc', '#aaaaaa']+readNCLcm('precip2_17lev')[3:-1], 'fname'  : ['PRES_221_HTGL', 'TMP_221_HTGL','DPT_221_HTGL'], 'filename': narrFlx, 'vertical':2}
fieldinfo['vvellev1'] = fieldinfo['vvel700'].copy()
fieldinfo['vvellev1']['fname'] = ['V_VEL_221_HYBL']
fieldinfo['wvflux'] = {'fname':['WVUFLX_221_ISBY_acc3h','WVVFLX_221_ISBY_acc3h'],'filename':narrFlx,'arrow':True}
fieldinfo['wvfluxconv'] = {'fname':['WVCONV_221_ISBY_acc3h'],'filename':narrFlx,'levels':np.array(fieldinfo['precip']['levels'])*10,'cmap': readNCLcm('prcp_1')}
fieldinfo['wcflux'] = {'fname':['WCUFLX_221_ISBY_acc3h','WCVFLX_221_ISBY_acc3h'],'filename':narrFlx,'arrow':True}
fieldinfo['wcfluxconv'] = {'fname':['WCCONV_221_ISBY_acc3h'],'filename':narrFlx,'levels':np.array(fieldinfo['precip']['levels'])*2,'cmap': readNCLcm('prcp_1')}
fieldinfo['zlcl']['filename'] = narrFlx
fieldinfo['zlcl']['fname'] = ['PRES_221_HTGL','TMP_221_HTGL','DPT_221_HTGL']
fieldinfo['zlcl']['vertical'] = 2

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


def units(data, info, debug=False):
    if "units" in info:
        if debug:
            print("converting to "+info["units"])
        data.metpy.convert_units(info["units"]) # Don't .sel(lv_HTGL3=10) before .metpy.convert_units('kt'). You get None in return.
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
            data = data.loc[time0]
        data.attrs['timetitle'] = str(time0) + "-h" # TODO: only works with zero. convert [ns] to hours. 

    return data



def vertical(data, info, debug=False):
    if debug:
        print("narr.vertical(): data",data.name,"info",info)

    # Define 'verttitle' attribute of data, no matter what.
    if 'verttitle' not in data.attrs:
        data.attrs['verttitle'] = ''

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
        elif len(data.shape) <= 2: # If data has only 2 dimensions assume it has no vertical dimension (like tropopause-level, or max-wind-level)
            if debug:
                print('narr.vertical():',data.name,'is 2D already.')
            if 'verttitle' not in data.attrs:
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
        info = {'vertical':int(bot)}
        ubot = vertical(nc_flx['U_GRD_221_HTGL'], info, debug)
        vbot = vertical(nc_flx['V_GRD_221_HTGL'], info, debug)
    else:
        info = {'vertical':int(bot)}
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
        info = {'vertical':int(top)}
        utop = vertical(nc_flx['U_GRD_221_HTGL'], info, debug)
        vtop = vertical(nc_flx['V_GRD_221_HTGL'], info, debug)
    else:
        info = {'vertical':int(top)}
        utop = vertical(nc3D['U_GRD_221_ISBL'], info, debug)
        vtop = vertical(nc3D['V_GRD_221_ISBL'], info, debug)

    du =  utop-ubot
    dv =  vtop-vbot
    du.attrs = utop.attrs
    dv.attrs = vtop.attrs
    du.attrs['long_name'] += ' shear'
    dv.attrs['long_name'] += ' shear'
    du.attrs['verttitle'] = ubot.attrs["verttitle"]+" to "+utop.attrs["verttitle"]
    dv.attrs['verttitle'] = vbot.attrs["verttitle"]+" to "+vtop.attrs["verttitle"]
    return du, dv

def pressure_to_height(target_p, hgt3D, debug=False):
  
    junktime = datetime.datetime(2015,1,1)
    surface_height = scalardata('surface_height', junktime, debug=debug)

    lcl_height_MSL =  pressure_to_height_std(target_p)
    
    # subtract surface geopotential height to get height AGL
    # switch from xarray to pint quantities for now.
    lcl_height_AGL =  lcl_height_MSL - surface_height.metpy.unit_array

    lv_ISBL0 = hgt3D.lv_ISBL0.metpy.unit_array

    # subtract pint quantities. surface_height broadcasted to multiple vertical levels in hgt3D 
    AGL3D = hgt3D.metpy.unit_array - surface_height.metpy.unit_array

    shp = target_p.shape
    nz = len(lv_ISBL0)
    junk = np.empty(shp)
    for i,p in enumerate(target_p.metpy.unit_array.flatten()):
        ituple = np.unravel_index(i,shp) # 277 lat , 349 lon  - ituple may have 2 or 3 dims
        # interpolate linearly in ln(p). convert to same units, remove units, before applying natural log 
        lclinterp = np.interp(np.log(p.to('hPa').m), np.log(lv_ISBL0.to('hPa').m), AGL3D[:,ituple[-2],ituple[-1]]) 
        if debug: # this slows things down
            lclinterp *= AGL3D.units # Prior to pint 0.11 np.interp lost units
            if np.abs(lclinterp - lcl_height_AGL[ituple]) > 415 * munits.meter:
                print('ituple={} lclinterp={} lclstd={}'.format(ituple,lclinterp,lcl_height_AGL[ituple]))
                pdb.set_trace()
        junk[ituple] = lclinterp.m # this is slow if you update an xarray. speed things up by updating a ndarray.

    # Change pint quantity to xarray
    attrs = hgt3D.attrs
    #attrs['units'] = lclinterp.units # replaces 'gpm' with 'meters'. TODO: figure out why this causes AttributeError: 'NoneType' object has no attribute 'evaluate' later when significant tornado function is called. 
    lcl_height_AGL = xarray.DataArray(data = junk, coords=target_p.coords, dims=target_p.dims, name='lcl_height_AGL', attrs=attrs)
    lcl_height_AGL.attrs['long_name'] = 'lifted condensation level height AGL'
    return lcl_height_AGL

def scalardata(field, valid_time, targetdir=".", debug=False):
    # Get color map, levels, and netCDF variable name appropriate for requested variable (from fieldinfo module).
    info = fieldinfo[field]
    if debug:
        print("scalardata: found",field,"fieldinfo:",info)
    cmap = colors.ListedColormap(info['cmap'])
    levels = info['levels']
    fvar = info['fname'][0]


    # Get narr file and filename.
    ifile = get(valid_time, targetdir=targetdir, narrtype=info['filename'])

    if debug:
        print("About to open "+ifile)
    nc = xarray.open_dataset(ifile).metpy.parse_cf() # TODO: figure out why filled contour didn't have .metpy.parse_cf()

    if fvar not in nc.variables:
        print(fvar,"not in",ifile,'. Try', nc.var())
        sys.exit(1)

    # Define data array. Speed and shear derived differently.
    # Define 'long_name' attribute
    # 
    if field[0:5] == "speed":
        u = nc[info['fname'][0]]
        v = nc[info['fname'][1]]
        data = u # copy metadata/coordinates from u
        data.values = wind_speed(u, v)
        data.attrs['long_name'] = "wind speed"
    elif field[0:3] == 'shr' and '_' in field:
        du, dv = shear(field, valid_time=valid_time, targetdir=targetdir, debug=debug)
        ws = wind_speed(du, dv)
        attrs = {'long_name':'wind shear', 'units':str(ws.units), 'verttitle' :du.attrs["verttitle"]}
        # Use .m magnitude because you can't transfer units of pint quantity to xarray numpy array (xarray.values)
        data = xarray.DataArray(data=ws.m, dims=du.dims, coords=du.coords, name=field, attrs=attrs)
    elif field == 'theta2':
        pres = nc[info['fname'][0]]
        temp = nc[info['fname'][1]]
        data = pres # retain xarray metadata/coordinates
        theta = potential_temperature(pres, temp)
        data.values = theta
        data.attrs['units'] = str(theta.units)
        data.attrs['long_name'] = 'potential temperature'
    elif field == 'thetae2':
        pres = nc[info['fname'][0]]
        temp = nc[info['fname'][1]]
        dwpt = nc[info['fname'][2]]
        data = pres # retain xarray metadata/coordinates
        thetae = equivalent_potential_temperature(pres, temp, dwpt)
        data.values = thetae
        data.attrs['units'] = str(thetae.units)
        data.attrs['long_name'] = 'equivalent potential temperature'
    elif field == 'scp' or field == 'stp' or field == 'tctp':
        cape = nc[info['fname'][0]]
        cin  = nc[info['fname'][1]]
        ifile = get(valid_time, targetdir=targetdir, narrtype=narrFlx)
        ncFlx = xarray.open_dataset(ifile).metpy.parse_cf() 
        srh  = ncFlx[info['fname'][2]]
        shear_layer = info['fname'][3]
        bulk_shear = scalardata(shear_layer, valid_time, targetdir=targetdir, debug=debug)
        lifted_condensation_level_height = scalardata('zlcl', valid_time, targetdir=targetdir, debug=debug)
       
        if field == 'scp':
            # In SPC help, cin is positive in SCP formulation.
            cin_term = -40/cin
            cin_term = cin_term.where(cin < -40, other=1)
            scp = supercell_composite(cape, srh, bulk_shear) * cin_term.metpy.unit_array
            attrs = {'units':str(scp.units), 'long_name': 'supercell composite parameter'}
            data = xarray.DataArray(data=scp, dims=cape.dims, coords=cape.coords, name=field, attrs=attrs) 
        if field == 'stp':
            cin_term = (200+cin)/150
            cin_term = cin_term.where(cin <= -50, other=1)
            cin_term = cin_term.where(cin >= -200, other=0)
            # CAPE, srh, bulk_shear, cin may be one vertical level, but LCL may be multiple heights.
            # xarray.broadcast() makes them all multiple heights with same shape, so significant_tornado doesn't 
            # complain about expecting lat/lon 2 dimensions and getting 3 dimensions..
            (cape, lifted_condensation_level_height, srh, bulk_shear, cin_term) = xarray.broadcast(cape, lifted_condensation_level_height, srh, bulk_shear, cin_term)
            stp = significant_tornado(cape, lifted_condensation_level_height, srh, bulk_shear) * cin_term.metpy.unit_array
            attrs = {'units':str(stp.units), 'long_name': 'significant tornado parameter', 'verttitle':lifted_condensation_level_height.attrs['verttitle']}
            data = xarray.DataArray(data=stp, dims=cape.dims, coords=cape.coords, name=field, attrs=attrs) 
        if field == 'tctp':
            tctp = srh/(40*munits['m**2/s**2']) * bulk_shear/(12*munits['m/s']) * (2000 - lifted_condensation_level_height)/(1400*munits.m)
            # NARR storm relative helicity is 0-3 km AGL. while TCTP expects 0-1 km AGL.
            # So the shear term is too large. Nornalize by a larger denominator. In STP, srh is normalized by 150 m**2/s**2
            tctp_0_3kmsrh = srh/(150*munits['m**2/s**2']) * bulk_shear/(12*munits['m/s']) * (2000 - lifted_condensation_level_height)/(1400*munits.m)
            data.values = tctp_0_3kmsrh
            data.attrs['long_name'] = 'TC tornado parameter'
    elif field=='lcl':
        pres = nc[info['fname'][0]]
        temp = nc[info['fname'][1]]
        dwpt = nc[info['fname'][2]]
        LCL_pressure, LCL_temperature = lcl(pres.fillna(pres.mean()), temp.fillna(temp.mean()), dwpt.fillna(dwpt.mean()))
        # convert units to string or xarray.DataArray.metpy.unit_array dies with ttributeError: 'NoneType' object has no attribute 'evaluate'
        attrs = {"long_name":"lifted condensation level", "units" : str(LCL_pressure.units), "from":"metpy.calc.lcl"}
        data = xarray.DataArray(data = LCL_pressure, coords=pres.coords, dims=pres.dims, name='LCL', attrs=attrs)
    elif field=='zlcl':
        LCL_pressure = scalardata('lcl', valid_time, targetdir=targetdir, debug=debug)
        ifile = get(valid_time, targetdir=targetdir, narrtype=narr3D)
        nc3D = xarray.open_dataset(ifile).metpy.parse_cf()
        hgt3D = nc3D["HGT_221_ISBL"] 
        data = pressure_to_height(LCL_pressure, hgt3D)
    else:
        data = nc[fvar]
    data = units(data, info, debug=debug)
    data = vertical(data, info, debug=debug)
    data = temporal(data, info, debug=debug)

    data.attrs['ifile'] = os.path.realpath(ifile)
    data.attrs['levels'] = levels
    data.attrs['cmap'] = cmap

    if data.min() > levels[-1] or data.max() < levels[0]:
        print('levels',levels,'out of range of data', data.min(), data.max())
        sys.exit(2)

    return data


def vectordata(field, valid_time, targetdir=".", debug=False):
    # Get color map, levels, and netCDF variable name appropriate for requested variable (from fieldinfo module).
    info = fieldinfo[field]
    if debug:
        print("found",field,"fieldinfo. Using",info)
    uname = info['fname'][0]
    vname = info['fname'][1]
    u = scalardata(uname, valid_time, targetdir=targetdir, debug=debug)
    v = scalardata(vname, valid_time, targetdir=targetdir, debug=debug)
    if 'arrow' in info:
        u.attrs['arrow'] = True
        v.attrs['arrow'] = True
    return u, v
