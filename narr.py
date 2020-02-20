from fieldinfo import fieldinfo, readNCLcm
import os # for NCARG_ROOT
import numpy as np
import subprocess
import tarfile
import re
import pdb
import xarray
import datetime
import glob


# Modify fieldinfo dictionary for MPAS. 
narrSfc = ('sfc', 'RS.sfc')
narrFlx = ('flx', 'RS.flx')
narr3D  = ('3D', '3D')
for v in fieldinfo:
    fieldinfo[v]['filename'] = narrSfc
plevs = ['200', '250', '300', '500', '700', '850', '900', '925','10m']
winds = ['wind'+plev for plev in plevs]
winds.extend(['speed'+plev for plev in plevs])
for wind in winds:
    if wind not in fieldinfo: fieldinfo[wind] = {}
    fieldinfo[wind]['filename'] = narr3D
    fieldinfo[wind]['fname'] = ['U_GRD_221_ISBL','V_GRD_221_ISBL']
    fieldinfo[wind]['vertical'] = int(re.search(r'\d+', wind)[0]) # extract numeric part as integer (works for 'wind10m' too)
    #fieldinfo[wind]['units'] = 'knots' # should remain m/s for publications
hgts = ['hgt'+plev for plev in plevs]
for hgt in hgts:
    if hgt not in fieldinfo: fieldinfo[hgt] = {}
    fieldinfo[hgt]['filename'] = narr3D
    fieldinfo[hgt]['fname'] = ['HGT_221_ISBL']
    fieldinfo[hgt]['vertical'] = int(re.search(r'\d+', hgt)[0]) # extract numeric part as integer
for var in ['wind', 'speed']:
    fieldinfo[var+'10m']['fname'] = ['U_GRD_221_HTGL','V_GRD_221_HTGL'] # different vertical coordinate
    fieldinfo[var+'10m']['filename'] = narrFlx
fieldinfo['bunkers']['fname'] = ['USTM_221_HTGY','VSTM_221_HTGY']
fieldinfo['bunkers']['filename'] = narrFlx
fieldinfo['bunkers']['arrow'] = True
fieldinfo['mslp']['fname'] = ['PRMSL_221_MSL']
fieldinfo['mslp']['filename'] = narrFlx
fieldinfo['mslp']['units'] = 'hPa'
fieldinfo['mslet'] = fieldinfo['mslp'].copy()
fieldinfo['mslet'] = ['MSLET_221_MSL']
fieldinfo['mucape']['fname'] = ['CAPE_221_SFC']
fieldinfo['pblh']['fname'] = ['HPBL_221_SFC']
fieldinfo['precipacc']['fname'] = ['RAINNC']
fieldinfo['rh_0deg'] = fieldinfo['rh700'] # thought about adding 'vertical' but there is no vertical dimension
fieldinfo['rh_0deg']['filename'] = narrFlx
fieldinfo['rh_0deg']['fname'] = ['R_H_221_0DEG']
fieldinfo['sh2'] = {'levels' : [0.2,0.5,1,2,4,8,12,14,16,18,20,22,24], 'cmap':fieldinfo['td2']['cmap'], 'fname'  : ['SPF_H_221_HTGL'], 'filename':narrFlx, 'vertical':2, 'units':'g/kg'},
fieldinfo['shr10_30']  = fieldinfo['speed850'].copy()
fieldinfo['shr10_500'] = fieldinfo['speed500'].copy()
fieldinfo['shr10_700'] = fieldinfo['speed700'].copy()
fieldinfo['shr10_850'] = fieldinfo['speed850'].copy()
fieldinfo['shr10_900'] = fieldinfo['speed900'].copy()
fieldinfo['shr10_925'] = fieldinfo['speed925'].copy()
fieldinfo['shr30_500'] = fieldinfo['speed500'].copy()
fieldinfo['shr30_700'] = fieldinfo['speed700'].copy()
fieldinfo['shrtrop'] = {'levels':np.array([2,5,10,15,20,30,50])*1e-3, 'cmap': fieldinfo['speed700']['cmap'], 
                        'filename': narrFlx, 'fname': ['VWSH_221_TRO'], 'vertical':'tropopause'} # shear at tropopause. https://www.emc.ncep.noaa.gov/mmb/rreanl/faq.html 
fieldinfo['shrlev1_trop'] = fieldinfo['shr10_700'].copy() # wind shear between tropopause and lowest model level
fieldinfo['srh'] = fieldinfo['srh1'].copy()
fieldinfo['srh']['fname'] = ['HLCY_221_HTGY']
fieldinfo['srh']['filename'] = narrFlx
fieldinfo['t2']['fname'] = ['TMP_221_SFC']
fieldinfo['t2']['units'] = 'degF'
fieldinfo['thetasfc'] = {  'levels' : np.arange(278,330,2), 'cmap': ['#eeeeee', '#dddddd', '#cccccc', '#aaaaaa']+readNCLcm('precip2_17lev')[3:-1], 'fname'  : ['POT_221_SFC'], 'filename': narrSfc}
fieldinfo['wvflux'] = {'fname':['WVUFLX_221_ISBY_acc3h','WVVFLX_221_ISBY_acc3h'],'filename':narrFlx,'arrow':True}
fieldinfo['wvfluxconv'] = {'fname':['WVCONV_221_ISBY_acc3h'],'filename':narrFlx,'levels':np.array(fieldinfo['precip']['levels'])*10,'cmap': readNCLcm('prcp_1')}
fieldinfo['wcflux'] = {'fname':['WCUFLX_221_ISBY_acc3h','WCVFLX_221_ISBY_acc3h'],'filename':narrFlx,'arrow':True}
fieldinfo['wcfluxconv'] = {'fname':['WCCONV_221_ISBY_acc3h'],'filename':narrFlx,'levels':np.array(fieldinfo['precip']['levels'])*2,'cmap': readNCLcm('prcp_1')}

#######################################################################
idir = "/glade/collections/rda/data/ds608.0/3HRLY/" # path to NARR
#######################################################################

# Get NARR file from tar file, convert to netCDF.
# Return netCDF filename.
def get(valid_time, targetdir='.', narrtype=narr3D, idir=idir):
    assert isinstance(valid_time, datetime.datetime) # make sure valid_time is a datetime object
    vartype, file_suffix = narrtype # 3D, clm, flx, pbl, or sfc 
    narr = targetdir + '/' + valid_time.strftime("merged_AWIP32.%Y%m%d%H."+file_suffix+".nc")
    # Convert to netCDF if netCDF file doesn't exist.
    if not os.path.exists(narr):
        narr_grb = narr[:-3] # without '.nc' suffix
        # Extract NARR grib file from tar file if it doesn't exist.
        if not os.path.exists(narr_grb):
            narrtars = glob.glob(idir + valid_time.strftime('%Y/NARR'+vartype+'_%Y%m_') + '*.tar')
            for n in narrtars:
                prefix, yyyymm, dddd = n.split('_')
                dd1 = datetime.datetime.strptime(yyyymm + dddd[0:2], '%Y%m%d')
                # NARR3D_201704_0103.tar starts on 1st at 0Z and ends just before 4th at 0Z. so add one to eday
                dd2 = datetime.datetime.strptime(yyyymm + dddd[2:4], '%Y%m%d') + datetime.timedelta(days=1)
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



def vertical(data, info, debug=False):
    if debug:
        print("narr.vertical(): info",info)

    # Define 'verttitle' attribute of data, no matter what.
    if 'verttitle' not in data.attrs:
        data.attrs['verttitle'] = ''

    if 'vertical' in info:
        vlevel = info['vertical']
        if hasattr(data.metpy, 'vertical'): 
            vertical, = data.metpy.coordinates('vertical')
            data = data.metpy.sel(vertical=vlevel)
        elif len(data.shape) <= 2: # If data has only 2 dimensions assume it has no vertical dimension (like tropopause-level, or max-wind-level)
            data.attrs['verttitle'] = info['vertical']
            if debug:
                print('narr.vertical(): assume no vertical dimension. setting verttitle=',info['vertical'])
            return data
        else:
            vertical = data.dims[0]
            assert 'lv_' in vertical
            if debug:
                print("narr.vertical(): metpy does not identify vertical coordinate. assume 1st dim is ("+vertical+")")
            vertical = data.coords[vertical]
            data = data.loc[vlevel]
        data.attrs['verttitle'] = str(vlevel) + vertical.units

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

