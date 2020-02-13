from fieldinfo import fieldinfo, readNCLcm
import os # for NCARG_ROOT
import numpy as np
import subprocess
import tarfile
import re
import pdb
import datetime
import glob


# Modify fieldinfo dictionary for MPAS. 
narrSfc = ('sfc', 'RS.sfc')
narrFlx = ('flx', 'RS.flx')
narr3D  = ('3D', '3D')
for v in fieldinfo:
    fieldinfo[v]['filename'] = narrSfc
winds = ['wind10m']
winds.extend(['wind'+plev for plev in ['200', '250', '300', '500', '700', '850', '925']])
for wind in winds:
    fieldinfo[wind] = {}
    fieldinfo[wind]['filename'] = narr3D
    fieldinfo[wind]['fname'] = ['U_GRD_221_ISBL','V_GRD_221_ISBL']
    fieldinfo[wind]['vertical'] = int(re.search(r'\d+', wind)[0]) # extract numeric part as integer (works for 'wind10m' too)
    fieldinfo[wind]['units'] = 'knots'
fieldinfo['wind10m']['fname'] = ['U_GRD_221_HTGL','V_GRD_221_HTGL']
fieldinfo['wind10m']['filename'] = narrFlx
fieldinfo['wind10m']['units'] = 'knots'
fieldinfo['wvflux'] = {'fname':['WVUFLX_221_ISBY_acc3h','WVVFLX_221_ISBY_acc3h'],'filename':narrFlx,'arrow':True}
fieldinfo['wvfluxconv'] = {'fname':['WVCONV_221_ISBY_acc3h'],'filename':narrFlx,'levels':np.array(fieldinfo['precip']['levels'])*10,'cmap': readNCLcm('prcp_1')}
fieldinfo['wcflux'] = {'fname':['WCUFLX_221_ISBY_acc3h','WCVFLX_221_ISBY_acc3h'],'filename':narrFlx,'arrow':True}
fieldinfo['wcfluxconv'] = {'fname':['WCCONV_221_ISBY_acc3h'],'filename':narrFlx,'levels':np.array(fieldinfo['precip']['levels'])*2,'cmap': readNCLcm('prcp_1')}
fieldinfo['mslp']['fname'] = ['PRMSL_221_MSL']
fieldinfo['mslp']['filename'] = narrFlx
fieldinfo['mslp']['units'] = 'hPa'
fieldinfo['mucape']['fname'] = ['CAPE_221_SFC']
fieldinfo['precipacc']['fname'] = ['RAINNC']
fieldinfo['t2']['fname'] = ['TMP_221_SFC']
fieldinfo['t2']['units'] = 'degF'

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


