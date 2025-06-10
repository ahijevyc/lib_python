import collections
import datetime
import logging
import os
import pdb
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import s3fs
import xarray

ftype_dim = {"event": "number_of_events",
             "group": "number_of_groups", "flash": "number_of_flashes"}

# path to save GLM files
# Use environmental variable GLMDIR if it exists.
# Otherwise $SCRATCH
GLMDIR = os.getenv("GLMDIR", os.getenv("SCRATCH")) + "/" # s3fs.get requires '/'


def get_das(ds: xarray.DataArray) -> Tuple[dict, dict]:
    # dictionary of DataArrays associated with each ftype
    das = {}
    coords = {}
    for ftype, dim in ftype_dim.items():
        # list DataArrays with dim in them
        das[ftype] = [name for name in ds.data_vars if dim in ds[name].dims]
        # extend with list of coordinates with dim in them
        coords[ftype] = [name for name in ds.coords if dim in ds[name].dims]
    return das, coords


def download(start, end, bucket="noaa-goes16", product="GLM-L2-LCFA", odir=GLMDIR, clobber=False):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)  # allow strings
    fs = s3fs.S3FileSystem(anon=True)
    ofiles = []
    for hourly in pd.date_range(start=start, end=end, freq='1H'):
        # download all files in an hourly directory
        path = f'{bucket}/{product}/{hourly.year}/{hourly.timetuple().tm_yday:03.0f}/{hourly.hour:02.0f}'
        logging.info(f"fs.ls({path})")
        try:
            files = fs.ls(path)
        except FileNotFoundError:
            logging.error(f"{path} not found")
            sys.exit(1)

        parentpath = "/".join(path.split("/")[:-1])  # parent directory
        assert files, (
            f"No files match {path}\n"
            f"choices {fs.ls(parentpath)}"
        )

        logging.debug(f"{len(files)} files {odir}")
        for f in files:
            if get_GLM_timestamp(f) < start:
                logging.debug(f"{f} before start")
                continue
            if get_GLM_timestamp(f) >= end:
                logging.debug(f"{f} at or past end")
                continue

            if os.path.exists(odir+f) and os.path.getsize(odir+f) and not clobber:
                logging.debug(f"{f} already downloaded")
            else:
                if os.path.exists(odir+f) and os.path.getsize(odir+f) == 0:
                    logging.warning(f"{f} zero size. downloading again")
                # Avoid FileNotFoundError: [Errno 2] No such file or directory:
                os.makedirs(os.path.dirname(odir+f), exist_ok=True)
                fs.get_file(f, odir+f)  # TODO: catch s3fs FileNotFoundError
                logging.info(f"downloaded {f}")
            ofiles.append(odir+f)

    return ofiles


def get_GLM_timestamp(GLMfile):
    # extract s timestamp string from filename
    # https://docs.opendata.aws/noaa-goes16/cics-readme.html
    # <Filename> delineated by underscores ‘_’ and looks like this:
    # OR_ABI-L1b-RadF-M3C02_G16_s20171671145342_e20171671156109_c20171671156144.nc
    GLMtime = os.path.basename(GLMfile).split("_")[3]
    # convert to datetimes
    GLMtime = datetime.datetime.strptime(GLMtime, 's%Y%j%H%M%S0')
    return GLMtime


def group_ids_in_flash(ds, flash_id):
    group_parent_flash_ids = ds['group_parent_flash_id'].values
    group_ids = ds['group_id'][group_parent_flash_ids == flash_id]
    return group_ids


def flashOK(ds: xarray.Dataset) -> xarray.DataArray:
    """
    # Input
    #   DataSet
    # Returns
    #   good - Boolean DataArray after counting events (set to False where too few events).
    """
    
    # TODO. would it make sense to convert to 2-level MultiIndex with flash_id and group_id?
    event_parent_group_id_counter = collections.Counter(
        ds.event_parent_group_id.values)

    # Count number of events in each flash.
    # If there are less than 3 groups AND less than 4 events in those 3 groups, then
    # it's a bad flash. Change good[i] to False, where i is the index of the flash.
    good = ~ds.flash_id.isnull()
    for i, flash_id in enumerate(ds.flash_id.values):
        logging.debug(f'checking flash {flash_id}')
        group_ids = group_ids_in_flash(ds, flash_id).values
        if group_ids.size >= 3:
            continue
        nevents = 0
        for group_id in group_ids:
            nevents += event_parent_group_id_counter[group_id]
            if nevents >= 4:
                continue
        if nevents < 4:
            logging.debug(
                f'flash {flash_id} {len(group_ids)} groups {nevents} events')
            good[i] = False

    return good


def mask_bad_groups_and_flashes(ds):
    das, _ = get_das(ds)
    # TODO: Find groups with just one event.
    logging.debug("group QC")
    # If group_id is in event_parent_group_id (i.e. associated with at least one event) then it's OK.
    groupOK = ds.group_id.isin(ds.event_parent_group_id)
    # mention how many are not OK
    nempty = (~groupOK).sum().values
    if nempty:
        logging.info(f"{nempty} empty groups {ds.group_id[~groupOK].values}")
    ds[das["group"]] = ds[das["group"]].where(groupOK)

    # Remove flashes with small number of events.
    logging.debug("flash QC")
    ds[das["flash"]] = ds[das["flash"]].where(flashOK(ds))
    return ds


def read(GLMfiles, qc=0):
    """
    # Read and combine GLM datasets in GLMfiles.
    # Return combined xarray Dataset

    # Tried to concatenate with nco but time offset is a different multiple of 20 sec for each GLM file
    # so that time offset can be stored as a short integer.
    # ncrcat keeps just the first one and assumes it doesn't change.
    # so the times are all interweaved when you concatenate with ncrcat.
    # Also, the scale and offset may change. ncrcat warns if they do, BLAMMO! Don't use concatenated file.

    # Tried to concatenate nc files with xarray, but flash_ids (uint16) repeat after 65536 flashes (a couple hours)
    # How can you count # of groups in a flash if the same group_parent_flash_id may refer to different flashes?
    # Therefore apply quality control with each GLM file (product) individually.
    """
    concat_all = False
    if concat_all:
        ds = xarray.open_dataset(GLMfiles[0])
        das, coords = get_das(ds)
        # How many times is a flash_id used in these GLM files.
        flash_ids = np.empty(0, dtype=ds.flash_id.dtype)
        flash_time_offsets = np.empty(
            0, dtype=ds.flash_frame_time_offset_of_first_event.dtype)
        for GLMfile in GLMfiles:
            ds = xarray.open_dataset(GLMfile)
            flash_ids = np.concatenate([flash_ids, ds.flash_id])
            flash_time_offsets = np.concatenate(
                [flash_time_offsets, ds.flash_frame_time_offset_of_first_event])
            ds.close()
        logging.info(collections.Counter(flash_ids).most_common(1))
        for ftype, das in das.items():
            logging.info(f"concatenating {das}")
            ds[ftype] = xarray.concat([ds[das] for ds in [xarray.open_dataset(
                GLMfile) for GLMfile in GLMfiles]], dim, combine_attrs="drop_conflicts")

    ds = xarray.open_dataset(GLMfiles[0])
    # keep data variables and coordinates separate. Can QC data variables but Can't QC coordinates.
    das, coords = get_das(ds)

    # Combine GLM datasets
    # Dictionary of empty lists to extend with numpy arrays. Keys are all data variables and coordinates.
    ds_dict = dict()
    for ftype in das.keys():
        # Will concatenate data variables and coordinates
        for da in das[ftype] + coords[ftype]:
            ds_dict[da] = []

    for GLMfile in GLMfiles:
        ds = xarray.open_dataset(GLMfile)
        logging.debug(GLMfile)

        # Mask bad groups and flashes
        if qc >= 2:
            ds = mask_bad_groups_and_flashes(ds)

        for ftype in das.keys():
            # Drop masked data.
            ds = ds.dropna(dim=ftype_dim[ftype])
            # Extend combined data variables in ds_dict.
            for da in das[ftype] + coords[ftype]:
                ds_dict[da].extend(ds[da].values)
        ds.close()
        print('.', end='', flush=True)

    # Make data and coordinate dictionaries to construct combined Dataset.
    data_vars = {}
    ds_coords = {}
    for ftype in das.keys():
        for da in das[ftype]:
            data_vars[da] = xarray.DataArray(
                data=ds_dict[da], dims=ftype_dim[ftype], attrs=ds[da].attrs)
        for c in coords[ftype]:
            ds_coords[c] = (ftype_dim[ftype], ds_dict[c])
    ds = xarray.Dataset(data_vars=data_vars, coords=ds_coords, attrs=ds.attrs)

    # TODO: Check out other quality flags like flash_quality_flag:percent_degraded_due_to_flash_constituent_events_out_of_time_order_qf
    if qc >= 1:
        # Used to be in GLMfiles loop, but it is slow. Moreover, we don't need events per group or flash.
        for ftype in ["group", "flash"]:
            # We can't use drop=True here because we would change dimension size in the middle of the data array loop.
            # i.e. ValueError: arguments without labels ... different dimension sizes...
            ds[das[ftype]] = ds[das[ftype]].where(
                ds[ftype+"_quality_flag"] == 0)
            ds = ds.dropna(dim=ftype_dim[ftype])

    return ds


if __name__ == "__main__":
    if sys.argv[1]:
        sdate = pd.to_datetime(sys.argv[1])
        edate = pd.to_datetime(sys.argv[2])
    else:
        sdate = datetime.datetime.now() - datetime.timedelta(minutes=5)
        edate = datetime.datetime.now()
    download(sdate, edate)
