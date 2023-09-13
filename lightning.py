"""
get lightning observations
"""
import logging
import os
import pandas as pd
from pathlib import Path
from scipy import spatial
import xarray
import G211

def get_obs(valid_start, valid_end, obsvar, twin, rptdist):
    if obsvar in ["cg", "ic", "cg.ic"]:
        # Add up lighting counts in 30-minute blocks spanning [valid_start, valid_end).
        # wbug lightning counts are indexed by time_coverage_start, so
        # the slice starts at valid_start and ends 30 minutes prior to valid_end.
        # Don't expect to find all 30-min blocks within valid time range.
        # If there is no lightning, that 30-min block will be missing.
        # So select with a time slice, not a list of 30-min blocks.
        # Fill missing blocks with the average of non-missing blocks
        # in the time window.
        wbugtimes = slice(valid_start, valid_end - pd.Timedelta(minutes=30))
        cgds = xarray.open_dataset(Path("wbug_lightning") / f"flash.{rptdist}km_30min.nc")
        if cgds.sel(time_coverage_start=wbugtimes).time_coverage_start.size == 0:
            logging.warning(f"no wbug times {wbugtimes}")
            return None
        ds = (
            cgds.sel(time_coverage_start=wbugtimes).mean(dim="time_coverage_start")
            * (valid_end - valid_start)
            / pd.Timedelta(minutes=30)
        )
        ds["cg.ic"] = ds.cg + ds.ic
    elif obsvar == "flashes":
        # {twin}-hour blocks spanning valid time range
        # Blocks are indexed by their time coverage center,
        # so the start index is 1/2 twin hours after valid_start
        # and the end index is 1/2 twin hour prior to valid_end.
        glmtimes = pd.date_range(
            start=valid_start + pd.Timedelta(hours=twin / 2),
            end=valid_end - pd.Timedelta(hours=twin / 2),
            freq=f"{twin}H",
        )
        ifiles = [
            os.path.join(
                "GLM",
                t.strftime("%Y"),
                t.strftime("%Y%m%d_%H%M") + f".glm_{rptdist}km_{twin}hr.nc",
            )
            for t in glmtimes
        ]
        ifiles = [x for x in ifiles if os.path.exists(x)]
        if len(ifiles) == 0:
            logging.warning(f"no GLM for {glmtimes}")
            return None
        ds = xarray.open_mfdataset(ifiles).sum(dim="time")
    else:
        logging.error(f"unexpected obsvar {obsvar}")

    if rptdist == 20:
        # in the case of rptdist=20 return pts on coarser grid 
        # find nearest indices of 40-km grid.
        lats = G211.x2().lat.ravel()
        lons = G211.x2().lon.ravel()
        x = G211.lon.ravel()
        y = G211.lat.ravel()
        tree = spatial.KDTree(list(zip(lons,lats)))
        dist, indices = tree.query(list(zip(x,y)))
        ltg_sum_coarse = ds.stack(pt=("y","x")).isel(pt=indices)

        # replace 40km grid coordinates with 80km grid coordinates
        c = ltg_sum_coarse.coords
        c.update(G211.mask.stack(pt=("y","x")).coords)
        ds = ltg_sum_coarse.assign_coords(c).unstack(dim="pt")


    assert obsvar in ds, f"{obsvar} not in obsgdf. did you assign the correct dataset?"

    return ds
