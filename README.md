# ahijevyc

A personal library of Python modules for atmospheric-science work on NCAR systems: tropical cyclone tracks, SPC storm reports, GOES GLM lightning, MPAS output, forecast verification plots and map helpers. It's a single package, `ahijevyc`, that installs into any conda environment so scripts and notebooks can import it from anywhere.

```python
from ahijevyc import atcf, spc
from ahijevyc.fieldinfo import fieldinfo, readNCLcm
```

## Install

The package is installed in editable mode, so changes to the files here take effect right away in every environment that has it:

```
<env>/bin/pip install --no-deps --no-build-isolation -e ~/lib_python
```

`pyproject.toml` declares no dependencies on purpose. Each module imports what it needs, and the environments it's used in already have those packages. `--no-deps` keeps pip from upgrading anything in an environment.

Check that an environment sees it, from outside this directory so the local copy doesn't mask a missing install:

```
cd /tmp && <env>/bin/python -I -c "import ahijevyc.atcf; print(ahijevyc.atcf.__file__)"
```

## Modules

**Tropical cyclones**
- `atcf`: read, write and plot ATCF best-track and forecast files: track plots and legends, wind radii, intensity categories, storm-relative winds, warm-core checks, Knaff–Zehr pressure.
- `ibtracs`: download IBTrACS and convert storms to ATCF format; map between storm names and ATCF IDs.
- `ncl`: `steering_flow()` runs an NCL script to compute deep-layer steering flow and shear around a storm.

**Severe weather and lightning**
- `spc`: download, filter and plot SPC storm reports (including NCEI Storm Events and SPC's tropical-cyclone tornado list), grid them, and write them out for MET.
- `glm`: download GOES GLM L2 files from AWS and read and quality-control flashes, groups and events. Files go to `$GLMDIR` if set, otherwise `$SCRATCH`.
- `G211`: helpers for the NCEP 211 (80-km Lambert conformal) grid: projection bounds, grid-cell polygons and masks.
- `st4`: Stage IV precipitation: clean the data and mask points to a shapefile.

**Models**
- `mpas`: MPAS ensemble file lists, mesh unzipping, cell geometry, and TC vitals from MPAS diagnostic files on the native mesh.
- `fieldinfo`: the `fieldinfo` dictionary of plotting definitions (contour levels, colormaps, file variable names) for model fields, plus plot domains and colormap readers.
- `get_model_time`: valid and initialization times from model netCDF files.

**Plotting and statistics**
- `statisticplot`: ROC curves, reliability and performance diagrams, Brier skill score, and `limits()` for axis ranges of verification statistics.
- `myskewt`: skew-T and hodograph drawing (built on SHARPpy).
- `gridlines`: tick labels on Lambert conformal maps in cartopy.
- `nclcmaps`: NCL colormaps as matplotlib colormaps.
- `Fanglin`: `confidence_interval()`, the 95% confidence half-width of a sample mean using t-values for small samples, per a 2016 email from Fanglin.
- `cache`: an on-disk cache for `urllib` downloads.

## Notes

- Some modules have NCAR paths built in, such as the colormap directory in `fieldinfo.readcm` and data locations on `/glade`. They're meant for use on Derecho and Casper.
- This repo used to be `~/lib/python3.6`, where scripts found modules through symlinks and `sys.path`. Old code that does `import atcf` should now use `from ahijevyc import atcf`.
- The MPAS ensemble web-plotting code (`webplot.py`) isn't here; it's in [ahijevyc/MPASplot](https://github.com/ahijevyc/MPASplot).
