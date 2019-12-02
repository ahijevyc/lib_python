from copy import copy
import cartopy.crs as ccrs
import numpy as np
import shapely.geometry as sgeom
import matplotlib.patches
import matplotlib.path

"""
Copied from https://gist.github.com/ajdawson/dd536f786741e987ae4e

"""

def find_side(ls, side):
    """
    Given a shapely LineString which is assumed to be rectangular, return the
    line corresponding to a given side of the rectangle.
    
    """
    minx, miny, maxx, maxy = ls.bounds
    points = {'left': [(minx, miny), (minx, maxy)],
              'right': [(maxx, miny), (maxx, maxy)],
              'bottom': [(minx, miny), (maxx, miny)],
              'top': [(minx, maxy), (maxx, maxy)],}
    return sgeom.LineString(points[side])


def lambert_xticks(ax, ticks, **kwdict):
    """Draw ticks on the bottom x-axis of a Lambert Conformal projection."""
    te = lambda xy: xy[0]
    lc = lambda t, n, b: np.vstack((np.zeros(n) + t, np.linspace(b[2], b[3], n))).T
    xticks, xticklabels = _lambert_ticks(ax, ticks, 'bottom', lc, te)
    ax.xaxis.tick_bottom()
    ax.set_xticks(xticks)
    ax.set_xticklabels([ax.xaxis.get_major_formatter()(xtick) for xtick in xticklabels], fontdict=kwdict)
    

def lambert_yticks(ax, ticks, **kwdict):
    """Draw ticks on the left y-axis of a Lambert Conformal projection."""
    te = lambda xy: xy[1]
    lc = lambda t, n, b: np.vstack((np.linspace(b[0], b[1], n), np.zeros(n) + t)).T
    yticks, yticklabels = _lambert_ticks(ax, ticks, 'left', lc, te)
    ax.yaxis.tick_left()
    ax.set_yticks(yticks)
    ax.set_yticklabels([ax.yaxis.get_major_formatter()(ytick) for ytick in yticklabels], fontdict=kwdict)

def _lambert_ticks(ax, ticks, tick_location, line_constructor, tick_extractor):
    """Get the tick locations and labels for an axis of a Lambert Conformal projection."""
    outline_patch = sgeom.LineString(ax.outline_patch.get_path().vertices.tolist())
    axis = find_side(outline_patch, tick_location)
    n_steps = 30
    extent = ax.get_extent(ccrs.PlateCarree())
    _ticks = []
    for t in ticks:
        xy = line_constructor(t, n_steps, extent)
        proj_xyz = ax.projection.transform_points(ccrs.Geodetic(), xy[:, 0], xy[:, 1])
        xyt = proj_xyz[..., :2]
        ls = sgeom.LineString(xyt.tolist())
        locs = axis.intersection(ls)
        if not locs:
            tick = [None]
        else:
            tick = tick_extractor(locs.xy)
        _ticks.append(tick[0])
    # Remove ticks that aren't visible:    
    ticklabels = copy(ticks)
    while True:
        try:
            index = _ticks.index(None)
        except ValueError:
            break
        _ticks.pop(index)
        ticklabels.pop(index)
    return _ticks, ticklabels

def box(box, alpha=0.3):
    vertices = [
        (box[0],box[2]),
        (box[1],box[2]),
        (box[1],box[3]),
        (box[0],box[3]),
        (box[0],box[2])]
    rect = matplotlib.path.Path(vertices, closed=True)
    curve_resolution=75
    # cartopy GeoAxes only breaks up lines every 30 degrees or so.
    # This is too crude; lines along parallels (in Lambert Conformal projection)
    # end up being crooked and not parallel. You must break up the line segment more frequently.
    patch = matplotlib.patches.PathPatch(rect.interpolated(curve_resolution),alpha=alpha,facecolor="grey",
            label="verification domain", transform=ccrs.Geodetic())
    return patch



# Update latitude and longitude grid line labels based on extent
# Show those within extent; hide those outside extent
def update_label_visibility(gl, debug=False):
    # gl is gridliner object
    west, east, south, north = gl.axes.get_extent()
    if debug:
        print("extent=",extent)
    for label in gl.ylabel_artists:
        x, y = label.get_position()
        if y < south or y > north:
            if debug:
                print("hiding",label)
            label.set_visible(False)
        else:
            if debug:
                print("showing",label)
            label.set_visible(True)
    for label in gl.xlabel_artists:
        x, y = label.get_position()
        if x < west or x > east:
            if debug:
                print("hiding",label)
            label.set_visible(False)
        else:
            if debug:
                print("showing",label)
            label.set_visible(True)


